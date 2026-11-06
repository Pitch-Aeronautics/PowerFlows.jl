# The restore walk after a failed post-snap solve, and the control solve's time limit (T156) and
# inner-solve limit (T160).

# IEEE 14 (PSB `c_sys14`) with eight discrete switched shunts and one FACTS, solved under discrete
# control; then every withdrawal times `overload`, which no device setting can carry, and shunt
# `moved` put 2.4 steps off its enrolment value so that its snap has a parameter to walk back.
function _restore_walk_fixture(; overload = 4.0, moved = 3)
    sys = _make_ieee14_scaled_load_system(1.0)
    for b in (4, 5, 7, 9, 10, 11, 13, 14)
        _add_switched_shunt!(sys, b; voltage_setpoint = 1.02, deadband = 0.02)
    end
    _add_facts_shunt!(sys, 12; voltage_setpoint = 1.05)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @assert solve_power_flow!(data)
    set = PowerFlows.get_controlled_devices(data)
    d = set.shunts[moved]
    PowerFlows.apply_parameter!(d, data, d.initial + 2.4 * minimum(abs, d.block_dB), 1)
    data.bus_active_power_withdrawals[:, 1] .*= overload
    data.bus_reactive_power_withdrawals[:, 1] .*= overload
    return pf, data, set
end

_all_parameters(set) = [PowerFlows.current_parameter(d)
                        for d in vcat(set.taps, set.shunts, set.facts, set.phase_shifters)]

@testset "restore walk: a failed start is not solved again, with the same result (T156)" begin
    kw = (; maxIterations = 8)
    # The walk as it ran before T156: one `_restore_one!` per device, none sharing a record.
    pfB, dataB, setB = _restore_walk_fixture()
    pre = PowerFlows._snapshot_state(dataB, 1)
    groups = (setB.taps, setB.shunts, setB.facts, setB.phase_shifters)
    conts = [PowerFlows._snap_device_group!(g, dataB, 1) for g in groups]
    setB.inner_solves[] = 0
    okB = Logging.with_logger(Logging.NullLogger()) do
        PowerFlows._ctrl_solve!(pfB, dataB, 1; kw...) && return true
        PowerFlows._restore_state!(dataB, 1, pre)
        ok = true
        scratch = PowerFlows._snapshot_state(dataB, 1)
        for (g, c) in zip(groups, conts), (i, d) in enumerate(g)
            ok &= PowerFlows._restore_one!(d, dataB, 1, c[i], pfB, scratch; kw...)
        end
        return ok
    end
    solves_old = setB.inner_solves[]

    pfA, dataA, setA = _restore_walk_fixture()
    setA.inner_solves[] = 0
    okA = Logging.with_logger(Logging.NullLogger()) do
        PowerFlows.snap_and_restore!(pfA, dataA, setA, 1, PowerFlows._snapshot_state(dataA, 1);
            kw...)
    end
    solves_new = setA.inner_solves[]

    n_dev = length(setA.shunts) + length(setA.facts)
    noparam = [abs(conts[gi][i] - PowerFlows.current_parameter(d)) < PowerFlows._param_tol(d)
               for (gi, g) in enumerate(groups) for (i, d) in enumerate(g)]
    @test n_dev == 9
    @test count(noparam) == 8          # the moved shunt is the only device with a walk
    @test !okB                          # the snapped network fails for a reason no device explains
    @test okA == okB
    @test !dataA.converged[1]
    # same device settings and the same state, to the bit
    @test _all_parameters(setA) == _all_parameters(setB)
    @test dataA.bus_magnitude[:, 1] == dataB.bus_magnitude[:, 1]
    @test dataA.bus_angles[:, 1] == dataB.bus_angles[:, 1]
    @test dataA.bus_type[:, 1] == dataB.bus_type[:, 1]
    @test dataA.bus_reactive_power_injections[:, 1] == dataB.bus_reactive_power_injections[:, 1]
    # Before: the post-snap solve, one solve per unmoved device, and the moved shunt's full move
    # and bisection. After: shunt 1 solves and shunt 2 reuses its failure; the moved shunt (3)
    # walks as before and clears the record, so shunt 4 solves again and the four shunts after it
    # and the FACTS reuse that failure.
    walk_moved = solves_old - 1 - count(noparam)
    @test walk_moved >= 2
    @test solves_new == 1 + 1 + walk_moved + 1
    @test solves_old - solves_new == count(noparam) - 2
end

@testset "restore walk: a solve that converges is not reused (T156)" begin
    # Without the overload the snapped network solves and the walk does not run; force the walk
    # on a feasible network by a failing post-snap cap, then let the devices solve at the full
    # cap: the first unmoved device converges, so every later one solves (from a converged state).
    pf, data, set = _restore_walk_fixture(; overload = 1.0)
    failed = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    scratch = PowerFlows._snapshot_state(data, 1)
    set.inner_solves[] = 0
    oks = Logging.with_logger(Logging.NullLogger()) do
        [PowerFlows._restore_one!(d, data, 1, PowerFlows.current_parameter(d), pf, scratch,
            failed) for d in vcat(set.shunts[1:2], set.facts)]
    end
    @test all(oks)
    @test set.inner_solves[] == 3
    @test isnothing(failed[])
end

@testset "restore walk: area interchange state written by a failed solve blocks reuse (T156)" begin
    pf, data, set = _restore_walk_fixture()
    d = set.shunts[1]
    scratch = PowerFlows._snapshot_state(data, 1)
    # a recorded failed start whose area ΔP differs from the data's: the device solves
    failed = Ref{Union{Nothing, Vector{Float64}}}([1.0])
    set.inner_solves[] = 0
    ok = Logging.with_logger(Logging.NullLogger()) do
        PowerFlows._restore_one!(d, data, 1, PowerFlows.current_parameter(d), pf, scratch,
            failed; maxIterations = 8)
    end
    @test !ok
    @test set.inner_solves[] == 1
    @test failed[] == Float64[]         # no area enrolled: the record is the empty column
    # the same record now matches: no solve
    ok2 = PowerFlows._restore_one!(d, data, 1, PowerFlows.current_parameter(d), pf, scratch,
        failed; maxIterations = 8)
    @test !ok2
    @test set.inner_solves[] == 1
end

@testset "control time limit: not converged, no solve after the limit, the reason recorded (T156)" begin
    # A solve that runs the full continuation (four discrete shunts and a FACTS), first unlimited.
    sys = _make_ieee14_scaled_load_system(1.0)
    for b in (9, 10, 13, 14)
        _add_switched_shunt!(sys, b; voltage_setpoint = 1.04)
    end
    _add_facts_shunt!(sys, 12; voltage_setpoint = 1.05)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    ref = PowerFlowData(pf, sys)
    @test solve_power_flow!(ref)
    @test !PowerFlows.get_control_time_limit_reached(ref)
    n_ref = PowerFlows.get_control_inner_solve_count(ref)
    @test n_ref > 5

    # a generous limit changes nothing
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data; control_time_limit = 3600.0)
    @test !PowerFlows.get_control_time_limit_reached(data)
    @test PowerFlows.get_control_inner_solve_count(data) == n_ref
    @test data.bus_magnitude[:, 1] == ref.bus_magnitude[:, 1]
    @test _all_parameters(PowerFlows.get_controlled_devices(data)) ==
          _all_parameters(PowerFlows.get_controlled_devices(ref))
    @test isinf(PowerFlows.get_controlled_devices(data).deadline[])   # disarmed after the call

    # a limit already passed: no inner solve starts
    data = PowerFlowData(pf, sys)
    conv = @test_logs (:warn, r"time limit of 0.0 s passed") match_mode = :any begin
        PowerFlows._ac_power_flow_with_area_relax!(data, pf, 1; control_time_limit = 0.0)
    end
    @test !conv
    @test !data.converged[1]
    @test PowerFlows.get_control_time_limit_reached(data)
    @test PowerFlows.get_control_inner_solve_count(data) == 0
    @test isinf(PowerFlows.get_controlled_devices(data).deadline[])

    # a limit that passes mid-continuation: stops within one inner solve of it
    d2 = PowerFlowData(pf, sys)
    t = @elapsed solve_power_flow!(d2)                  # one full solve, compiled
    lim = t / 4
    data = PowerFlowData(pf, sys)
    conv = Logging.with_logger(Logging.NullLogger()) do
        PowerFlows._ac_power_flow_with_area_relax!(data, pf, 1; control_time_limit = lim)
    end
    if PowerFlows.get_control_time_limit_reached(data)
        @test !conv
        @test PowerFlows.get_control_inner_solve_count(data) < n_ref
    else
        @test conv        # a fast machine finished inside a quarter of a timed solve: no stop
    end

    # the next call without a limit clears the reason and solves as before
    @test solve_power_flow!(data)
    @test !PowerFlows.get_control_time_limit_reached(data)
end

@testset "control solve limit: a count, so it stops at the same inner solve every time (T160)" begin
    # the T156 fixture: four discrete shunts and a FACTS, the full continuation
    sys = _make_ieee14_scaled_load_system(1.0)
    for b in (9, 10, 13, 14)
        _add_switched_shunt!(sys, b; voltage_setpoint = 1.04)
    end
    _add_facts_shunt!(sys, 12; voltage_setpoint = 1.05)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    ref = PowerFlowData(pf, sys)
    @test solve_power_flow!(ref)
    n_ref = PowerFlows.get_control_inner_solve_count(ref)
    @test n_ref > 5
    @test PowerFlows.get_control_solves_started(ref) == n_ref       # counted with no limit armed
    @test !PowerFlows.get_control_solve_limit_reached(ref)

    # a limit the solve does not reach changes nothing, and is disarmed after the call
    data = PowerFlowData(pf, sys)
    @test PowerFlows._ac_power_flow_with_area_relax!(data, pf, 1; control_solve_limit = n_ref)
    @test !PowerFlows.get_control_solve_limit_reached(data)
    @test PowerFlows.get_control_solves_started(data) == n_ref
    @test data.bus_magnitude[:, 1] == ref.bus_magnitude[:, 1]
    @test _all_parameters(PowerFlows.get_controlled_devices(data)) ==
          _all_parameters(PowerFlows.get_controlled_devices(ref))
    @test PowerFlows.get_controlled_devices(data).solve_limit[] == typemax(Int)

    # 0 starts none
    data = PowerFlowData(pf, sys)
    conv = @test_logs (:warn, r"inner-solve limit of 0 was used up") match_mode = :any begin
        PowerFlows._ac_power_flow_with_area_relax!(data, pf, 1; control_solve_limit = 0)
    end
    @test !conv && !data.converged[1]
    @test PowerFlows.get_control_solve_limit_reached(data)
    @test !PowerFlows.get_control_time_limit_reached(data)
    @test PowerFlows.get_control_solves_started(data) == 0

    # a limit inside the continuation stops there, at the same count and the same state each time
    k = n_ref ÷ 2
    runs = map(1:2) do _
        d = PowerFlowData(pf, sys)
        c = Logging.with_logger(Logging.NullLogger()) do
            PowerFlows._ac_power_flow_with_area_relax!(d, pf, 1; control_solve_limit = k)
        end
        (c, PowerFlows.get_control_solves_started(d), PowerFlows.get_control_solve_limit_reached(d),
            copy(d.bus_magnitude[:, 1]), _all_parameters(PowerFlows.get_controlled_devices(d)))
    end
    for r in runs
        @test !r[1] && r[2] == k && r[3]
    end
    @test runs[1][4] == runs[2][4] && runs[1][5] == runs[2][5]

    # with a generous time limit beside it, the count decides; the next unlimited call solves as before
    data = PowerFlowData(pf, sys)
    conv = Logging.with_logger(Logging.NullLogger()) do
        PowerFlows._ac_power_flow_with_area_relax!(data, pf, 1; control_solve_limit = k,
            control_time_limit = 3600.0)
    end
    @test !conv && PowerFlows.get_control_solve_limit_reached(data) &&
          !PowerFlows.get_control_time_limit_reached(data)
    @test isinf(PowerFlows.get_controlled_devices(data).deadline[])
    @test solve_power_flow!(data)
    @test !PowerFlows.get_control_solve_limit_reached(data)
end
