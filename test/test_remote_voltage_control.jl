# GAP-015: generators regulating a remote bus (PSS/E IREG), sharing its reactive power by
# RMPCT. Synthetic system: REF(1) ─ X(3) ─ 5, loads at X and 5; plants on buses 2 and 4 tied to
# X, and on bus 6 behind bus 5 (not X's neighbour). Each `units` entry is a NamedTuple
# (bus, ireg, rmpct, qmin, qmax[, vs, key]); `key = :psse` writes the fields in `ext["psse"]`
# in lower case, as the ingest does, else in `ext` (PowerSystems' v33 reader's keys).
function _rv_system(; units = [(bus = 2, ireg = 3, rmpct = 100.0, qmin = -3.0, qmax = 3.0)],
    x_type = ACBusTypes.PQ, x_vm = 1.0, vs = 1.02, shunt = nothing, tap = false)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, x_type, 230, x_vm, 0.0)
    b5 = _add_simple_bus!(sys, 5, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_line!(sys, b1, b3, 0.01, 0.15, 0.0)
    if tap
        add_component!(sys, TapTransformer(; name = "t35", available = true,
            active_power_flow = 0.0, reactive_power_flow = 0.0,
            arc = Arc(; from = b3, to = b5), r = 0.005, x = 0.08,
            primary_shunt = 0.0 + 0.0im, tap = 1.0, rating = 1.0, base_power = 100.0,
            control_objective = PSY.TransformerControlObjective.VOLTAGE,
            voltage_setpoint = 1.03))
    else
        _add_simple_line!(sys, b3, b5, 0.01, 0.1, 0.0)
    end
    _add_simple_load!(sys, b3, 80.0, 50.0)
    _add_simple_load!(sys, b5, 20.0, 10.0)
    for u in units
        b = get_bus(sys, u.bus)
        if isnothing(b)
            b = _add_simple_bus!(sys, u.bus, ACBusTypes.PV, 230, 1.0, 0.0)
            _add_simple_line!(sys, b, get_bus(sys, u.bus == 6 ? 5 : 3), 0.005, 0.08, 0.0)
        end
        g = _add_simple_thermal_standard!(sys, b, 0.3, 0.0)
        set_reactive_power_limits!(g, (min = u.qmin, max = u.qmax))
        v = get(u, :vs, vs)
        if get(u, :key, :ext) == :psse
            get_ext(g)["psse"] = Dict{String, Any}("ireg" => u.ireg, "rmpct" => u.rmpct,
                "vs" => v)
        else
            get_ext(g)["IREG"] = u.ireg
            get_ext(g)["RMPCT"] = u.rmpct
            isnothing(v) || (get_ext(g)["VS"] = v)
        end
    end
    if !isnothing(shunt)
        add_component!(sys, SwitchedAdmittance(; name = "X_Z2", available = true, bus = b3,
            Y = 0.1im, initial_status = [0], number_of_steps = [1], Y_increase = [1.0im],
            admittance_limits = (min = shunt, max = shunt),
            control_mode = PSY.SwitchedAdmittanceControlMode.CONTINUOUS_VOLTAGE,
            regulated_bus_number = 0,
            ext = Dict{String, Any}("MODSW" => 2, "ADJM" => 0, "RMPCT" => 100.0)))
    end
    return sys
end

function _rv_solve(; q = false, control = false, remote = true, solver = NewtonRaphsonACPowerFlow,
    kw...)
    pf = ACPolarPowerFlow{solver}(; check_reactive_power_limits = q,
        control_discrete_devices = control, remote_voltage_control = remote)
    data = PowerFlowData(pf, _rv_system(; kw...))
    @test solve_power_flow!(data)
    return data, PF.get_bus_lookup(data)
end

_rv_q(data, bl, n) = data.bus_reactive_power_injections[bl[n], 1]
_rv_v(data, bl, n) = data.bus_magnitude[bl[n], 1]

const _RV_AB = [(bus = 2, ireg = 3, rmpct = 60.0, qmin = -3.0, qmax = 3.0),
    (bus = 4, ireg = 3, rmpct = 40.0, qmin = -3.0, qmax = 3.0)]

@testset "remote voltage control: data contract and enrolment (GAP-015)" begin
    sys = _rv_system(; units = [(bus = 2, ireg = 3, rmpct = 60.0, qmin = -1.0, qmax = 1.0,
        key = :psse, vs = 1.015)])
    g = only(get_components(ThermalStandard, sys))
    @test PF._generator_remote_bus(g) == 3
    @test PF._generator_rmpct(g) == 60.0
    @test PF._generator_remote_setpoint(g) == 1.015
    delete!(get_ext(g), "psse")
    @test PF._generator_rmpct(g) == 100.0              # PSS/E's default
    @test isnothing(PF._generator_remote_setpoint(g))
    get_ext(g)["RMPCT"] = 0.0
    @test (@test_logs (:warn, r"not positive") PF._generator_rmpct(g)) == 100.0

    data = PowerFlowData(ACPolarPowerFlow(), _rv_system())
    bl = PF.get_bus_lookup(data)
    grp = only(PF.get_remote_voltage_groups(data))
    @test grp.regulated_number == 3 && grp.member_numbers == [2]
    @test grp.vset == 1.02                               # VS
    @test data.bus_type[bl[3], 1] == ACBusTypes.PV && data.bus_magnitude[bl[3], 1] == 1.02
    @test data.bus_type[bl[2], 1] == ACBusTypes.PQ
    @test PF.get_reported_bus_type(data, bl[2], 1) == ACBusTypes.PV
    @test PF.get_reported_bus_type(data, bl[3], 1) == ACBusTypes.PQ
    # Without VS (PowerSystems' v33 reader drops it): the regulated bus's stored |V|.
    data = PowerFlowData(ACPolarPowerFlow(), _rv_system(; vs = nothing, x_vm = 1.01))
    @test only(PF.get_remote_voltage_groups(data)).vset == 1.01
    # Off, or the fast decoupled solver: every unit holds its own terminal.
    for pf in (ACPolarPowerFlow(; remote_voltage_control = false),
        ACRectangularPowerFlow())
        data = PowerFlowData(pf, _rv_system())
        @test isempty(PF.get_remote_voltage_groups(data))
        @test data.bus_type[PF.get_bus_lookup(data)[2], 1] == ACBusTypes.PV
    end
    data = @test_logs (:warn, r"not supported by the fast decoupled") match_mode = :any PowerFlowData(
        ACPolarPowerFlow{FastDecoupledXB}(), _rv_system())
    @test isempty(PF.get_remote_voltage_groups(data))
    # A reference bus is not regulated remotely.
    data = @test_logs (:warn, r"reference bus") match_mode = :any PowerFlowData(
        ACPolarPowerFlow(), _rv_system(; units = [(bus = 2, ireg = 1, rmpct = 100.0,
            qmin = -3.0, qmax = 3.0)]))
    @test isempty(PF.get_remote_voltage_groups(data))
    # A chain (a plant regulating a bus whose own plant regulates another) is cut.
    data = @test_logs (:warn, r"other plants regulate its own bus") match_mode = :any PowerFlowData(
        ACPolarPowerFlow(), _rv_system(; units = [
            (bus = 2, ireg = 4, rmpct = 100.0, qmin = -3.0, qmax = 3.0),
            (bus = 4, ireg = 3, rmpct = 100.0, qmin = -3.0, qmax = 3.0)]))
    grp = only(PF.get_remote_voltage_groups(data))
    @test grp.regulated_number == 4 && grp.member_numbers == [4, 2]
end

@testset "remote voltage control: one unit holds the remote bus, its terminal free (GAP-015)" begin
    data, bl = _rv_solve()
    @test _rv_v(data, bl, 3) ≈ 1.02 atol = 1e-9
    v2 = _rv_v(data, bl, 2)
    @test abs(v2 - 1.0) > 1e-3                           # no longer its stored |V|
    # The same state solves with the unit holding its terminal at that |V|.
    sys = _rv_system()
    set_magnitude!(get_bus(sys, 2), v2)
    ref = PowerFlowData(ACPolarPowerFlow(; remote_voltage_control = false), sys)
    @test solve_power_flow!(ref)
    @test ref.bus_magnitude[:, 1] ≈ data.bus_magnitude[:, 1] atol = 1e-8
    @test ref.bus_reactive_power_injections[bl[2], 1] ≈ _rv_q(data, bl, 2) atol = 1e-8
    # The ingest's keys (ext["psse"], lower case) give the same solve.
    d2, _ = _rv_solve(; units = [(bus = 2, ireg = 3, rmpct = 100.0, qmin = -3.0, qmax = 3.0,
        key = :psse)])
    @test d2.bus_magnitude[:, 1] ≈ data.bus_magnitude[:, 1] atol = 1e-10
    # Trust region and Levenberg-Marquardt solve it too.
    for s in (TrustRegionACPowerFlow, LevenbergMarquardtACPowerFlow)
        d, _ = _rv_solve(; solver = s)
        @test d.bus_magnitude[:, 1] ≈ data.bus_magnitude[:, 1] atol = 1e-6
    end
end

@testset "remote voltage control: two units share by RMPCT (GAP-015)" begin
    data, bl = _rv_solve(; units = _RV_AB)
    @test _rv_v(data, bl, 3) ≈ 1.02 atol = 1e-9
    @test _rv_q(data, bl, 2) / _rv_q(data, bl, 4) ≈ 60 / 40 rtol = 1e-9
    # A plant behind another bus (not X's neighbour) shares the same way.
    far = [_RV_AB[1], (bus = 6, ireg = 3, rmpct = 40.0, qmin = -3.0, qmax = 3.0)]
    data, bl = _rv_solve(; units = far)
    @test _rv_v(data, bl, 3) ≈ 1.02 atol = 1e-9
    @test _rv_q(data, bl, 2) / _rv_q(data, bl, 6) ≈ 60 / 40 rtol = 1e-9
    # X with its own plant: X's plant shares with the remote one, X held at its own setpoint.
    own = [(bus = 3, ireg = 0, rmpct = 50.0, qmin = -3.0, qmax = 3.0),
        (bus = 2, ireg = 3, rmpct = 50.0, qmin = -3.0, qmax = 3.0)]
    data, bl = _rv_solve(; units = own, x_type = ACBusTypes.PV, x_vm = 1.025, vs = 1.025)
    grp = only(PF.get_remote_voltage_groups(data))
    @test grp.member_numbers == [3, 2] && grp.vset == 1.025
    @test _rv_v(data, bl, 3) ≈ 1.025 atol = 1e-9
    @test _rv_q(data, bl, 3) ≈ _rv_q(data, bl, 2) rtol = 1e-9
end

@testset "remote voltage control: a unit at its limit drops out, then all saturate (GAP-015)" begin
    free, fbl = _rv_solve(; units = _RV_AB, q = true)
    qa, qb = _rv_q(free, fbl, 2), _rv_q(free, fbl, 4)
    @test qa > 0 && qb > 0
    # A capped at half its share: it sits at Qmax, B carries the rest, X still on setpoint.
    cap = qa / 2
    units = [merge(_RV_AB[1], (qmax = cap,)), _RV_AB[2]]
    data, bl = _rv_solve(; units = units, q = true)
    @test _rv_v(data, bl, 3) ≈ 1.02 atol = 1e-9
    @test _rv_q(data, bl, 2) ≈ cap atol = 1e-9
    @test _rv_q(data, bl, 4) > qb + 0.9 * (qa - cap)
    @test data.bus_type[bl[3], 1] == ACBusTypes.PV
    st = PF.get_remote_voltage_status(data)[3]
    @test [s for (_, _, s) in st] == [1, 0]
    @test PF.get_reported_bus_type(data, bl[2], 1) == ACBusTypes.PQ
    @test PF.get_reported_bus_type(data, bl[4], 1) == ACBusTypes.PV
    # Both capped: X becomes PQ below its setpoint, each unit at its Qmax.
    units = [merge(_RV_AB[1], (qmax = cap,)), merge(_RV_AB[2], (qmax = qb / 2,))]
    data, bl = _rv_solve(; units = units, q = true)
    @test data.bus_type[bl[3], 1] == ACBusTypes.PQ
    @test _rv_v(data, bl, 3) < 1.02 - 1e-3
    @test _rv_q(data, bl, 2) ≈ cap atol = 1e-9
    @test _rv_q(data, bl, 4) ≈ qb / 2 atol = 1e-9
    # The other side: at 0.95 the shares are below both floors, so both sit at Qmin and X
    # ends above its setpoint.
    d95, bl = _rv_solve(; units = _RV_AB, q = true, vs = 0.95)
    @test 0.0 < _rv_q(d95, bl, 2) < 0.25 && 0.0 < _rv_q(d95, bl, 4) < 0.25
    low = [merge(u, (qmin = 0.25,)) for u in _RV_AB]
    data, bl = _rv_solve(; units = low, q = true, vs = 0.95)
    @test data.bus_type[bl[3], 1] == ACBusTypes.PQ
    @test _rv_v(data, bl, 3) > 0.95 + 1e-3
    @test _rv_q(data, bl, 2) ≈ 0.25 atol = 1e-9
    @test _rv_q(data, bl, 4) ≈ 0.25 atol = 1e-9
    @test [s for (_, _, s) in PF.get_remote_voltage_status(data)[3]] == [-1, -1]
end

@testset "remote voltage control: a fixed unit rejoins, a saturated group returns (GAP-015)" begin
    free, fbl = _rv_solve(; units = _RV_AB, q = true)
    qa = _rv_q(free, fbl, 2)
    units = [merge(_RV_AB[1], (qmax = qa / 2,)), _RV_AB[2]]
    data, bl = _rv_solve(; units = units, q = true)
    # Its bound raised past its share: the check frees it, and the solve is the free one.
    data.remote_voltage_control.q_bounds[1] = (-3.0, 3.0)
    @test !PF._check_q_limit_bounds!(data, 1)
    @test [s for (_, _, s) in PF.get_remote_voltage_status(data)[3]] == [0, 0]
    @test solve_power_flow!(data)
    @test data.bus_magnitude[:, 1] ≈ free.bus_magnitude[:, 1] atol = 1e-8
    @test _rv_q(data, bl, 2) ≈ qa atol = 1e-8
    # Saturated, with |V_X| moved above the setpoint (a transient): the group returns to PV.
    units = [merge(u, (qmax = 0.05,)) for u in _RV_AB]
    data, bl = _rv_solve(; units = units, q = true)
    @test data.bus_type[bl[3], 1] == ACBusTypes.PQ
    data.bus_magnitude[bl[3], 1] = 1.03
    @test !PF._check_q_limit_bounds!(data, 1)
    @test data.bus_type[bl[3], 1] == ACBusTypes.PV
    @test data.bus_magnitude[bl[3], 1] == 1.02
    # ... and the next solve releases it again (truly at its limits).
    @test solve_power_flow!(data)
    @test data.bus_type[bl[3], 1] == ACBusTypes.PQ
    # Below the setpoint at Qmax: no return.
    @test PF._check_q_limit_bounds!(data, 1)
    # The rejoin cap.
    rvc = data.remote_voltage_control
    data.bus_magnitude[bl[3], 1] = 1.03
    rvc.returns[(1, 1)] = PF.MAX_Q_LIMIT_FLIP_FLOPS
    rvc.returns[(2, 1)] = PF.MAX_Q_LIMIT_FLIP_FLOPS
    @test_logs (:warn, r"MAX_Q_LIMIT_FLIP_FLOPS") match_mode = :any PF._check_q_limit_bounds!(
        data, 1)
    @test data.bus_type[bl[3], 1] == ACBusTypes.PQ
    PF._reset_q_limit_flip_flops!(data, 1)
    @test isempty(rvc.returns)
end

@testset "remote voltage control: a shunt on the regulated bus holds unless the group saturates (GAP-015, GAP-056)" begin
    # Group active: the shunt sees no voltage change and keeps its stored susceptance.
    data, bl = _rv_solve(; units = _RV_AB, q = true, control = true, shunt = 1.03)
    sh = only(PF.get_controlled_devices(data).shunts)
    @test sh.current == sh.initial
    @test sh.vset_hi == sh.vset_lo                     # no co-regulator band: IREG is modelled
    @test _rv_v(data, bl, 3) ≈ 1.02 atol = 1e-9
    @test PF.get_control_batched_fallback_count(data) == 0
    # Group saturated: the shunt is the bus's only regulator and holds its own setpoint.
    units = [merge(u, (qmax = 0.05,)) for u in _RV_AB]
    data, bl = _rv_solve(; units = units, q = true, control = true, shunt = 1.0)
    sh = only(PF.get_controlled_devices(data).shunts)
    @test data.bus_type[bl[3], 1] == ACBusTypes.PQ
    @test sh.current > sh.initial
    @test _rv_v(data, bl, 3) ≈ 1.0 atol = 1e-6
    # With remote control off, GAP-056's band holds the shunt beside the IREG generators.
    data, bl = _rv_solve(; units = _RV_AB, control = true, shunt = 1.03, remote = false)
    sh = only(PF.get_controlled_devices(data).shunts)
    @test sh.vset_hi - sh.vset_lo ≈ 2 * PF.CONTROL_SHARED_SETPOINT_BAND
end

@testset "remote voltage control: discrete control's batched path, no fallbacks (GAP-015)" begin
    data, bl = _rv_solve(; units = _RV_AB, q = true, control = true, tap = true)
    tx = only(PF.get_controlled_devices(data).taps)
    @test tx.current != tx.initial
    @test _rv_v(data, bl, 5) ≈ 1.03 atol = 2e-3
    @test _rv_v(data, bl, 3) ≈ 1.02 atol = 1e-9
    @test _rv_q(data, bl, 2) / _rv_q(data, bl, 4) ≈ 1.5 rtol = 1e-9
    @test PF.get_control_batched_fallback_count(data) == 0
    # With one unit at its limit through the continuation too.
    free, fbl = _rv_solve(; units = _RV_AB, q = true, tap = true)
    units = [merge(_RV_AB[1], (qmax = _rv_q(free, fbl, 2) / 2,)), _RV_AB[2]]
    data, bl = _rv_solve(; units = units, q = true, control = true, tap = true)
    @test _rv_v(data, bl, 3) ≈ 1.02 atol = 1e-9
    @test [s for (_, _, s) in PF.get_remote_voltage_status(data)[3]] == [1, 0]
    @test PF.get_control_batched_fallback_count(data) == 0
end

@testset "remote voltage control: Jacobian against finite differences (GAP-015)" begin
    far = [_RV_AB[1], (bus = 6, ireg = 3, rmpct = 40.0, qmin = -3.0, qmax = 3.0)]
    own = [(bus = 3, ireg = 0, rmpct = 50.0, qmin = -3.0, qmax = 3.0), _RV_AB[1]]
    for (label, kw, fix) in (("two units", (; units = _RV_AB), 0),
        ("non-neighbour unit", (; units = far), 0),
        ("one unit fixed", (; units = _RV_AB), 1),
        ("own plant", (; units = own, x_type = ACBusTypes.PV, x_vm = 1.025), 0),
        ("own plant fixed", (; units = own, x_type = ACBusTypes.PV, x_vm = 1.025), 1))
        data = PowerFlowData(ACPolarPowerFlow(), _rv_system(; kw...))
        grp = only(PF.get_remote_voltage_groups(data))
        X = grp.regulated_ix
        fix > 0 && (data.remote_voltage_control.status[grp.offset + fix, 1] = 1)
        residual = PF.ACPowerFlowResidual(data, 1)
        J = PF.ACPowerFlowJacobian(residual, 1)
        x0 = PF.calculate_x0(data, 1)
        Random.seed!(7)
        x0 .+= 0.02 .* randn(length(x0))
        residual(x0, 1)
        J(1)
        # The Q_tot column: −s/Σs at each sharing member's Q row.
        S = sum(grp.shares[k] for k in eachindex(grp.members) if k != fix)
        for (k, b) in enumerate(grp.members)
            @test J.Jv[2b, 2X - 1] ≈ (k == fix ? 0.0 : -grp.shares[k] / S)
        end
        verify_jacobian_asymptotic(residual, deepcopy(J.Jv), x0, 1;
            label = "remote voltage control, $label")
    end
end

@testset "remote voltage control: write-back keeps the plant PV and its free terminal (GAP-015)" begin
    sys = _rv_system(; units = _RV_AB)
    pf = ACPolarPowerFlow()
    @test solve_and_store_power_flow!(pf, sys)
    data, bl = _rv_solve(; units = _RV_AB)
    for n in (2, 4)
        bus = get_bus(sys, n)
        @test get_bustype(bus) == ACBusTypes.PV
        @test get_magnitude(bus) ≈ _rv_v(data, bl, n) atol = 1e-9
        g = only(get_components(x -> get_bus(x) == bus, ThermalStandard, sys))
        @test get_reactive_power(g) ≈ _rv_q(data, bl, n) atol = 1e-8
    end
    @test get_bustype(get_bus(sys, 3)) == ACBusTypes.PQ
    @test get_magnitude(get_bus(sys, 3)) ≈ 1.02 atol = 1e-9
end

@testset "remote voltage control: two time steps solve alike (GAP-015)" begin
    units = [merge(_RV_AB[1], (qmax = 0.2,)), _RV_AB[2]]
    one, bl = _rv_solve(; units = units, q = true)
    pf = ACPolarPowerFlow(; check_reactive_power_limits = true, time_steps = 2)
    data = PowerFlowData(pf, _rv_system(; units = units))
    @test solve_power_flow!(data)
    for ts in 1:2
        @test data.bus_magnitude[:, ts] ≈ one.bus_magnitude[:, 1] atol = 1e-9
        @test PF.get_remote_voltage_status(data, ts) == PF.get_remote_voltage_status(one)
    end
end

# A group saturated in the stored state (PSS/E's VARLIM 0 leaves every plant at its limit and X
# off VS) starts saturated: plants fixed at their limits, X PQ at its stored |V|. Stored states
# come from `solve_and_store_power_flow!` on the synthetic systems.
function _rv_stored(; units, q = true, kw...)
    sys = _rv_system(; units = units, kw...)
    @test solve_and_store_power_flow!(ACPolarPowerFlow(; check_reactive_power_limits = q), sys)
    # A plant at its limit is written back PQ, as a Q-limit release is; in a PSS/E stored case
    # its bus stays a generator bus.
    for u in units
        set_bustype!(get_bus(sys, u.bus), ACBusTypes.PV)
    end
    return sys
end

function _rv_initial_residual(data)
    residual = PF.ACPowerFlowResidual(data, 1)
    residual(PF.calculate_x0(data, 1), 1)
    return maximum(abs, residual.Rv)
end

const _RV_TIGHT = [merge(u, (qmax = 0.1,)) for u in _RV_AB]

@testset "remote voltage control: a group saturated in the stored state starts saturated (GAP-015)" begin
    sys = _rv_stored(; units = _RV_TIGHT)
    vx = get_magnitude(get_bus(sys, 3))
    @test vx < 1.02 - 1e-3                               # stored below VS, both plants at Qmax
    pf = ACPolarPowerFlow(; check_reactive_power_limits = true)
    data = PowerFlowData(pf, sys)
    bl = PF.get_bus_lookup(data)
    @test [s for (_, _, s) in PF.get_remote_voltage_status(data)[3]] == [1, 1]
    @test data.bus_type[bl[3], 1] == ACBusTypes.PQ
    @test data.bus_magnitude[bl[3], 1] == vx
    # The warm start is the stored point: no mismatch, and the solve leaves it there.
    @test _rv_initial_residual(data) < 1e-8
    stored = copy(data.bus_magnitude[:, 1])
    @test solve_power_flow!(data; maxIterations = 2)
    @test data.bus_magnitude[:, 1] ≈ stored atol = 1e-9
    @test data.bus_type[bl[3], 1] == ACBusTypes.PQ
    # Started sharing at VS instead (Q limits off), the same stored point is far off.
    free = PowerFlowData(ACPolarPowerFlow(), sys)
    @test free.bus_type[PF.get_bus_lookup(free)[3], 1] == ACBusTypes.PV
    @test _rv_initial_residual(free) > 1e-3
    # Not saturated in its stored state (the free solve): it starts sharing at VS.
    data = PowerFlowData(pf, _rv_stored(; units = _RV_AB))
    @test data.bus_type[PF.get_bus_lookup(data)[3], 1] == ACBusTypes.PV
    @test all(s == 0 for (_, _, s) in PF.get_remote_voltage_status(data)[3])
end

@testset "remote voltage control: a stored-saturated group desaturates when V_X crosses VS (GAP-015)" begin
    sys = _rv_stored(; units = _RV_TIGHT)
    # VS just above the stored |V_X| (still saturated on PSS/E's side), and a stronger source
    # (REF raised to 1.08): with both plants at Qmax, |V_X| now rises past VS.
    vs = get_magnitude(get_bus(sys, 3)) + 0.005
    for g in get_components(ThermalStandard, sys)
        get_ext(g)["VS"] = vs
    end
    set_magnitude!(get_bus(sys, 1), 1.08)
    data = PowerFlowData(ACPolarPowerFlow(; check_reactive_power_limits = true), sys)
    bl = PF.get_bus_lookup(data)
    @test data.bus_type[bl[3], 1] == ACBusTypes.PQ        # starts saturated
    @test solve_power_flow!(data)
    @test data.bus_type[bl[3], 1] == ACBusTypes.PV
    @test _rv_v(data, bl, 3) ≈ vs atol = 1e-9
    @test all(s == 0 for (_, _, s) in PF.get_remote_voltage_status(data)[3])
    @test _rv_q(data, bl, 2) < 0.1 && _rv_q(data, bl, 4) < 0.1
    @test _rv_q(data, bl, 2) / _rv_q(data, bl, 4) ≈ 1.5 rtol = 1e-9
end

@testset "remote voltage control: X's own plant, stored Q in the slot and stored saturation (GAP-015)" begin
    own = [(bus = 3, ireg = 0, rmpct = 50.0, qmin = -3.0, qmax = 3.0),
        (bus = 2, ireg = 3, rmpct = 50.0, qmin = -3.0, qmax = 3.0)]
    kw = (; x_type = ACBusTypes.PV, x_vm = 1.02)
    # Sharing: Q_tot starts as the members' summed stored Q, X's own plant included.
    sys = _rv_stored(; units = own, kw...)
    data = PowerFlowData(ACPolarPowerFlow(; check_reactive_power_limits = true), sys)
    grp = only(PF.get_remote_voltage_groups(data))
    X = grp.regulated_ix
    qsum = PSY.with_units_base(sys, PSY.UnitSystem.SYSTEM_BASE) do
        sum(get_reactive_power(g) for g in get_components(ThermalStandard, sys))
    end
    @test PF.calculate_x0(data, 1)[2X - 1] ≈ qsum atol = 1e-9
    @test _rv_initial_residual(data) < 1e-8
    # Saturated: X's plant and the remote one at Qmax, X (PV in the data) below VS: X starts PQ
    # at its stored |V|, and the stored point solves as it stands.
    tight = [merge(u, (qmax = 0.1,)) for u in own]
    sys = _rv_stored(; units = tight, kw...)
    vx = get_magnitude(get_bus(sys, 3))
    @test vx < 1.02 - 1e-3
    data = PowerFlowData(ACPolarPowerFlow(; check_reactive_power_limits = true), sys)
    bl = PF.get_bus_lookup(data)
    @test only(PF.get_remote_voltage_groups(data)).vset == 1.02
    @test data.bus_type[bl[3], 1] == ACBusTypes.PQ
    @test data.bus_magnitude[bl[3], 1] == vx
    @test [s for (_, _, s) in PF.get_remote_voltage_status(data)[3]] == [1, 1]
    @test _rv_initial_residual(data) < 1e-8
    stored = copy(data.bus_magnitude[:, 1])
    @test solve_power_flow!(data; maxIterations = 2)
    @test data.bus_magnitude[:, 1] ≈ stored atol = 1e-9
end

@testset "remote voltage control: off VS but not at its limits, a group starts sharing (GAP-015)" begin
    # The free stored state with VS moved 0.01 above it: X is off VS, but no plant is at a limit,
    # so the group starts sharing at VS and the solve takes X there.
    sys = _rv_stored(; units = _RV_AB)
    for g in get_components(ThermalStandard, sys)
        get_ext(g)["VS"] = 1.03
    end
    data = PowerFlowData(ACPolarPowerFlow(; check_reactive_power_limits = true), sys)
    bl = PF.get_bus_lookup(data)
    @test data.bus_type[bl[3], 1] == ACBusTypes.PV
    @test data.bus_magnitude[bl[3], 1] == 1.03
    @test solve_power_flow!(data)
    @test _rv_v(data, bl, 3) ≈ 1.03 atol = 1e-9
    @test all(s == 0 for (_, _, s) in PF.get_remote_voltage_status(data)[3])
end

@testset "remote voltage control: non-generator injections at a plant's bus are not shared (GAP-015)" begin
    # X has its own plant and a Source injecting fixed reactive power: the plant (X's generators)
    # shares with the remote one; the Source's Q stays as stored and outside Q_tot.
    own = [(bus = 3, ireg = 0, rmpct = 50.0, qmin = -3.0, qmax = 3.0),
        (bus = 2, ireg = 3, rmpct = 50.0, qmin = -3.0, qmax = 3.0)]
    sys = _rv_system(; units = own, x_type = ACBusTypes.PV, x_vm = 1.02)
    src = _add_simple_source!(sys, get_bus(sys, 3), 0.0, 0.0)
    PSY.with_units_base(sys, PSY.UnitSystem.SYSTEM_BASE) do
        set_reactive_power!(src, 0.2)
    end
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    grp = only(PF.get_remote_voltage_groups(data))
    X = grp.regulated_ix
    rvc = data.remote_voltage_control
    @test rvc.q_fixed[grp.offset + 1, 1] ≈ 0.2 atol = 1e-12   # X's row: the Source
    @test rvc.q_fixed[grp.offset + 2, 1] ≈ 0.0 atol = 1e-12
    @test PF.calculate_x0(data, 1)[2X - 1] ≈ 0.0 atol = 1e-12  # plants' stored Q (0 each)
    @test solve_power_flow!(data)
    bl = PF.get_bus_lookup(data)
    @test _rv_v(data, bl, 3) ≈ 1.02 atol = 1e-9
    @test _rv_q(data, bl, 3) - 0.2 ≈ _rv_q(data, bl, 2) rtol = 1e-9
end
