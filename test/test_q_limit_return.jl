# GAP-027: the reactive-limit check returns a released bus to PV (PSS/E's VARLIM rule). A bus
# released to PQ at Qmax whose |V| has risen above its setpoint, or at Qmin with |V| below it,
# regulates again; before, a release was one-way, so a unit pushed past a limit by a transient
# (a discrete-control continuation's early passes) ended at that limit, off its setpoint.

"""REF bus 1 ─ PV generator bus 2 (|V| = `vg`, P = 0.2) ─ PQ load bus 3 (0.5 + j0.2). At the
base state the unit makes Q ≈ 0.543. `units` splits its Q limits over that many identical units
on bus 2; `ireg` gives each unit a PSS/E IREG (a remote regulated bus) in `ext["psse"]`. With
`vreg` a continuous switched shunt at bus 3, range [0, 1], holds bus 3 at `vreg`."""
function _q_return_system(; vg = 1.03, qmin = -0.05, qmax = 0.6, units = 1, ireg = nothing,
    vreg = nothing)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230, vg, 0.0)
    for _ in 1:units
        g = _add_simple_thermal_standard!(sys, b2, 0.2 / units, 0.0)
        set_reactive_power_limits!(g, (min = qmin / units, max = qmax / units))
        isnothing(ireg) || (get_ext(g)["psse"] = Dict{String, Any}("ireg" => ireg))
    end
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_load!(sys, b3, 50.0, 20.0)
    _add_simple_line!(sys, b1, b2, 0.005, 0.1, 0.0)
    _add_simple_line!(sys, b2, b3, 0.005, 0.05, 0.0)
    if !isnothing(vreg)
        add_component!(sys, SwitchedAdmittance(; name = "sh3", available = true, bus = b3,
            Y = 0.0im, initial_status = [0], number_of_steps = [10], Y_increase = [0.1im],
            admittance_limits = (min = vreg, max = vreg),
            control_mode = PSY.SwitchedAdmittanceControlMode.CONTINUOUS_VOLTAGE,
            regulated_bus_number = 0,
            ext = Dict{String, Any}("MODSW" => 2, "ADJM" => 0)))
    end
    return sys
end

function _q_return_data(; control = false, kw...)
    pf = ACPolarPowerFlow(;
        check_reactive_power_limits = true, control_discrete_devices = control)
    data = PowerFlowData(pf, _q_return_system(; kw...))
    @test solve_power_flow!(data)
    return data, PF.get_bus_lookup(data)
end

# A capacitor stepped in at bus 3 and back out, on one PowerFlowData: in, the unit is driven to
# its floor and released; out, the second solve starts with the bus PQ at Qmin.
function _step_in_and_out!(data, bl)
    w = data.bus_reactive_power_constant_impedance_withdrawals
    w[bl[3], 1] = -0.6
    @test solve_power_flow!(data)
    q_in = data.bus_reactive_power_injections[bl[2], 1]
    v_in = data.bus_magnitude[bl[2], 1]
    type_in = data.bus_type[bl[2], 1]
    w[bl[3], 1] = 0.0
    @test solve_power_flow!(data)
    return type_in, q_in, v_in
end

@testset "Q-limit return: a unit released by a transient returns to PV at its setpoint" begin
    ref, rbl = _q_return_data()
    @test ref.bus_type[rbl[2], 1] == ACBusTypes.PV
    @test PF.get_q_limit_return_count(ref) == 0
    data, bl = _q_return_data()
    type_in, q_in, v_in = _step_in_and_out!(data, bl)
    # Capacitor in: truly at its floor (|V| above the setpoint it cannot pull down to), so PQ.
    @test type_in == ACBusTypes.PQ
    @test q_in == -0.05
    @test v_in > 1.03 + 1e-3
    # Capacitor out: at Qmin with |V| below the setpoint, so back to PV, on the base solution.
    # (One-way, it ended PQ at -0.05 with |V| = 0.971.)
    @test data.bus_type[bl[2], 1] == ACBusTypes.PV
    @test data.bus_magnitude[bl[2], 1] == 1.03
    @test PF.get_q_limit_return_count(data) == 1
    @test data.bus_reactive_power_injections[bl[2], 1] ≈
          ref.bus_reactive_power_injections[rbl[2], 1] atol = 1e-8
    @test data.bus_magnitude[:, 1] ≈ ref.bus_magnitude[:, 1] atol = 1e-9
    @test data.bus_angles[:, 1] ≈ ref.bus_angles[:, 1] atol = 1e-9
    @test isempty(PF.get_q_limit_capped_buses(data))
end

@testset "Q-limit return: a unit truly at its limit stays PQ" begin
    # Floor 0.5 above what the base needs: released at its floor with |V| above the setpoint.
    data, bl = _q_return_data(; qmin = 0.6, qmax = 1.0)
    @test data.bus_type[bl[2], 1] == ACBusTypes.PQ
    @test data.bus_reactive_power_injections[bl[2], 1] == 0.6
    @test data.bus_magnitude[bl[2], 1] > 1.03
    # Ceiling below what the base needs: at Qmax with |V| below the setpoint.
    data, bl = _q_return_data(; qmin = -0.5, qmax = 0.4)
    @test data.bus_type[bl[2], 1] == ACBusTypes.PQ
    @test data.bus_reactive_power_injections[bl[2], 1] == 0.4
    @test data.bus_magnitude[bl[2], 1] < 1.03
    @test PF.get_q_limit_return_count(data) == 0
    # A bus with no Q range never returns, wrong side or not.
    data, bl = _q_return_data(; qmin = 0.3, qmax = 0.3)
    @test data.bus_type[bl[2], 1] == ACBusTypes.PQ
    @test PF.get_q_limit_return_count(data) == 0
    # A bus that is PQ by its data never becomes PV.
    data, bl = _q_return_data()
    @test all(!=(ACBusTypes.PV), data.bus_type[[bl[1], bl[3]], 1])
end

@testset "Q-limit return: multi-unit and IREG buses switch as one, on their own terminal" begin
    ref, rbl = _q_return_data()
    for (units, ireg) in ((2, nothing), (1, 3), (3, 3))
        data, bl = _q_return_data(; units = units, ireg = ireg)
        type_in, q_in, _ = _step_in_and_out!(data, bl)
        @test type_in == ACBusTypes.PQ
        @test q_in ≈ -0.05     # the bus's floor, the sum over its units
        @test data.bus_type[bl[2], 1] == ACBusTypes.PV
        # A remote-regulating unit holds its own terminal in PowerFlows: it returns to that.
        @test data.bus_magnitude[bl[2], 1] == 1.03
        @test PF.get_q_limit_return_count(data) == 1
        @test data.bus_magnitude[:, 1] ≈ ref.bus_magnitude[:, 1] atol = 1e-9
    end
end

@testset "Q-limit return: a continuation transient no longer strands a unit at its floor" begin
    # The shunt holds bus 3 4e-3 above its base voltage. The continuation's low-steepness stages
    # overshoot (B toward the middle of its range), the unit absorbs, crosses its floor 0.12
    # below its base Q and is released; the shunt then comes back to B ≈ 0.078.
    vreg = 1.0174441722951675 + 0.004
    free, fbl = _q_return_data(; control = true, vreg = vreg, qmin = -5.0)
    @test PF.get_q_limit_return_count(free) == 0
    q_free = free.bus_reactive_power_injections[fbl[2], 1]
    @test q_free ≈ 0.4607 atol = 1e-3
    for units in (1, 2)
        data, bl = _q_return_data(; control = true, vreg = vreg,
            qmin = 0.5429825441268852 - 0.12, units = units)
        sh = only(PF.get_controlled_devices(data).shunts)
        # One-way, the unit ended PQ at 0.423 with |V| 1.2e-3 low, and the shunt at B = 0.101.
        @test data.bus_type[bl[2], 1] == ACBusTypes.PV
        @test data.bus_magnitude[bl[2], 1] == 1.03
        @test PF.get_q_limit_return_count(data) >= 1
        @test data.bus_reactive_power_injections[bl[2], 1] ≈ q_free atol = 1e-7
        @test sh.current ≈ only(PF.get_controlled_devices(free).shunts).current atol = 1e-7
        @test data.bus_magnitude[:, 1] ≈ free.bus_magnitude[:, 1] atol = 1e-8
        # The flips (out and back) rebuilt the batched context; no pass fell back.
        @test PF.get_control_context_rebuild_count(data) >= 2
        @test PF.get_control_batched_fallback_count(data) == 0
    end
    # A floor the end state needs more than: released and kept at it, |V| above the setpoint.
    data, bl = _q_return_data(; control = true, vreg = vreg, qmin = 0.4729825441268852)
    @test data.bus_type[bl[2], 1] == ACBusTypes.PQ
    @test data.bus_reactive_power_injections[bl[2], 1] ≈ 0.4729825441268852
    @test data.bus_magnitude[bl[2], 1] > 1.03
end

@testset "Q-limit return: flip-flops are capped per bus" begin
    data, bl = _q_return_data()
    st = data.q_limit_switches
    ix = bl[2]
    qmin = data.bus_reactive_power_bounds[ix, 1][1]
    empty!(st.returned_this_loop)       # one Q-limit loop
    for k in 1:PF.MAX_Q_LIMIT_FLIP_FLOPS
        # Released at its floor with |V| below its setpoint: returns.
        data.bus_type[ix, 1] = ACBusTypes.PQ
        st.vset[(ix, 1)] = 1.03
        data.bus_reactive_power_injections[ix, 1] = qmin
        data.bus_magnitude[ix, 1] = 1.02
        @test !PF._check_q_limit_bounds!(data, 1)
        @test data.bus_type[ix, 1] == ACBusTypes.PV
        @test data.bus_magnitude[ix, 1] == 1.03
        # The next solve of the same loop has it below its floor again: released.
        data.bus_reactive_power_injections[ix, 1] = qmin - 0.1
        if k < PF.MAX_Q_LIMIT_FLIP_FLOPS
            @test !PF._check_q_limit_bounds!(data, 1)
            @test isempty(PF.get_q_limit_capped_buses(data))
        else
            @test_logs (:warn, r"MAX_Q_LIMIT_FLIP_FLOPS") match_mode = :any @test(
                !PF._check_q_limit_bounds!(data, 1))
            @test PF.get_q_limit_capped_buses(data) == [ix]
        end
        @test data.bus_type[ix, 1] == ACBusTypes.PQ
        @test data.bus_reactive_power_injections[ix, 1] == qmin
    end
    # Capped: it stays PQ on the wrong side for the rest of this time step's solve.
    data.bus_magnitude[ix, 1] = 1.02
    @test PF._check_q_limit_bounds!(data, 1)
    @test data.bus_type[ix, 1] == ACBusTypes.PQ
    @test PF.get_q_limit_return_count(data) == PF.MAX_Q_LIMIT_FLIP_FLOPS
    # A return that holds is not a flip-flop: a fresh loop, returned, not released again.
    PF._reset_q_limit_flip_flops!(data, 1)
    @test isempty(PF.get_q_limit_capped_buses(data))
    empty!(st.returned_this_loop)
    @test !PF._check_q_limit_bounds!(data, 1)
    @test data.bus_type[ix, 1] == ACBusTypes.PV
    @test isempty(st.flip_flops)
    # The next solve resets the count: the whole solve runs and lands on the base solution.
    @test solve_power_flow!(data)
    @test data.bus_type[ix, 1] == ACBusTypes.PV
    @test data.bus_magnitude[ix, 1] == 1.03
end

@testset "Q-limit return: no change where no unit crosses back" begin
    # c_sys14 with limits enforced: its units released at Qmax stay there (|V| below setpoint).
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    set_units_base_system!(sys, UnitSystem.SYSTEM_BASE)
    pf = ACPowerFlow{PF.TrustRegionACPowerFlow}(;
        check_reactive_power_limits = true, correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    @test PF.get_q_limit_return_count(data) == 0
    @test any(==(ACBusTypes.PQ), data.bus_type[collect(keys(data.q_limit_switches.vset) .|>
                                                       first), 1])
    # Without the limit check nothing is ever recorded.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    set_units_base_system!(sys, UnitSystem.SYSTEM_BASE)
    data = PowerFlowData(ACPowerFlow{PF.TrustRegionACPowerFlow}(; correct_bustypes = true), sys)
    @test solve_power_flow!(data)
    @test isempty(data.q_limit_switches.vset)
end
