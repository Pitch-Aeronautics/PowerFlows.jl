# The discrete-control continuation batches its passes (one joint solve per pass) through a
# sensitivity context built at a converged state. A Q-limit PV→PQ flip in a joint solve changes
# the bus types the context was built under; the context must be rebuilt there, not abandoned
# (before, every later pass ran the per-device path: one full solve per device).

"""REF ─tap→ PQ load buses 11..10+n, each tap regulating its bus to 1.03; a PV generator
(bus 2, V = 1.0) hangs off bus 11 by a short line. As the taps raise |V₁₁| above 1.0 the
generator absorbs reactive power, and its floor `qmin` binds part-way through the control."""
function _q_limited_tap_system(; n_taps = 4, qmin = -0.1)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    loads = ACBus[]
    for k in 1:n_taps
        bk = _add_simple_bus!(sys, 10 + k, ACBusTypes.PQ, 230, 1.0, 0.0)
        _add_simple_load!(sys, bk, 0.3, 0.15)
        push!(loads, bk)
        add_component!(sys, TapTransformer(; name = "t$k", available = true,
            active_power_flow = 0.0, reactive_power_flow = 0.0,
            arc = Arc(; from = b1, to = bk), r = 0.005, x = 0.08 + 0.01k,
            primary_shunt = 0.0 + 0.0im, tap = 1.0, rating = 1.0, base_power = 100.0,
            control_objective = PSY.TransformerControlObjective.VOLTAGE,
            voltage_setpoint = 1.03))
    end
    bg = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230, 1.0, 0.0)
    gen = _add_simple_thermal_standard!(sys, bg, 0.1, 0.0)
    set_reactive_power_limits!(gen, (min = qmin, max = 1.0))
    _add_simple_line!(sys, bg, loads[1], 0.005, 0.05, 0.0)
    return sys
end

function _solve_q_limited(pf_type)
    sys = _q_limited_tap_system()
    pf = pf_type(; control_discrete_devices = true, check_reactive_power_limits = true)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    return data
end

@testset "discrete control: a Q-limit flip mid-control rebuilds the sensitivity context" begin
    data = _solve_q_limited(ACPolarPowerFlow)
    bl = PF.get_bus_lookup(data)
    set = PF.get_controlled_devices(data)
    # The generator hit its floor during the control (it is within it at the base state).
    base = PowerFlowData(ACPolarPowerFlow(; check_reactive_power_limits = true),
        _q_limited_tap_system())
    @test solve_power_flow!(base)
    @test base.bus_type[PF.get_bus_lookup(base)[2], 1] == ACBusTypes.PV
    @test data.bus_type[bl[2], 1] == ACBusTypes.PQ
    @test data.bus_reactive_power_injections[bl[2], 1] ≈ -0.1
    # Every pass stayed batched: the context was rebuilt after the flip, never abandoned.
    @test PF.get_control_context_rebuild_count(data) >= 1
    @test PF.get_control_batched_fallback_count(data) == 0
    # A rebuild is not a symbolic analysis: the NR cache's and the probe-phase build only.
    @test PF.get_control_symbolic_factor_count(data) == 2
    n_batched = PF.get_control_inner_solve_count(data)

    # The per-device path (the rectangular formulation has no linear sensitivity path, so
    # every pass steps the devices one solve at a time) lands on the same solution.
    ref = _solve_q_limited(ACRectangularPowerFlow)
    ref_set = PF.get_controlled_devices(ref)
    @test [d.current for d in set.taps] == [d.current for d in ref_set.taps]
    @test ref.bus_type[:, 1] == data.bus_type[:, 1]
    @test data.bus_magnitude[:, 1] ≈ ref.bus_magnitude[:, 1] atol = 1e-8
    @test data.bus_angles[:, 1] ≈ ref.bus_angles[:, 1] atol = 1e-8
    @test PF.get_control_inner_solve_count(ref) > n_batched
end
