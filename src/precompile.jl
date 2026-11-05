# PrecompileTools workload (T113): the AC solves a caller makes first, on the IEEE 14-bus test
# system (the public 1962 case, `test/test_data/14_bus.raw`, built here through the PowerSystems
# API) and on a three-bus remote-regulation synthetic, so a fresh session does not pay their
# compilation. Disabled by the standard preference: `Preferences.set_preferences!(PowerFlows,
# "precompile_workload" => false; force = true)` (or PrecompileTools' global
# "precompile_workloads").
import PrecompileTools

function _pc_bus!(sys, n, bt, vm, va)
    b = PSY.ACBus(; number = n, name = "pc_bus_$n", available = true, bustype = bt,
        angle = deg2rad(va), magnitude = vm, voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 138.0)
    PSY.add_component!(sys, b)
    return b
end

function _pc_gen!(sys, bus, p, q, qmin, qmax; ext = Dict{String, Any}())
    g = PSY.ThermalStandard(; name = "pc_gen_$(PSY.get_number(bus))", available = true,
        status = true, bus = bus, active_power = p, reactive_power = q, rating = 10.0,
        active_power_limits = (min = 0.0, max = 10.0),
        reactive_power_limits = (min = qmin, max = qmax), ramp_limits = nothing,
        operation_cost = PSY.ThermalGenerationCost(nothing), base_power = 100.0,
        time_limits = nothing, prime_mover_type = PSY.PrimeMovers.OT,
        fuel = PSY.ThermalFuels.OTHER, services = PSY.Device[], dynamic_injector = nothing,
        ext = ext)
    PSY.add_component!(sys, g)
    return g
end

function _pc_load!(sys, bus, p, q)
    PSY.add_component!(sys, PSY.PowerLoad(; name = "pc_load_$(PSY.get_number(bus))",
        available = true, bus = bus, active_power = p, reactive_power = q, base_power = 100.0,
        max_active_power = p, max_reactive_power = q))
    return
end

function _pc_line!(sys, f, t, r, x, b)
    PSY.add_component!(sys, PSY.Line(;
        name = "pc_line_$(PSY.get_number(f))_$(PSY.get_number(t))", available = true,
        active_power_flow = 0.0, reactive_power_flow = 0.0, arc = PSY.Arc(; from = f, to = t),
        r = r, x = x, b = (from = b / 2, to = b / 2), rating = 10.0,
        angle_limits = (min = -pi / 2, max = pi / 2)))
    return
end

function _pc_tap!(sys, f, t, x, tap, objective)
    PSY.add_component!(sys, PSY.TapTransformer(;
        name = "pc_tap_$(PSY.get_number(f))_$(PSY.get_number(t))", available = true,
        active_power_flow = 0.0, reactive_power_flow = 0.0, arc = PSY.Arc(; from = f, to = t),
        r = 0.0, x = x, primary_shunt = 0.0im, tap = tap, rating = 10.0, base_power = 100.0,
        control_objective = objective, voltage_setpoint = 1.06))
    return
end

"""The IEEE 14-bus case (system base, 100 MVA). With `controls`, its three transformers regulate
voltage and a discrete switched shunt sits at bus 14."""
function _pc_ieee14(; controls::Bool = false)
    sys = PSY.System(100.0)
    T = PSY.ACBusTypes
    buses = [(1, T.REF, 1.06, 0.0), (2, T.PV, 1.045, -4.98), (3, T.PV, 1.01, -12.73),
        (4, T.PQ, 1.018, -10.31), (5, T.PQ, 1.02, -8.77), (6, T.PV, 1.07, -14.22),
        (7, T.PQ, 1.062, -13.36), (8, T.PV, 1.09, -13.36), (9, T.PQ, 1.056, -14.94),
        (10, T.PQ, 1.051, -15.1), (11, T.PQ, 1.057, -14.79), (12, T.PQ, 1.055, -15.08),
        (13, T.PQ, 1.05, -15.16), (14, T.PQ, 1.036, -16.03)]
    b = Dict(n => _pc_bus!(sys, n, bt, vm, va) for (n, bt, vm, va) in buses)
    PSY.add_component!(sys, PSY.Source(; name = "pc_source_1", available = true, bus = b[1],
        active_power = 2.324, reactive_power = -0.165, R_th = 1e-5, X_th = 1e-5))
    for (n, p, q, qmin, qmax) in ((2, 0.4, 0.4356, -0.4, 0.5), (3, 0.0, 0.2508, 0.0, 0.4),
        (6, 0.0, 0.1273, -0.06, 0.24), (8, 0.0, 0.1762, -0.06, 0.24))
        _pc_gen!(sys, b[n], p, q, qmin, qmax)
    end
    for (n, p, q) in ((2, 0.217, 0.127), (3, 0.942, 0.19), (4, 0.478, -0.039),
        (5, 0.076, 0.016), (6, 0.112, 0.075), (9, 0.295, 0.166), (10, 0.09, 0.058),
        (11, 0.035, 0.018), (12, 0.061, 0.016), (13, 0.135, 0.058), (14, 0.149, 0.05))
        _pc_load!(sys, b[n], p, q)
    end
    PSY.add_component!(sys, PSY.FixedAdmittance(; name = "pc_shunt_9", available = true,
        bus = b[9], Y = 0.19im))
    for (f, t, r, x, bb) in ((1, 2, 0.01938, 0.05917, 0.0528), (1, 5, 0.05403, 0.22304, 0.0492),
        (2, 3, 0.04699, 0.19797, 0.0438), (2, 4, 0.05811, 0.17632, 0.034),
        (2, 5, 0.05695, 0.17388, 0.0346), (3, 4, 0.06701, 0.17103, 0.0128),
        (4, 5, 0.01335, 0.04211, 0.0), (6, 11, 0.09498, 0.1989, 0.0),
        (6, 12, 0.12291, 0.25581, 0.0), (6, 13, 0.06615, 0.13027, 0.0),
        (7, 8, 0.0, 0.17615, 0.0), (7, 9, 0.0, 0.11001, 0.0), (9, 10, 0.03181, 0.0845, 0.0),
        (9, 14, 0.12711, 0.27038, 0.0), (10, 11, 0.08205, 0.19207, 0.0),
        (12, 13, 0.22092, 0.19988, 0.0), (13, 14, 0.17093, 0.34802, 0.0))
        _pc_line!(sys, b[f], b[t], r, x, bb)
    end
    objective = controls ? PSY.TransformerControlObjective.VOLTAGE :
                PSY.TransformerControlObjective.FIXED
    for (f, t, x, tap) in ((4, 7, 0.20912, 0.978), (4, 9, 0.55618, 0.969), (5, 6, 0.25202, 0.932))
        _pc_tap!(sys, b[f], b[t], x, tap, objective)
    end
    if controls
        PSY.add_component!(sys, PSY.SwitchedAdmittance(; name = "pc_swshunt_14",
            available = true, bus = b[14], Y = 0.0im, initial_status = [0],
            number_of_steps = [4], Y_increase = [0.05im],
            admittance_limits = (min = 1.03, max = 1.05),
            control_mode = PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE))
    end
    return sys
end

"""REF(1) ─ X(3, PQ, load) with a PV plant at bus 2 regulating X (PSS/E IREG, RMPCT 100)."""
function _pc_remote_system()
    sys = PSY.System(100.0)
    T = PSY.ACBusTypes
    b1 = _pc_bus!(sys, 1, T.REF, 1.0, 0.0)
    b2 = _pc_bus!(sys, 2, T.PV, 1.0, 0.0)
    b3 = _pc_bus!(sys, 3, T.PQ, 1.0, 0.0)
    PSY.add_component!(sys, PSY.Source(; name = "pc_source_1", available = true, bus = b1,
        active_power = 0.0, reactive_power = 0.0, R_th = 1e-5, X_th = 1e-5))
    _pc_line!(sys, b1, b3, 0.01, 0.15, 0.0)
    _pc_line!(sys, b2, b3, 0.005, 0.08, 0.0)
    _pc_load!(sys, b3, 0.8, 0.5)
    _pc_gen!(sys, b2, 0.3, 0.0, -3.0, 3.0;
        ext = Dict{String, Any}("IREG" => 3, "RMPCT" => 100.0, "VS" => 1.02))
    return sys
end

# The chain under `solve_power_flow!` specializes on the keyword set (`solver_settings` plus the
# call's kwargs, `_sorted_kwargs`), and a new set costs about 18 s of compilation here, a different
# solver type (trust region: every `PowerFlowData{ACPolarPowerFlow{S}}` method) about 60 s. So the
# workload covers no settings, `maxIterations` + `tol`, and those with the line search; the
# controls, Q limits and remote regulation are fields, not kwargs, and ride on those chains.
# Iwamoto and the trust region compile at first use. This took a full solve from 108 s to 6 s of
# first-call compilation and adds about 105 s and 4 GB to the package's own precompile.
function _pc_workload()
    NR = NewtonRaphsonACPowerFlow
    PowerFlows.solve_power_flow(ACPowerFlow{NR}(), _pc_ieee14())
    solve_and_store_power_flow!(ACPowerFlow{NR}(), _pc_ieee14())
    for extra in ((), (:line_search => :backtracking,))
        st = Dict{Symbol, Any}(:maxIterations => 50, :tol => 1e-9, extra...)
        for q in (false, true)
            data = PowerFlowData(
                ACPowerFlow{NR}(; check_reactive_power_limits = q, solver_settings = st),
                _pc_ieee14())
            solve_power_flow!(data)
        end
        solve_and_store_power_flow!(
            ACPowerFlow{NR}(; control_discrete_devices = true,
                check_reactive_power_limits = true, solver_settings = st),
            _pc_ieee14(; controls = true))
        data = PowerFlowData(
            ACPowerFlow{NR}(; check_reactive_power_limits = true, solver_settings = st),
            _pc_remote_system())
        solve_power_flow!(data)
    end
    return
end

PrecompileTools.@setup_workload begin
    PrecompileTools.@compile_workload begin
        # A failed workload must not stop the package from loading: warn and go on.
        try
            Logging.with_logger(Logging.NullLogger()) do
                _pc_workload()
            end
        catch e
            @warn "PowerFlows precompile workload failed" exception = (e, catch_backtrace())
        end
    end
end
