# Tap orientation: by measured dV/dtap (default), by PSS/E's CONT1 sign convention
# (`tap_direction = :cont_sign`), and parallel taps on one regulated bus aligned.

"""REF(1) ─line(x = `xline`)─ F(2) ─tap F→T─ T(3) with a PV generator at T (|V₃| = 1.0) and a
load at F. The tap regulates its own winding-1 bus F (`regulated_bus_number = 2`), band
[`vmi`, `vma`] as the PSS/E parser stores it. With T held, raising the ratio pushes reactive
power into F, so the measured dV_F/dtap is positive; PSS/E's convention reads CONT1 > 0 as
"bus on the winding-2 side", where raising the ratio lowers the voltage."""
function _winding1_regulated_tap_system(; cont1 = 2, vmi = 1.0, vma = 1.02, tap = 1.0,
    xline = 0.2, psse_ext::Bool = false)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    bf = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    bt = _add_simple_bus!(sys, 3, ACBusTypes.PV, 115, 1.0, 0.0)
    _add_simple_line!(sys, b1, bf, 0.01, xline, 0.0)
    _add_simple_load!(sys, bf, 20.0, 15.0)
    _add_simple_thermal_standard!(sys, bt, 0.0, 0.0)
    ext = Dict{String, Any}("VMI1" => vmi, "VMA1" => vma)
    if psse_ext
        ext["psse"] = Dict{String, Any}("CONT1" => cont1)
    else
        ext["CONT1"] = cont1
    end
    tx = TapTransformer(; name = "tw1", available = true, active_power_flow = 0.0,
        reactive_power_flow = 0.0, arc = Arc(; from = bf, to = bt), r = 0.002, x = 0.08,
        primary_shunt = 0.0im, tap = tap, rating = 1.0, base_power = 100.0,
        control_objective = PSY.TransformerControlObjective.VOLTAGE,
        voltage_setpoint = (vmi + vma) / 2, regulated_bus_number = abs(cont1), ext = ext)
    add_component!(sys, tx)
    return sys
end

function _solve_direction(sys, direction)
    pf = ACPolarPowerFlow(; control_discrete_devices = true, tap_direction = direction)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    d = only(PF.get_controlled_devices(data).taps)
    return d, data.bus_magnitude[d.controlled_ix, 1], data
end

@testset "tap direction: CONT1 sign read from ext, option plumbing" begin
    sys = _winding1_regulated_tap_system(; cont1 = 2)
    tx = only(get_components(TapTransformer, sys))
    @test PowerFlows._tap_cont_orientation(tx) == -1      # winding-2 side convention
    sysn = _winding1_regulated_tap_system(; cont1 = -2)
    @test PowerFlows._tap_cont_orientation(only(get_components(TapTransformer, sysn))) == 1
    sysp = _winding1_regulated_tap_system(; cont1 = -2, psse_ext = true)
    @test PowerFlows._tap_cont_orientation(only(get_components(TapTransformer, sysp))) == 1
    # An API-built tap has no CONT1: measured orientation.
    sys_api = _make_solvable_tap_shunt_system()
    @test PowerFlows._tap_cont_orientation(only(get_components(TapTransformer, sys_api))) ==
          0
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    bl = PF.get_bus_lookup(data)
    @test only(PowerFlows.build_controlled_device_set(sys, bl, data.power_network_matrix;
        tap_direction = :cont_sign).taps).orientation == -1
    @test only(PowerFlows.build_controlled_device_set(sys, bl,
        data.power_network_matrix).taps).orientation == 0
    @test ACPolarPowerFlow().tap_direction == :sensitivity
    @test ACRectangularPowerFlow().tap_direction == :sensitivity
    @test ACMixedPowerFlow().tap_direction == :sensitivity
    @test ACPolarPowerFlow(; tap_direction = :cont_sign).tap_direction == :cont_sign
    @test_throws ArgumentError ACPolarPowerFlow(; tap_direction = :psse)
end

@testset "tap direction: the two orientations disagree on a winding-1 regulated tap" begin
    sys0 = _winding1_regulated_tap_system()
    base = PowerFlowData(ACPolarPowerFlow(), sys0)
    @test solve_power_flow!(base)
    v0 = base.bus_magnitude[PF.get_bus_lookup(base)[2], 1]
    @test v0 < 1.0                                     # below the band [1.0, 1.02]
    # Measured: raising the ratio raises |V_F|; the tap steps up into the band.
    d_s, v_s, _ = _solve_direction(_winding1_regulated_tap_system(), :sensitivity)
    @test d_s.current > d_s.initial
    @test 1.0 - PowerFlows.CONTROL_TAP_BAND_TOL <= v_s <= 1.02
    # PSS/E's convention with CONT1 = +2 (winding-2 side): lower the ratio to raise the
    # voltage. Here that lowers |V_F|, so the tap runs to its lower limit, short of the band.
    d_c, v_c, _ = _solve_direction(_winding1_regulated_tap_system(), :cont_sign)
    @test d_c.orientation == -1
    @test d_c.current == d_c.p_min
    @test v_c < v0
    # CONT1 = −2 (winding-1 side) agrees with the measurement: same answer as :sensitivity.
    d_n, v_n, _ = _solve_direction(_winding1_regulated_tap_system(; cont1 = -2), :cont_sign)
    @test d_n.orientation == 1
    @test d_n.current == d_s.current
    @test v_n ≈ v_s atol = 1e-8
end

"""REF(1) ─line(0.01 + j0.1)─ F(2) ═ two parallel taps A, B (F→T, x = 0.02) ═ T(3), a 40 MW /
10 MVAr load at T. Both regulate T to [1.04, 1.05] (|V_T| ≈ 0.90 below it). A sits at 1.1, B at
0.9, so a large reactive current circulates between them, and measured one at a time dV_T/dtap
is −1.07 for A but +0.49 for B: raising B cuts the circulation, which raises V_T. Oriented by
those signs the continuation walks A down and B up, a tap fight."""
function _parallel_tap_system(; cont1 = nothing)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    bf = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    bt = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 115, 1.0, 0.0)
    _add_simple_line!(sys, b1, bf, 0.01, 0.1, 0.0)
    _add_simple_load!(sys, bt, 40.0, 10.0)
    for (nm, t) in (("A", 1.1), ("B", 0.9))
        ext = Dict{String, Any}("VMI1" => 1.04, "VMA1" => 1.05)
        isnothing(cont1) || (ext["CONT1"] = cont1)
        add_component!(sys, TapTransformer(; name = nm, available = true,
            active_power_flow = 0.0, reactive_power_flow = 0.0,
            arc = Arc(; from = bf, to = bt), r = 0.002, x = 0.02, primary_shunt = 0.0im,
            tap = t, rating = 1.0, base_power = 100.0,
            control_objective = PSY.TransformerControlObjective.VOLTAGE,
            voltage_setpoint = 1.045, regulated_bus_number = 3, ext = ext))
    end
    return sys
end

function _solve_parallel(; direction = :sensitivity, kw...)
    pf = ACPolarPowerFlow(; control_discrete_devices = true, tap_direction = direction)
    data = PowerFlowData(pf, _parallel_tap_system(; kw...))
    @test solve_power_flow!(data)
    set = PF.get_controlled_devices(data)
    taps = Dict(d.name => d for d in set.taps)
    return taps, data.bus_magnitude[PF.get_bus_lookup(data)[3], 1], data
end

@testset "tap direction: parallel taps on one regulated bus measure opposite signs" begin
    data = PowerFlowData(ACPolarPowerFlow(), _parallel_tap_system())
    @test solve_power_flow!(data)
    set = PowerFlows.build_controlled_device_set(_parallel_tap_system(),
        PF.get_bus_lookup(data), data.power_network_matrix)
    ctx = PowerFlows._build_sensitivity_context(data, 1)
    s = Dict(d.name => first(PowerFlows._linear_plant_sign(d, data, 1, ctx))
             for d in set.taps)
    @test s["A"] < 0 < s["B"]
    # Aligned to the joint effect (negative): both lower the ratio to raise V_T.
    dVdp = [s[d.name] for d in set.taps]
    PowerFlows._align_parallel_taps!(set.taps, dVdp, fill(false, 2))
    @test all(<(0), dVdp)
    @test all(d.aligned == -1 for d in set.taps)
    @test abs.(dVdp) ≈ abs.([s[d.name] for d in set.taps])   # magnitudes kept
end

@testset "tap direction: parallel taps move together, no tap fight" begin
    taps, v, _ = _solve_parallel()
    @test all(d.current <= d.initial for d in values(taps))   # neither raises its ratio
    @test taps["A"].current < taps["A"].initial
    @test 1.04 - PowerFlows.CONTROL_TAP_BAND_TOL <= v <= 1.05
    # PSS/E's CONT sign convention (CONT1 = +3, winding-2 side for both) also moves them
    # the same way.
    taps_c, v_c, _ = _solve_parallel(; direction = :cont_sign, cont1 = 3)
    @test all(d.orientation == -1 for d in values(taps_c))
    @test all(d.current <= d.initial for d in values(taps_c))
    @test 1.04 - PowerFlows.CONTROL_TAP_BAND_TOL <= v_c <= 1.05
end
