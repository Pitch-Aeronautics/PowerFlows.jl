# Tap orientation: by measured dV/dtap (default), by PSS/E's CONT1 convention (a terminal bus
# on its own winding's side, a remote bus by the sign)
# (`tap_direction = :cont_sign`), and parallel taps on one regulated bus aligned.

"""REF(1) ─line(x = `xline`)─ F(2) ─tap F→T─ T(3) with a PV generator at T (|V₃| = 1.0) and a
load at F. The tap regulates its own winding-1 bus F (`regulated_bus_number = 2`), band
[`vmi`, `vma`] as the PSS/E parser stores it; with `remote`, a load bus R(4) hangs off F by a
short line and the tap regulates R instead, a bus that is neither terminal. With T held,
raising the ratio pushes reactive power into F (and R), so the measured dV/dtap is positive.
PSS/E puts a terminal bus on its own winding's side whatever CONT1's sign, so F is on the
winding-1 side; R is placed by the sign, CONT1 > 0 reading "winding-2 side", where raising the
ratio lowers the voltage."""
function _winding1_regulated_tap_system(; cont1 = 2, vmi = 1.0, vma = 1.02, tap = 1.0,
    xline = 0.2, psse_ext::Bool = false, remote::Bool = false)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    bf = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    bt = _add_simple_bus!(sys, 3, ACBusTypes.PV, 115, 1.0, 0.0)
    _add_simple_line!(sys, b1, bf, 0.01, xline, 0.0)
    _add_simple_load!(sys, bf, 20.0, 15.0)
    _add_simple_thermal_standard!(sys, bt, 0.0, 0.0)
    if remote
        br = _add_simple_bus!(sys, 4, ACBusTypes.PQ, 230, 1.0, 0.0)
        _add_simple_line!(sys, bf, br, 0.002, 0.01, 0.0)
        _add_simple_load!(sys, br, 2.0, 1.0)
    end
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

_orientation(sys) = PowerFlows._tap_cont_orientation(only(get_components(TapTransformer, sys)))

@testset "tap direction: CONT1 read from ext, terminal buses by position, option plumbing" begin
    # A terminal bus is on its own winding's side whatever the sign: the from (winding-1) bus
    # +1, the to (winding-2) bus −1.
    @test _orientation(_winding1_regulated_tap_system(; cont1 = 2)) == 1
    @test _orientation(_winding1_regulated_tap_system(; cont1 = -2)) == 1
    @test _orientation(_winding1_regulated_tap_system(; cont1 = 2, psse_ext = true)) == 1
    @test _orientation(_winding1_regulated_tap_system(; cont1 = 3)) == -1
    @test _orientation(_winding1_regulated_tap_system(; cont1 = -3)) == -1
    @test _orientation(_winding1_regulated_tap_system(; cont1 = -3, psse_ext = true)) == -1
    # A remote bus is placed by the sign: > 0 the winding-2 side, < 0 the winding-1 side.
    @test _orientation(_winding1_regulated_tap_system(; cont1 = 4, remote = true)) == -1
    @test _orientation(_winding1_regulated_tap_system(; cont1 = -4, remote = true)) == 1
    @test _orientation(
        _winding1_regulated_tap_system(; cont1 = -4, remote = true, psse_ext = true)) == 1
    # An API-built tap has no CONT1: measured orientation.
    @test _orientation(_make_solvable_tap_shunt_system()) == 0
    sys = _winding1_regulated_tap_system(; cont1 = 4, remote = true)
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

@testset "tap direction: a tap regulating its own winding-1 bus raises the ratio, whatever CONT1's sign" begin
    sys0 = _winding1_regulated_tap_system()
    base = PowerFlowData(ACPolarPowerFlow(), sys0)
    @test solve_power_flow!(base)
    v0 = base.bus_magnitude[PF.get_bus_lookup(base)[2], 1]
    @test v0 < 1.0                                     # below the band [1.0, 1.02]
    # Measured: raising the ratio raises |V_F|; the tap steps up into the band.
    d_s, v_s, _ = _solve_direction(_winding1_regulated_tap_system(), :sensitivity)
    @test d_s.current > d_s.initial
    @test 1.0 - PowerFlows.CONTROL_TAP_BAND_TOL <= v_s <= 1.02
    # PSS/E puts F on winding 1 by position, for CONT1 = +2 (as PSS/E files store a terminal
    # bus) and −2 alike: the same answer as the measured orientation.
    for c in (2, -2)
        d_c, v_c, _ = _solve_direction(_winding1_regulated_tap_system(; cont1 = c),
            :cont_sign)
        @test d_c.orientation == 1
        @test d_c.current == d_s.current
        @test v_c ≈ v_s atol = 1e-8
    end
end

@testset "tap direction: a remote regulated bus is placed by CONT1's sign" begin
    base = PowerFlowData(ACPolarPowerFlow(),
        _winding1_regulated_tap_system(; cont1 = 4, remote = true))
    @test solve_power_flow!(base)
    v0 = base.bus_magnitude[PF.get_bus_lookup(base)[4], 1]
    @test v0 < 1.0
    sysr(c) = _winding1_regulated_tap_system(; cont1 = c, remote = true)
    d_s, v_s, _ = _solve_direction(sysr(4), :sensitivity)
    @test d_s.controlled_ix ∉ (d_s.from_ix, d_s.to_ix)
    @test d_s.current > d_s.initial
    @test 1.0 - PowerFlows.CONTROL_TAP_BAND_TOL <= v_s <= 1.02
    # CONT1 = +4 says "winding-2 side": lower the ratio to raise the voltage. R hangs off F, so
    # that lowers |V_R|, and the tap runs to its lower limit, short of the band.
    d_c, v_c, _ = _solve_direction(sysr(4), :cont_sign)
    @test d_c.orientation == -1
    @test d_c.current == d_c.p_min
    @test v_c < v0
    # CONT1 = −4 (winding-1 side) agrees with the measurement: same answer as :sensitivity.
    d_n, v_n, _ = _solve_direction(sysr(-4), :cont_sign)
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
    # PSS/E's CONT convention also moves them the same way.
    # T is both units' to bus, so a terminal on winding 2 whatever CONT1's sign.
    for c in (3, -3)
        taps_c, v_c, _ = _solve_parallel(; direction = :cont_sign, cont1 = c)
        @test all(d.orientation == -1 for d in values(taps_c))
        @test all(d.current <= d.initial for d in values(taps_c))
        @test 1.04 - PowerFlows.CONTROL_TAP_BAND_TOL <= v_c <= 1.05
    end
end

"""S(2) ═ two parallel units T1, T2 (R→S, x = 0.05) ═ R(3), with S fed from REF(1) by a short
line and a 300 MW / 100 MVAr load at R, which has no other connection. Both units regulate their
own winding-1 bus R (CONT1 = 3, the from bus: raising either ratio raises |V_R|) and sit 15
steps apart, T1 at 1.05 and T2 at 0.95625 (33 positions over [0.9, 1.1]), so a reactive current
circulates between them. Band [`vmi`, `vma`]."""
function _unequal_parallel_system(vmi, vma)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    bs = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    br = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 115, 1.0, 0.0)
    _add_simple_line!(sys, b1, bs, 0.002, 0.02, 0.0)
    _add_simple_load!(sys, br, 300.0, 100.0)
    for (nm, t) in (("T1", 1.05), ("T2", 0.95625))
        add_component!(sys, TapTransformer(; name = nm, available = true,
            active_power_flow = 0.0, reactive_power_flow = 0.0,
            arc = Arc(; from = br, to = bs), r = 0.002, x = 0.05, primary_shunt = 0.0im,
            tap = t, rating = 1.0, base_power = 100.0,
            control_objective = PSY.TransformerControlObjective.VOLTAGE,
            voltage_setpoint = (vmi + vma) / 2, regulated_bus_number = 3,
            ext = Dict{String, Any}("VMI1" => vmi, "VMA1" => vma, "CONT1" => 3)))
    end
    return sys
end

function _solve_unequal_parallel(vmi, vma, direction)
    pf = ACPolarPowerFlow(; control_discrete_devices = true, tap_direction = direction)
    data = PowerFlowData(pf, _unequal_parallel_system(vmi, vma))
    @test solve_power_flow!(data)
    taps = Dict(d.name => d for d in PF.get_controlled_devices(data).taps)
    return taps, data.bus_magnitude[PF.get_bus_lookup(data)[3], 1]
end

@testset "tap direction: parallel units at unequal ratios move by the same steps (GAP-058)" begin
    base = PowerFlowData(ACPolarPowerFlow(), _unequal_parallel_system(0.5, 1.5))
    @test solve_power_flow!(base)
    v0 = base.bus_magnitude[PF.get_bus_lookup(base)[3], 1]
    for (vmi, vma, dir) in ((v0 + 0.01, v0 + 0.03, 1), (v0 - 0.03, v0 - 0.01, -1)),
        direction in (:sensitivity, :cont_sign)

        taps, v = _solve_unequal_parallel(vmi, vma, direction)
        t1, t2 = taps["T1"], taps["T2"]
        step = t1.levels[2] - t1.levels[1]
        n1 = round(Int, (t1.current - t1.initial) / step)
        n2 = round(Int, (t2.current - t2.initial) / step)
        @test t1.current in t1.levels && t2.current in t2.levels
        @test n1 == n2                         # together, keeping the 15-step offset
        @test sign(n1) == dir                  # in the direction that corrects |V_R|
        @test vmi - PowerFlows.CONTROL_TAP_BAND_TOL <= v <= vma + PowerFlows.CONTROL_TAP_BAND_TOL
        @test t1.offset ≈ (1.05 - 0.95625) / 2 && t2.offset ≈ -(1.05 - 0.95625) / 2
    end
end
