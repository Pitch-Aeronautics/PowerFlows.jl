# Phase-shifter MW control (GAP-029): a `PhaseShiftingTransformer` under PSS/E COD1 = 3 steps
# its angle over NTP1 positions in [RMI1, RMA1] (degrees) to hold its winding-1 → winding-2 MW
# flow, measured at the metered end, inside [VMI1, VMA1] (MW), stopping at the first position
# inside. Synthetic two-area systems only.

"""Area A: REF(1) ─line─ bus 2. Area B: load bus 3 (200 MW / 40 MVAr) and a PV unit at bus 4
(50 MW) on a short line. A and B are joined 2 → 3 by `length(angles)` parallel phase shifters
(x = `x_pst` each, angle `angles[k]` degrees) and a parallel path 2 → 5 → 3 (two lines,
x = `x_line` in all), so the shifters
share the ~150 MW interchange with the line and their angle moves MW between the two paths.
`ctrl` is a `Dict` of PSS/E control fields (COD1, RMI1, RMA1, NTP1, VMI1, VMA1, ...) put in each
shifter's `ext` (or `ext["psse"]` with `psse_ext`); `objective` is its `control_objective`."""
function _pst_two_area_system(; angles = [0.0], x_pst = 0.05, x_line = 0.1,
    ctrl = Dict{String, Any}("COD1" => 3, "RMI1" => -30.0, "RMA1" => 30.0, "NTP1" => 61,
        "VMI1" => 80.0, "VMA1" => 90.0),
    objective = PSY.TransformerControlObjective.ACTIVE_POWER_FLOW, psse_ext::Bool = false,
    phase_angle_limits = nothing, r_pst = 0.002)
    sys = System(100.0)
    area_a = Area(; name = "A")
    area_b = Area(; name = "B")
    add_component!(sys, area_a)
    add_component!(sys, area_b)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    b4 = _add_simple_bus!(sys, 4, ACBusTypes.PV, 230, 1.0, 0.0)
    for (b, a) in ((b1, area_a), (b2, area_a), (b3, area_b), (b4, area_b))
        set_area!(b, a)
    end
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_line!(sys, b1, b2, 0.002, 0.02, 0.0)
    # The parallel path runs through bus 5 (area A): PowerNetworkMatrices v0.24.5 keeps one
    # branch per arc in its direct-branch map when a Line and a PhaseShiftingTransformer share an
    # arc, so the stored-flow check of `solve_and_store_power_flow!` would see only one of them.
    b5 = _add_simple_bus!(sys, 5, ACBusTypes.PQ, 230, 1.0, 0.0)
    set_area!(b5, area_a)
    _add_simple_line!(sys, b2, b5, 0.002, x_line / 2, 0.0)
    _add_simple_line!(sys, b5, b3, 0.002, x_line / 2, 0.0)
    _add_simple_line!(sys, b3, b4, 0.002, 0.03, 0.0)
    _add_simple_load!(sys, b3, 200.0, 40.0)           # on its 1 MVA base: 200 MW / 40 MVAr
    _add_simple_thermal_standard!(sys, b4, 0.5, 0.0)  # 50 MW
    for (k, a) in enumerate(angles)
        ext = Dict{String, Any}()
        if psse_ext
            ext["psse"] = copy(ctrl)
        else
            merge!(ext, ctrl)
        end
        kw = isnothing(phase_angle_limits) ? (;) :
             (; phase_angle_limits = phase_angle_limits)
        pst = PhaseShiftingTransformer(; name = "pst$k", available = true,
            active_power_flow = 0.0, reactive_power_flow = 0.0,
            arc = Arc(; from = b2, to = b3), r = r_pst, x = x_pst, primary_shunt = 0.0im,
            tap = 1.0, α = deg2rad(a), rating = 5.0, base_power = 100.0,
            control_objective = objective, ext = ext, kw...)
        add_component!(sys, pst)
    end
    return sys
end

_pst_ctrl(; vmi, vma, rmi = -30.0, rma = 30.0, ntp = 61, cod = 3, extra...) =
    Dict{String, Any}("COD1" => cod, "RMI1" => rmi, "RMA1" => rma, "NTP1" => ntp,
        "VMI1" => vmi, "VMA1" => vma, (String(k) => v for (k, v) in extra)...)

function _pst_solve(sys; kwargs...)
    pf = ACPolarPowerFlow(; control_discrete_devices = true, kwargs...)
    data = PowerFlowData(pf, sys)
    ok = solve_power_flow!(data)
    return ok, data
end

# MW flow of shifter `d` at its metered end in the solved `data`.
_pst_mw(d, data) = 100.0 * PowerFlows._metered_flow(d, data, 1)

# The same system with every shifter fixed at `angles_deg` (no control): its metered MW flows.
function _pst_fixed_mw(angles_deg; kwargs...)
    sys = _pst_two_area_system(; angles = angles_deg, kwargs...)
    for p in get_components(PhaseShiftingTransformer, sys)
        set_control_objective!(p, PSY.TransformerControlObjective.FIXED)
    end
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    @test solve_power_flow!(data)
    set = PowerFlows.build_controlled_device_set(
        _pst_two_area_system(; angles = angles_deg, kwargs...),
        PF.get_bus_lookup(data), data.power_network_matrix)
    return [_pst_mw(d, data) for d in set.phase_shifters]
end

@testset "phase shifter: enrollment, field contract and options" begin
    sys = _pst_two_area_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    bl = PF.get_bus_lookup(data)
    build(s; kw...) = PowerFlows.build_controlled_device_set(s, PF.get_bus_lookup(data),
        data.power_network_matrix; kw...)
    d = only(build(sys).phase_shifters)
    @test d.p_min ≈ deg2rad(-30.0) && d.p_max ≈ deg2rad(30.0)
    @test length(d.levels) == 61 && d.levels[2] - d.levels[1] ≈ deg2rad(1.0)
    @test d.band && d.v_lo ≈ 0.8 && d.v_hi ≈ 0.9   # MW over the 100 MVA base
    @test d.metered == 1 && d.from_ix == bl[2] && d.to_ix == bl[3]
    # Our ingest's spelling, ext["psse"], and COD1 = 3 with an undefined objective.
    @test length(build(_pst_two_area_system(; psse_ext = true)).phase_shifters) == 1
    @test length(build(_pst_two_area_system(;
        objective = PSY.TransformerControlObjective.UNDEFINED)).phase_shifters) == 1
    # COD1 = −3 (PSS/E: adjustment suppressed) and other objectives keep the stored angle.
    @test isempty(build(_pst_two_area_system(;
        objective = PSY.TransformerControlObjective.ACTIVE_POWER_FLOW_DISABLED)).phase_shifters)
    @test isempty(build(_pst_two_area_system(;
        objective = PSY.TransformerControlObjective.UNDEFINED,
        ctrl = _pst_ctrl(; vmi = 80.0, vma = 90.0, cod = -3))).phase_shifters)
    @test isempty(build(_pst_two_area_system(;
        objective = PSY.TransformerControlObjective.FIXED)).phase_shifters)
    # The option.
    @test isempty(build(sys; phase_shifter_control = false).phase_shifters)
    @test ACPolarPowerFlow().phase_shifter_control
    @test ACRectangularPowerFlow().phase_shifter_control
    @test ACMixedPowerFlow().phase_shifter_control
    @test !ACPolarPowerFlow(; phase_shifter_control = false).phase_shifter_control
    # Incomplete control data: locked, with a warning.
    no_band = Dict{String, Any}("COD1" => 3, "RMI1" => -30.0, "RMA1" => 30.0)
    @test isempty((@test_logs (:warn, r"no MW band") match_mode = :any build(
        _pst_two_area_system(; ctrl = no_band))).phase_shifters)
    no_lims = Dict{String, Any}("COD1" => 3, "VMI1" => 80.0, "VMA1" => 90.0)
    @test isempty((@test_logs (:warn, r"no angle limits") match_mode = :any build(
        _pst_two_area_system(; ctrl = no_lims))).phase_shifters)
    # Without RMI1/RMA1, phase_angle_limits is read when it is not PowerSystems' default ±π.
    d = only(build(_pst_two_area_system(; ctrl = no_lims,
        phase_angle_limits = (min = -0.2, max = 0.3))).phase_shifters)
    @test d.p_min == -0.2 && d.p_max == 0.3
    # NTP1 absent: PSS/E's 33 positions; below 2: continuous.
    d = only(build(_pst_two_area_system(; ctrl = Dict{String, Any}("COD1" => 3,
        "RMI1" => -30.0, "RMA1" => 30.0, "VMI1" => 80.0, "VMA1" => 90.0))).phase_shifters)
    @test length(d.levels) == 33
    @test isempty(only(build(_pst_two_area_system(;
        ctrl = _pst_ctrl(; vmi = 80.0, vma = 90.0, ntp = 1))).phase_shifters).levels)
    # NMETR = 1 meters winding 2; the flow is still winding 1 → winding 2.
    d2 = only(build(_pst_two_area_system(;
        ctrl = _pst_ctrl(; vmi = 80.0, vma = 90.0, NMETR = 1))).phase_shifters)
    @test d2.metered == 2
    @test solve_power_flow!(data)
    d1 = only(build(sys).phase_shifters)
    p1 = PowerFlows._metered_flow(d1, data, 1)
    p2 = PowerFlows._metered_flow(d2, data, 1)
    @test p1 > p2 > 0.0                     # the series loss lies between the two ends
    @test p1 - p2 < 0.01
end

@testset "phase shifter: linear dP/dα matches a finite difference" begin
    for angles in ([0.0], [0.0, 2.0])
        sys = _pst_two_area_system(; angles = angles, x_pst = 0.1,
            ctrl = _pst_ctrl(; vmi = 40.0, vma = 50.0))
        pf = ACPolarPowerFlow(; control_discrete_devices = true)
        data = PowerFlowData(pf, sys)
        @test PF._solve_with_q_limits!(pf, data, 1)
        set = PF.get_controlled_devices(data)
        n = length(set.phase_shifters)
        frozen = fill(false, n)
        dVdp = zeros(n)
        PowerFlows._align_parallel_taps!(set.phase_shifters, dVdp, frozen)
        ctx = PowerFlows._build_sensitivity_context(data, 1)
        snap = PowerFlows._snapshot_state(data, 1)
        for d in set.phase_shifters
            g_lin, ok = PowerFlows._linear_plant_sign(d, data, 1, ctx)
            @test ok
            g_fd, rel = PowerFlows._plant_sign(d, data, 1, pf, snap)
            @test rel
            @test g_lin < 0.0                  # raising α lowers the winding-1 → 2 flow
            @test g_lin ≈ g_fd rtol = 2e-2
        end
    end
end

@testset "phase shifter: out of band, first step inside" begin
    # Base flow at α = 0 is ~101 MW on the shifter (x 0.05 against the line's 0.1).
    base = only(_pst_fixed_mw([0.0]))
    @test 95.0 < base < 110.0
    for (vmi, vma) in ((70.0, 85.0), (115.0, 125.0))
        ok, data = _pst_solve(_pst_two_area_system(; ctrl = _pst_ctrl(; vmi, vma)))
        @test ok
        set = PF.get_controlled_devices(data)
        d = only(set.phase_shifters)
        mw = _pst_mw(d, data)
        @test vmi - 1e-4 <= mw <= vma + 1e-4
        step = d.levels[2] - d.levels[1]
        k = round(Int, (d.current - d.levels[1]) / step) + 1
        @test d.current == d.levels[k]           # on a position
        # Above the band the angle rises (raising α lowers the flow), below it falls.
        moved = Int(sign(d.current - d.initial))
        @test moved == (base > vma ? 1 : -1)
        # One position back toward the start the flow is still outside: the first inside.
        back = only(_pst_fixed_mw([rad2deg(d.current - moved * step)];
            ctrl = _pst_ctrl(; vmi, vma)))
        @test moved > 0 ? back > vma : back < vmi
        # Equal to the fixed-angle solve at the final angle.
        @test only(_pst_fixed_mw([rad2deg(d.current)]; ctrl = _pst_ctrl(; vmi, vma))) ≈
              mw atol = 1e-6
        @test PF.get_control_batched_fallback_count(data) == 0
        res = PF.get_controlled_device_results(data)
        row = only(eachrow(res[res.family .== "PhaseShiftingTransformer", :]))
        @test row.name == "pst1" && row.final == d.current && row.initial == 0.0
    end
end

@testset "phase shifter: in band holds its angle" begin
    for a in (0.0, 3.0)
        mw0 = only(_pst_fixed_mw([a]))
        ctrl = _pst_ctrl(; vmi = mw0 - 8.0, vma = mw0 + 8.0)
        ok, data = _pst_solve(_pst_two_area_system(; angles = [a], ctrl = ctrl))
        @test ok
        d = only(PF.get_controlled_devices(data).phase_shifters)
        @test d.current == d.initial == deg2rad(a)
        @test _pst_mw(d, data) ≈ mw0 atol = 1e-6
    end
end

@testset "phase shifter: angle limits hold" begin
    # The band needs ~8° more than RMA1 = 5° allows: the shifter stops at its limit.
    ctrl = _pst_ctrl(; vmi = 5.0, vma = 15.0, rmi = -5.0, rma = 5.0, ntp = 11)
    ok, data = _pst_solve(_pst_two_area_system(; ctrl = ctrl))
    @test ok
    d = only(PF.get_controlled_devices(data).phase_shifters)
    @test d.current == d.p_max == deg2rad(5.0)
    @test _pst_mw(d, data) > 15.0
    # And the lower limit, below the band.
    ctrl = _pst_ctrl(; vmi = 190.0, vma = 200.0, rmi = -4.0, rma = 4.0, ntp = 9)
    ok, data = _pst_solve(_pst_two_area_system(; ctrl = ctrl))
    @test ok
    d = only(PF.get_controlled_devices(data).phase_shifters)
    @test d.current == d.p_min == deg2rad(-4.0)
    @test _pst_mw(d, data) < 190.0
    @test PF.get_control_batched_fallback_count(data) == 0
end

@testset "phase shifter: parallel units move together" begin
    # Two units (x = 0.1 each) beside the line. One degree on one unit alone circulates ~17 MW
    # between them, so a unit stepping alone would split the pair: they must step together.
    ctrl = _pst_ctrl(; vmi = 40.0, vma = 48.0)
    # Equal angles: the pair carries ~50 MW each, above the band, so both raise their angle.
    # Four degrees apart: the mean is ~38.7 MW, below it, so both lower theirs by the same step.
    for (angles, dir) in (([0.0, 0.0], 1), ([0.0, 4.0], -1))
        mean0 = sum(_pst_fixed_mw(angles; x_pst = 0.1, ctrl = ctrl)) / 2
        @test dir > 0 ? mean0 > 48.0 : mean0 < 40.0
        ok, data = _pst_solve(_pst_two_area_system(; angles = angles, x_pst = 0.1,
            ctrl = ctrl))
        @test ok
        set = PF.get_controlled_devices(data)
        a, b = set.phase_shifters
        step = a.levels[2] - a.levels[1]
        # Equal steps from where each started, the offset kept.
        @test a.current - a.initial ≈ b.current - b.initial atol = 1e-12
        @test Int(sign(a.current - a.initial)) == dir
        @test length(a.group) == 2 && a.group === b.group
        # The pair's mean flow is in the band, and one step back together it is not.
        mean_mw = (_pst_mw(a, data) + _pst_mw(b, data)) / 2
        @test 40.0 - 1e-4 <= mean_mw <= 48.0 + 1e-4
        back = sum(_pst_fixed_mw(rad2deg.([a.current, b.current] .- dir * step);
            x_pst = 0.1, ctrl = ctrl)) / 2
        @test dir > 0 ? back > 48.0 : back < 40.0
        @test PF.get_control_batched_fallback_count(data) == 0
    end
end

@testset "phase shifter: batched path, multiperiod and write-back" begin
    ctrl = _pst_ctrl(; vmi = 70.0, vma = 85.0)
    ok, data = _pst_solve(_pst_two_area_system(; ctrl = ctrl))
    @test ok
    @test PF.get_control_batched_fallback_count(data) == 0
    @test PF.get_control_inner_solve_count(data) > 0
    d = only(PF.get_controlled_devices(data).phase_shifters)
    # Two time steps reproduce the single-step answer at each step.
    pf2 = ACPolarPowerFlow(; control_discrete_devices = true, time_steps = 2)
    data2 = PowerFlowData(pf2, _pst_two_area_system(; ctrl = ctrl))
    @test solve_power_flow!(data2)
    set2 = PF.get_controlled_devices(data2)
    @test set2.pst_current[1, 1] == set2.pst_current[1, 2] == d.current
    # solve_and_store writes the angle back; the stored flows pass the consistency check.
    sys = _pst_two_area_system(; ctrl = ctrl)
    @test solve_and_store_power_flow!(ACPolarPowerFlow(; control_discrete_devices = true),
        sys)
    pst = get_component(PhaseShiftingTransformer, sys, "pst1")
    @test get_α(pst) == d.current
    mw = PSY.with_units_base(sys, PSY.UnitSystem.NATURAL_UNITS) do
        get_active_power_flow(pst)
    end
    @test 70.0 - 1e-3 <= mw <= 85.0 + 1e-3
    # Two parallel units: both angles written back. (Through `write_device_settings!`, not
    # `solve_and_store_power_flow!`: PowerNetworkMatrices v0.24.5 keeps only one of two phase
    # shifters on one arc in its direct-branch map, so the stored-flow check fails on that arc
    # with or without control.)
    sys = _pst_two_area_system(; angles = [0.0, 0.0], x_pst = 0.1,
        ctrl = _pst_ctrl(; vmi = 40.0, vma = 48.0))
    data4 = PowerFlowData(ACPolarPowerFlow(; control_discrete_devices = true), sys)
    @test solve_power_flow!(data4)
    PF.write_device_settings!(sys, data4)
    @test get_α(get_component(PhaseShiftingTransformer, sys, "pst1")) ==
          get_α(get_component(PhaseShiftingTransformer, sys, "pst2")) == deg2rad(1.0)
    # Option off: the angle is held.
    ok, data3 = _pst_solve(_pst_two_area_system(; ctrl = ctrl);
        phase_shifter_control = false)
    @test ok
    @test isnothing(PF.get_controlled_devices(data3))
end

# T68 (PowerNetworkMatrices): a phase shifter sharing its arc with a line, or with another phase
# shifter, is a parallel set in the branch maps (before, the direct map kept only the arc's last
# branch, so the stored-flow check failed on that arc). Each branch gets its own flow.
@testset "phase shifter beside another branch on one arc: flows stored per branch (T68)" begin
    for other in (:line, :pst)
        sys = System(100.0)
        b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
        _add_simple_source!(sys, b1, 0.0, 0.0)
        b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
        _add_simple_load!(sys, b2, 50.0, 10.0)
        mk(name, α) = PhaseShiftingTransformer(; name = name, available = true,
            active_power_flow = 0.0, reactive_power_flow = 0.0, arc = Arc(; from = b1, to = b2),
            r = 0.005, x = 0.1, primary_shunt = 0.0, tap = 1.0, α = α, rating = 1.0,
            base_power = 100.0)
        add_component!(sys, mk("pA", deg2rad(3.0)))
        if other == :line
            _add_simple_line!(sys, b1, b2, 0.005, 0.1, 0.0)
        else
            add_component!(sys, mk("pB", deg2rad(-2.0)))
        end
        data = PowerFlowData(ACPolarPowerFlow(), sys)
        @test solve_power_flow!(data)
        arc_p = data.arc_active_power_flow_from_to[PF.get_arc_lookup(data)[(1, 2)], 1]
        @test solve_and_store_power_flow!(ACPolarPowerFlow(), sys)
        p = PSY.with_units_base(sys, PSY.UnitSystem.SYSTEM_BASE) do
            [get_active_power_flow(br) for br in get_components(ACTransmission, sys)]
        end
        @test length(p) == 2
        @test sum(p) ≈ arc_p atol = 1e-8
        @test abs(p[1] - p[2]) > 0.05          # the angles split the arc unequally
    end
end
