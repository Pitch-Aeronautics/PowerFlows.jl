# Impedance correction (PSS/E tables, `PSY.ImpedanceCorrectionData`) in PowerFlows: PNM's
# `Ybus` scales a transformer's series impedance by the table's factor at its tap. Reported
# branch flows must use the same corrected block, and discrete tap control must re-evaluate the
# factor at every tap it applies.

# Factor 1.1 at the nominal tap, slope −2 per unit tap ratio: one step of the default
# 33-position [0.9, 1.1] grid (0.00625) changes the factor by 0.0125.
const _ICD_CURVE_POINTS = [(x = 0.9, y = 1.3), (x = 1.1, y = 0.9)]

_icd_factor(tap) = 1.3 + (tap - 0.9) * (0.9 - 1.3) / (1.1 - 0.9)

function _attach_tap_ratio_table!(sys, tx; points = _ICD_CURVE_POINTS)
    ict = ImpedanceCorrectionData(;
        table_number = 1,
        impedance_correction_curve = IS.PiecewiseLinearData(points),
        transformer_winding = WindingCategory.TR2W_WINDING,
        transformer_control_mode = ImpedanceCorrectionTransformerControlMode.TAP_RATIO,
    )
    add_supplemental_attribute!(sys, tx, ict)
    return ict
end

function _icd_tap_system()
    sys = _make_solvable_tap_shunt_system()
    tx = only(get_components(TapTransformer, sys))
    _attach_tap_ratio_table!(sys, tx)
    return sys, tx
end

# The transformer's 2×2 block with its series impedance scaled by `factor`, from first
# principles (PNM's `ybus_branch_entries` without `nr` is the uncorrected block).
function _corrected_block(tx, tap, factor)
    yt = 1 / ((get_r(tx) + im * get_x(tx)) * factor)
    t = tap * cis(get_α(tx))
    return (yt / abs2(t) + get_primary_shunt(tx), -yt / conj(t), -yt / t, yt)
end

function _terminal_voltages(data, tx, ts = 1)
    bl = PF.get_bus_lookup(data)
    arc = get_arc(tx)
    f = bl[get_number(get_from(arc))]
    t = bl[get_number(get_to(arc))]
    V(ix) = data.bus_magnitude[ix, ts] * cis(data.bus_angles[ix, ts])
    return V(f), V(t)
end

function _block_flows(block, Vf, Vt)
    (y11, y12, y21, y22) = block
    return Vf * conj(y11 * Vf + y12 * Vt), Vt * conj(y21 * Vf + y22 * Vt)
end

@testset "impedance correction: reported transformer flow uses the corrected block" begin
    sys, tx = _icd_tap_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    nrd = PF.get_network_reduction_data(data)
    @test PNM._impedance_correction_factor(tx, nrd) ≈ 1.1
    @test solve_power_flow!(data)
    Vf, Vt = _terminal_voltages(data, tx)
    S_ft, S_tf = _block_flows(_corrected_block(tx, get_tap(tx), 1.1), Vf, Vt)
    S_ft_nominal, _ = _block_flows(PNM.ybus_branch_entries(tx), Vf, Vt)
    # The case is loaded enough that the correction moves the flow well past the tolerances.
    @test abs(S_ft - S_ft_nominal) > 1e-3

    arc = PNM.get_arc_tuple(tx)
    entry = only(PF._compute_segment_flows(tx, data, arc, 1))
    @test entry.P_from_to ≈ real(S_ft) atol = 1e-10
    @test entry.Q_from_to ≈ imag(S_ft) atol = 1e-10
    @test entry.P_to_from ≈ real(S_tf) atol = 1e-10
    @test entry.Q_to_from ≈ imag(S_tf) atol = 1e-10
    @test entry.P_losses ≈ real(S_ft + S_tf) atol = 1e-10
    @test entry.Q_losses ≈ imag(S_ft + S_tf) atol = 1e-10
    # ... and agrees with the arc flow the solve stored (arc admittance matrices).
    ix = PF.get_arc_lookup(data)[arc]
    @test entry.P_from_to ≈ data.arc_active_power_flow_from_to[ix, 1] atol = 1e-10
    @test entry.Q_from_to ≈ data.arc_reactive_power_flow_from_to[ix, 1] atol = 1e-10

    # The DataFrame outputs agree, branch-level and arc-level.
    base = get_base_power(sys)
    res_branch = solve_power_flow(ACPolarPowerFlow(), sys, FlowReporting.BRANCH_FLOWS)
    res_arc = solve_power_flow(ACPolarPowerFlow(), sys, FlowReporting.ARC_FLOWS)
    fb = res_branch["flow_results"]
    row = only(eachrow(fb[fb.flow_name .== get_name(tx), :]))
    @test row.P_from_to ≈ real(S_ft) * base atol = 1e-6
    @test row.Q_from_to ≈ imag(S_ft) * base atol = 1e-6
    @test row.P_losses ≈ real(S_ft + S_tf) * base atol = 1e-6
    fa = res_arc["flow_results"]
    arow = only(eachrow(fa[(fa.bus_from .== arc[1]) .& (fa.bus_to .== arc[2]), :]))
    @test arow.P_from_to ≈ row.P_from_to atol = 1e-6
    @test arow.Q_from_to ≈ row.Q_from_to atol = 1e-6

    # Storing the solution: the recompute-vs-stored assertion holds and the component carries
    # the corrected flow.
    @test solve_and_store_power_flow!(ACPolarPowerFlow(), sys)
    @test get_active_power_flow(tx) ≈ real(S_ft) atol = 1e-6
    @test get_reactive_power_flow(tx) ≈ imag(S_ft) atol = 1e-6
end

@testset "impedance correction: a tap step re-evaluates the factor (in-place == rebuild)" begin
    sys, tx = _icd_tap_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    set = PowerFlows.build_controlled_device_set(
        sys, PF.get_bus_lookup(data), data.power_network_matrix)
    d = only(set.taps)
    @test !isnothing(d.correction)
    # At the enrollment tap the device's terms are the Y-bus's own corrected block.
    ff, ft, tf, tt = PF._branch_terms(d, d.current)
    @test collect((ff + get_primary_shunt(tx), ft, tf, tt)) ≈
          collect(_corrected_block(tx, 1.0, _icd_factor(1.0)))
    newtap = 1.05
    @test !(_icd_factor(newtap) ≈ _icd_factor(1.0))
    PowerFlows.apply_parameter!(d, data, newtap, 1)
    after = copy(data.power_network_matrix.data)
    # A fresh build evaluates the table at the new tap.
    set_tap!(tx, newtap)
    data_ref = PowerFlowData(ACPolarPowerFlow(), sys)
    @test PNM._impedance_correction_factor(tx, PF.get_network_reduction_data(data_ref)) ≈
          _icd_factor(newtap)
    @test after ≈ data_ref.power_network_matrix.data
    # Y22 moved too: it is tap-independent only without a correction table.
    A = data.power_network_matrix.data
    @test A.nzval[d.nz_offsets[4]] ≈ data_ref.power_network_matrix.data[d.to_ix, d.to_ix]
    # No allocation on the hot path with a table attached.
    let
        f() = PowerFlows.apply_parameter!(d, data, 1.05, 1)
        f()
        @test (@allocated f()) == 0
    end
end

@testset "impedance correction: tap sensitivity ∂S/∂p includes the factor's slope" begin
    sys, tx = _icd_tap_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    @test solve_power_flow!(data)
    set = PowerFlows.build_controlled_device_set(
        sys, PF.get_bus_lookup(data), data.power_network_matrix)
    d = only(set.taps)
    n = size(data.power_network_matrix.data, 1)
    rhs = zeros(2 * n)
    @test PF._dF_dp!(rhs, d, data, 1)
    V = data.bus_magnitude[:, 1] .* cis.(data.bus_angles[:, 1])
    S(Y) = V .* conj.(Y * V)
    p0 = d.current
    h = 1e-6
    PowerFlows.apply_parameter!(d, data, p0 + h, 1)
    S_plus = S(data.power_network_matrix.data)
    PowerFlows.apply_parameter!(d, data, p0 - h, 1)
    S_minus = S(data.power_network_matrix.data)
    PowerFlows.apply_parameter!(d, data, p0, 1)
    dS = (S_plus .- S_minus) ./ (2h)
    for b in (d.from_ix, d.to_ix)
        @test rhs[2b - 1] ≈ real(dS[b]) rtol = 1e-6
        @test rhs[2b] ≈ imag(dS[b]) rtol = 1e-6
    end
end

@testset "impedance correction: discrete control steps on the corrected impedance" begin
    sys, tx = _icd_tap_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    set = PF.get_controlled_devices(data)
    d = only(set.taps)
    p = d.current
    @test p != d.initial
    @test p in d.levels
    @test !(_icd_factor(p) ≈ _icd_factor(d.initial))
    # The Y-bus the voltages were solved on is the one a fresh build at the solved tap gives.
    sys_ref, tx_ref = _icd_tap_system()
    set_tap!(tx_ref, p)
    data_ref = PowerFlowData(ACPolarPowerFlow(), sys_ref)
    @test data.power_network_matrix.data ≈ data_ref.power_network_matrix.data
    # The controlled bus sits where that network puts it (one tap step of tolerance).
    Vf, Vt = _terminal_voltages(data, tx)
    @test abs(abs(Vt) - d.vset) < 0.02
    # Reported flow (branch level, solved tap, factor at the solved tap) and the stored arc
    # flow both equal V·conj(Y_corrected·V).
    S_ft, S_tf = _block_flows(_corrected_block(tx, p, _icd_factor(p)), Vf, Vt)
    arc = PNM.get_arc_tuple(tx)
    entry = only(PF._compute_segment_flows(tx, data, arc, 1))
    @test entry.P_from_to ≈ real(S_ft) atol = 1e-10
    @test entry.Q_from_to ≈ imag(S_ft) atol = 1e-10
    @test entry.P_to_from ≈ real(S_tf) atol = 1e-10
    @test entry.Q_to_from ≈ imag(S_tf) atol = 1e-10
    ix = PF.get_arc_lookup(data)[arc]
    @test data.arc_active_power_flow_from_to[ix, 1] ≈ real(S_ft) atol = 1e-10
    @test data.arc_reactive_power_flow_to_from[ix, 1] ≈ imag(S_tf) atol = 1e-10
    # The stored path: the tap is written back and the component carries the same flow.
    @test solve_and_store_power_flow!(pf, sys)
    @test get_tap(tx) == p
    @test get_active_power_flow(tx) ≈ real(S_ft) atol = 1e-6
end

@testset "impedance correction: a phase-angle table folds a fixed factor into yt" begin
    sys = _make_solvable_tap_shunt_system()
    tx = only(get_components(TapTransformer, sys))
    ict = ImpedanceCorrectionData(;
        table_number = 2,
        impedance_correction_curve = IS.PiecewiseLinearData([
            (x = -30.0, y = 1.2), (x = 30.0, y = 1.2)]),
        transformer_winding = WindingCategory.TR2W_WINDING,
        transformer_control_mode = ImpedanceCorrectionTransformerControlMode.PHASE_SHIFT_ANGLE,
    )
    add_supplemental_attribute!(sys, tx, ict)
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    set = PowerFlows.build_controlled_device_set(
        sys, PF.get_bus_lookup(data), data.power_network_matrix)
    d = only(set.taps)
    @test isnothing(d.correction)
    @test d.yt ≈ 1 / ((get_r(tx) + im * get_x(tx)) * 1.2)
    PowerFlows.apply_parameter!(d, data, 1.05, 1)
    set_tap!(tx, 1.05)
    @test data.power_network_matrix.data ≈
          PowerFlowData(ACPolarPowerFlow(), sys).power_network_matrix.data
end

@testset "impedance correction: area-tie self-term reads the corrected block" begin
    sys, tx = _icd_tap_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    nrd = PF.get_network_reduction_data(data)
    (y11, y22) = PF._primitive_diag(tx, nrd)
    block = _corrected_block(tx, get_tap(tx), 1.1)
    @test y11 ≈ block[1]
    @test y22 ≈ block[4]
end
