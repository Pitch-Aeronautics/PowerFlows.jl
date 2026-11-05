const IEEE14_FACTS_RAW = joinpath(TEST_DATA_DIR, "14_bus.raw")

"""Build the IEEE 14-bus system (PSS/E `14_bus.raw`) with a `FACTSControlDevice` at bus 14
(`control_mode=NML`). Reused by discrete-control / reactive-power-control tests.

`stress` scales every `StandardLoad`'s ZIP P/Q components (the PSS/E parser's load type);
`shunt9_off` disables the bus-9 fixed shunt so the FACTS device carries more of the
reactive burden. `svc`/`shunt_control_type` selects the SVC vs STATCOM reactive-limit law;
`regulated_bus_number` sets FCREG (0 ⇒ local/sending-bus regulation)."""
function build_ieee14_facts_system(;
    regulated_bus_number::Int = 0,
    shmx_mva::Float64 = 25.0,
    mva_cap::Float64 = 9999.0,
    svc::Bool = false,
    vset::Float64 = 1.0,
    stress::Float64 = 1.0,
    shunt9_off::Bool = false,
)
    sys = System(IEEE14_FACTS_RAW; runchecks = false)
    if !isone(stress)
        for load in get_components(StandardLoad, sys)
            set_constant_active_power!(load, get_constant_active_power(load) * stress)
            set_constant_reactive_power!(
                load, get_constant_reactive_power(load) * stress)
            set_impedance_active_power!(load, get_impedance_active_power(load) * stress)
            set_impedance_reactive_power!(
                load, get_impedance_reactive_power(load) * stress)
            set_current_active_power!(load, get_current_active_power(load) * stress)
            set_current_reactive_power!(
                load, get_current_reactive_power(load) * stress)
        end
    end
    if shunt9_off
        for fa in get_components(PSY.FixedAdmittance, sys)
            get_number(get_bus(fa)) == 9 && set_available!(fa, false)
        end
    end
    shunt_control_type = PSY.FACTSShuntControlType.STATCOM
    if svc
        shunt_control_type = PSY.FACTSShuntControlType.SVC
    end
    facts = FACTSControlDevice(;
        name = "facts_14",
        available = true,
        bus = get_bus(sys, 14),
        control_mode = PSY.FACTSOperationModes.NML,
        voltage_setpoint = vset,
        max_shunt_current = shmx_mva,
        max_reactive_power = mva_cap,
        shunt_control_type = shunt_control_type,
        regulated_bus_number = regulated_bus_number,
    )
    add_component!(sys, facts)
    return sys
end

@testset "discrete control: FACTS enrolled by default (no experimental flag)" begin
    sys = build_ieee14_facts_system()
    pf = ACPowerFlow(; control_discrete_devices = true)
    data = PowerFlows.PowerFlowData(pf, sys)
    set = PowerFlows.get_controlled_devices(data)
    @test !isnothing(set)
    @test length(set.facts) == 1
end

@testset "FACTS remote FCREG regulation targets the remote bus" begin
    sys = build_ieee14_facts_system(; regulated_bus_number = 13)   # device at 14, regulates 13
    pf = ACPowerFlow(; control_discrete_devices = true)
    data = PowerFlows.PowerFlowData(pf, sys)
    set = PowerFlows.get_controlled_devices(data)
    f = only(set.facts)
    bl = PowerFlows.get_bus_lookup(data)
    @test PowerFlows.controlled_bus_ix(f) == bl[13]   # remote, not the device's own bus 14
    @test PowerFlows.controlled_bus_ix(f) != f.bus_ix
end

@testset "discrete control: pf field defaults" begin
    @test ACPolarPowerFlow().control_discrete_devices == false
    @test ACRectangularPowerFlow().control_discrete_devices == false
    @test ACMixedPowerFlow().control_discrete_devices == false
    @test ACPolarPowerFlow(; control_discrete_devices = true).control_discrete_devices ==
          true
    @test ACRectangularPowerFlow(;
        control_discrete_devices = true,
    ).control_discrete_devices ==
          true
    @test ACMixedPowerFlow(; control_discrete_devices = true).control_discrete_devices ==
          true
    @test PowerFlows.get_control_discrete_devices(ACPolarPowerFlow()) == false
    @test PowerFlows.get_control_discrete_devices(
        ACPolarPowerFlow(; control_discrete_devices = true),
    ) ==
          true
    @test PowerFlows.get_control_discrete_devices(DCPowerFlow()) == false
end

@testset "discrete control: build set" begin
    sys = _make_tap_shunt_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    bus_lookup = PF.get_bus_lookup(data)
    ybus = data.power_network_matrix
    set = PowerFlows.build_controlled_device_set(sys, bus_lookup, ybus)
    @test set isa PowerFlows.ControlledDeviceSet
    @test length(set.taps) == 1
    @test length(set.shunts) == 1
    t = set.taps[1]
    @test 0.9 <= t.p_min < t.p_max <= 1.1
    @test t.controlled_ix == t.to_ix
    @test t.vset > 0.0
    @test t.vset == 1.0
    s = set.shunts[1]
    @test s.b_min <= s.b0 <= s.b_max
    @test length(s.block_n) == length(s.block_dB)
    @test s.b0 == 0.0
    @test s.b_min == 0.0
    @test s.b_max == 0.2
    @test s.current == 0.0
    @test s.vset == (0.9 + 1.1) / 2
end

@testset "discrete control: per-ts store width validation" begin
    sys = _make_tap_shunt_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    # Default n_time_steps=1: matches single-ts data, rejects any other horizon.
    set = PowerFlows.build_controlled_device_set(
        sys,
        PF.get_bus_lookup(data),
        data.power_network_matrix,
    )
    @test PowerFlows.store_time_steps(set) == 1
    @test PowerFlows.validate_device_store_width(set, 1) === nothing
    @test_throws ErrorException PowerFlows.validate_device_store_width(set, 3)
    # No discrete control enrolled: validation is a dispatched no-op.
    @test PowerFlows.validate_device_store_width(nothing, 3) === nothing
end

@testset "discrete control: MODSW gating" begin
    function _build(mode)
        sys = _make_tap_shunt_system()
        sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
        PSY.set_control_mode!(sa, mode)
        data = PowerFlowData(ACPolarPowerFlow(), sys)
        return PowerFlows.build_controlled_device_set(
            sys, PF.get_bus_lookup(data), data.power_network_matrix)
    end

    # FIXED (locked): shunt not enrolled; tap unaffected.
    set0 = _build(PSY.SwitchedAdmittanceControlMode.FIXED)
    @test length(set0.shunts) == 0
    @test length(set0.taps) == 1

    # DISCRETE_VOLTAGE: enrolled, continuous == false.
    set1 = _build(PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE)
    @test length(set1.shunts) == 1
    @test set1.shunts[1].continuous == false

    # CONTINUOUS_VOLTAGE: enrolled, continuous == true.
    set2 = _build(PSY.SwitchedAdmittanceControlMode.CONTINUOUS_VOLTAGE)
    @test length(set2.shunts) == 1
    @test set2.shunts[1].continuous == true

    # Remote reactive-power / remote-device control modes: unsupported ⇒ the shunt is
    # de-enrolled with a warning and stays locked (PSS/E posture); never an error.
    unsupported = (
        PSY.SwitchedAdmittanceControlMode.DISCRETE_REACTIVE_PLANT,
        PSY.SwitchedAdmittanceControlMode.DISCRETE_REACTIVE_VSC,
        PSY.SwitchedAdmittanceControlMode.DISCRETE_ADMITTANCE_REMOTE,
    )
    for mode in unsupported
        setm = @test_logs (:warn, r"not supported") match_mode = :any _build(mode)
        @test length(setm.shunts) == 0
        @test length(setm.taps) == 1   # the tap is unaffected by the bad shunt record
    end
end

@testset "discrete control: warn-and-lock on unresolvable controlled bus" begin
    # A remote controlled-bus number that does not exist in the network must
    # de-enroll the device with a warning, not abort PowerFlowData construction.
    sys = _make_tap_shunt_system()
    tx = first(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_regulated_bus_number!(tx, 99)   # no bus 99
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    set = @test_logs (:warn, r"controlled bus 99") match_mode = :any (
        PowerFlows.build_controlled_device_set(
        sys, PF.get_bus_lookup(data), data.power_network_matrix)
    )
    @test length(set.taps) == 0
    @test length(set.shunts) == 1   # the shunt still enrolls
end

@testset "discrete control: tap reads first-class control fields" begin
    # The builder reads regulated_bus_number (controlled bus), tap_limits (ratio band),
    # number_of_tap_positions, and voltage_setpoint directly off the PSY component.
    sys = _make_tap_shunt_system()
    tx = first(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_regulated_bus_number!(tx, 3)
    PSY.set_tap_limits!(tx, (min = 0.88, max = 1.12))
    PSY.set_number_of_tap_positions!(tx, 25)
    PSY.set_voltage_setpoint!(tx, 1.03)
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    bl = PF.get_bus_lookup(data)
    set = PowerFlows.build_controlled_device_set(
        sys, bl, data.power_network_matrix)
    @test length(set.taps) == 1
    t = set.taps[1]
    @test t.controlled_ix == bl[3]
    @test t.p_min ≈ 0.88
    @test t.p_max ≈ 1.12
    @test length(t.levels) == 25
    @test t.vset ≈ 1.03
end

@testset "discrete control: implausible vset locks the device" begin
    # An API-built shunt whose admittance_limits hold actual susceptance bounds
    # (per the PSY docstring) would yield a garbage voltage setpoint; the builder
    # must de-enroll it with a warning instead of regulating |V| toward ~0.
    sys = _make_tap_shunt_system()
    sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
    PSY.set_admittance_limits!(sa, (min = -0.3, max = 0.3))
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    set = @test_logs (:warn, r"voltage setpoint") match_mode = :any (
        PowerFlows.build_controlled_device_set(
        sys, PF.get_bus_lookup(data), data.power_network_matrix)
    )
    @test length(set.shunts) == 0
end

@testset "discrete control: PSS/E parser shunt convention (Y = BINIT)" begin
    # A full-length zeroed initial_status marks the parser (BINIT) convention: Y holds
    # the TOTAL in-service admittance, so the reachable range is spanned by the blocks
    # alone (base 0) and the control baseline sits at BINIT.
    sys = _make_tap_shunt_system()
    sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
    PSY.set_control_mode!(sa, PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE)
    PSY.set_Y!(sa, 0.0 + 0.1im)   # BINIT = 0.1 p.u. (two of the four 0.05 blocks in)
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    set = PowerFlows.build_controlled_device_set(
        sys, PF.get_bus_lookup(data), data.power_network_matrix)
    s = set.shunts[1]
    @test s.b0 == 0.0            # block-counting base
    @test s.b_min == 0.0
    @test s.b_max ≈ 0.2
    @test s.current ≈ 0.1        # baseline at BINIT, inside the reachable range
end

@testset "discrete control: shunt controlled bus (regulated_bus_number)" begin
    function _shunt(regulated_bus_number = nothing)
        sys = _make_tap_shunt_system()
        sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
        isnothing(regulated_bus_number) ||
            PSY.set_regulated_bus_number!(sa, regulated_bus_number)
        data = PowerFlowData(ACPolarPowerFlow(), sys)
        bl = PF.get_bus_lookup(data)
        set = PowerFlows.build_controlled_device_set(
            sys, bl, data.power_network_matrix)
        return set.shunts[1], bl
    end

    # regulated_bus_number set ⇒ remote controlled bus.
    s_remote, bl = _shunt(2)
    @test s_remote.controlled_ix == bl[2]

    # regulated_bus_number 0 (default) ⇒ falls back to the shunt's local bus (3).
    s_local, bl2 = _shunt()
    @test s_local.controlled_ix == bl2[3]
end

@testset "discrete control: shunt invariant validation (warn-and-lock)" begin
    # b0 outside [b_min, b_max]: warn, device de-enrolled (returns false).
    r1 = @test_logs (:warn, r"outside") match_mode = :any PowerFlows._validate_shunt(
        "bad_shunt",
        0.0, # b_min
        0.5, # b0 — above b_max, outside [b_min, b_max]
        0.2, # b_max
        [4],
        [0.05],
    )
    @test r1 == false
    # Zero-step block with nonzero dB: malformed metadata → warn + false.
    r2 = @test_logs (:warn, r"zero steps") match_mode = :any PowerFlows._validate_shunt(
        "bad_shunt2",
        0.0,  # b_min
        0.0,  # b0
        0.2,  # b_max
        [0],  # zero steps
        [0.05], # nonzero dB — malformed
    )
    @test r2 == false
    # b_min == b_max (no controllable range): warn + false.
    r3 = @test_logs (:warn, r"no controllable") match_mode = :any (
        PowerFlows._validate_shunt("no_range", 0.5, 0.5, 0.5, [0], [0.0])
    )
    @test r3 == false
    # Valid case — true, no logs.
    @test PowerFlows._validate_shunt("ok_shunt", 0.0, 0.0, 0.2, [4], [0.05]) == true
end

@testset "discrete control: tap invariant validation (warn-and-lock)" begin
    # p_min > p_max (malformed): warn + false.
    r1 = @test_logs (:warn, r"exceeds") match_mode = :any (
        PowerFlows._validate_tap("bad_tap", 1.1, 0.9, 33)
    )
    @test r1 == false
    # p_min == p_max (no controllable range): warn + false.
    r2 = @test_logs (:warn, r"no controllable") match_mode = :any (
        PowerFlows._validate_tap("no_range", 1.0, 1.0, 33)
    )
    @test r2 == false
    # ntp < 2: a degenerate/locked changer must NOT become an active 33-level
    # controller; it is de-enrolled with a warning.
    r3 = @test_logs (:warn, r"fewer than 2 tap positions") match_mode = :any (
        PowerFlows._validate_tap("one_pos", 0.9, 1.1, 1)
    )
    @test r3 == false
    # Valid case — true.
    @test PowerFlows._validate_tap("ok_tap", 0.9, 1.1, 33) == true
end

@testset "tap ratio band comes from tap_limits" begin
    sys = _make_tap_shunt_system()
    tx = first(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_tap_limits!(tx, (min = 0.9, max = 1.1))
    md = PowerFlows._tap_metadata(tx, 2)
    @test md.pmin ≈ 0.9
    @test md.pmax ≈ 1.1
end

@testset "out-of-band initial tap ratio de-enrolls with a warning" begin
    sys = _make_tap_shunt_system()
    tx = first(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_tap!(tx, 1.5)   # far outside any [0.9, 1.1]-class band
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data =
        @test_logs (:warn, r"outside the tap-ratio band") match_mode = :any PowerFlowData(
            pf,
            sys,
        )
    set = PowerFlows.get_controlled_devices(data)
    @test isnothing(set) || isempty(set.taps)
end

@testset "discrete control: sigmoid law" begin
    # The raw sigmoid: midpoint at x = xset, saturating to hi (x ≪ xset) and
    # lo (x ≫ xset) when called as _sigmoid(lo, hi, ...) with hi > lo.
    S = 100.0
    @test PowerFlows._sigmoid(0.9, 1.1, S, 1.0, 1.0) ≈ 1.0 atol = 1e-9
    @test PowerFlows._sigmoid(0.9, 1.1, S, 1.5, 1.0) ≈ 0.9 atol = 1e-6
    @test PowerFlows._sigmoid(0.9, 1.1, S, 0.5, 1.0) ≈ 1.1 atol = 1e-6
    # Swapped limits give the increasing orientation.
    @test PowerFlows._sigmoid(1.1, 0.9, S, 1.5, 1.0) ≈ 1.1 atol = 1e-6
end

@testset "discrete control: negative-feedback orientation" begin
    # The effective control law `_control_target` must produce negative feedback:
    # sign(d target / dV) opposite to sign(dV/dp), so the closed-loop gain
    # g' = σ'(V)·dV/dp ≤ 0 for ANY device wiring — the orientation comes solely
    # from the measured plant sensitivity, for both signs of dV/dp.
    S = 100.0
    δ = 0.02
    tap = PowerFlows.ControlledTap("tp", 1, 2, 2, 1.0,
        1.0 / (0.01 + 0.1im), 0.0, 0.9, 1.1,
        collect(range(0.9, 1.1; length = 33)), (1, 2, 3, 4), 1.0, 1.0, 1.0)
    remote = PowerFlows.ControlledTap("ts", 1, 2, 1, 1.0,
        1.0 / (0.01 + 0.1im), 0.0, 0.9, 1.1,
        collect(range(0.9, 1.1; length = 33)), (1, 2, 3, 4), 1.0, 1.0, 1.0)
    shunt = PowerFlows.ControlledSwitchedShunt("sh", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [4], [0.05], 0.0, 0.2, zeros(Int, 1), false, 0.0, 0.0, false)
    for d in (tap, remote, shunt)
        vset = PowerFlows.voltage_setpoint(d)
        for dVdp in (1.0, -1.0)
            up = PowerFlows._control_target(d, vset + δ, S, dVdp)
            dn = PowerFlows._control_target(d, vset - δ, S, dVdp)
            # negative feedback: the slope in V and dV/dp have opposite signs.
            @test (up - dn) * dVdp <= 0.0
        end
    end
end

@testset "discrete control: relaxation yields monotone slope" begin
    # `_relaxation` must keep the damped fixed-point map slope NON-NEGATIVE
    # (monotone, no oscillation) for the worst-case gain. The slope is
    # m = 1 + ω·(g'−1) with g' ≤ 0 and |g'| ≤ gbound = 0.25|hi-lo|·S·|dVdp|, so
    # m ≥ 0 iff ω·(1+gbound) ≤ 1. (A negative slope is what previously made the
    # iterate alternate every step and tripped the oscillation-freeze detector.)
    d = PowerFlows.ControlledTap("t", 1, 2, 2, 1.0, 1.0 / (0.01 + 0.1im),
        0.0, 0.9, 1.1, collect(range(0.9, 1.1; length = 33)),
        (1, 2, 3, 4), 1.0, 1.0, 1.0)
    lo, hi = PowerFlows.parameter_limits(d)
    for S in (1.0e2, 1.0e3, 5.0e3), dVdp in (-5.0, -1.0, -0.1, 0.1, 1.0, 5.0)
        ω = PowerFlows._relaxation(d, S, dVdp)
        gbound = 0.25 * abs(hi - lo) * S * abs(dVdp)
        # ω = (1−θ)/(1+gbound) ≤ 1−θ = 0.5 by construction (no separate cap).
        @test 0.0 < ω <= 1.0 - PowerFlows.CONTROL_CONTRACTION
        @test ω * (1.0 + gbound) <= 1.0 + 1e-12   # ⇒ map slope m ≥ 0 (monotone)
    end
end

@testset "discrete control: ramp completes, tight regulation" begin
    # After the monotone-convergence fix the steepness ramp runs to completion
    # (no oscillation freeze at the initial steepness), so the controlled bus is
    # regulated to within one discrete tap step — previously it stalled ~5e-3 off
    # and only passed under a much looser +1e-2 tolerance.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    t = data.controlled_devices.taps[1]
    @test t.current in t.levels
    spacing = (t.p_max - t.p_min) / (length(t.levels) - 1)
    @test abs(data.bus_magnitude[t.controlled_ix, 1] - t.vset) < spacing
end

@testset "discrete control: continuous shunt not grid-pinned" begin
    sys = _make_solvable_tap_shunt_system()
    sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
    PSY.set_control_mode!(sa, PSY.SwitchedAdmittanceControlMode.CONTINUOUS_VOLTAGE)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    sh = data.controlled_devices.shunts[1]
    @test sh.continuous == true
    # final susceptance stays within the controllable band (continuous, not snapped).
    @test sh.b_min - 1e-9 <= sh.current <= sh.b_max + 1e-9
end

@testset "discrete control: snap" begin
    d = PowerFlows.ControlledTap("t", 1, 2, 2, 1.0, 1.0 + 0im,
        0.0, 0.9, 1.1, collect(range(0.9, 1.1; length = 5)),
        (1, 2, 3, 4), 1.0, 1.0, 1.0)  # levels: 0.9,0.95,1.0,1.05,1.1
    @test PowerFlows.snap_to_discrete(d, 1.03) == 1.05
    @test PowerFlows.snap_to_discrete(d, 1.20) == 1.1   # clamp
    block_dB_sh_snap = [0.05]
    sh = PowerFlows.ControlledSwitchedShunt("s", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [4], block_dB_sh_snap, 0.0, 0.2,
        zeros(Int, length(block_dB_sh_snap)),
        false, 0.0, 0.0, false)  # reachable: 0,0.05,0.10,0.15,0.20
    @test PowerFlows.snap_to_discrete(sh, 0.12) == 0.10
    @test sh.block_n == [2]
    block_dB_sh2 = [0.1, 0.02]
    sh2 = PowerFlows.ControlledSwitchedShunt("s2", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [2, 3], block_dB_sh2, 0.0, 0.26,
        zeros(Int, length(block_dB_sh2)),
        false, 0.0, 0.0, false)  # PSS/E cumulative chain: blocks activate in listed order
    # Chain totals: 0.1, 0.2 (block 1), then 0.22, 0.24, 0.26 (block 2).
    # 0.16 (= 1×0.1 + 3×0.02 with block 1 partially on) is NOT physically reachable;
    # nearest chain point to 0.17 is 0.2.
    @test PowerFlows.snap_to_discrete(sh2, 0.17) == 0.2
    @test sh2.block_n == [2, 0]
end

@testset "mixed-sign shunt snap reaches both chain sides" begin
    d = PowerFlows.ControlledSwitchedShunt(
        "mixed", 1, 1, 1.0, 0.95, 1.05, 0.0,
        0.0,                    # b0: all blocks off = neutral
        [1, 1],                 # one reactor step, one capacitor step
        [-0.5, 0.5],            # reactor listed first (RAW convention)
        -0.5, 0.5,              # envelope
        [0, 0], false, 0.0, 0.0, false)
    @test PowerFlows.snap_to_discrete(d, 0.45) == 0.5    # capacitive side reachable
    @test PowerFlows.snap_to_discrete(d, -0.45) == -0.5  # reactive side reachable
    @test iszero(PowerFlows.snap_to_discrete(d, 0.1))    # neutral still nearest
    @test (@allocated PowerFlows.snap_to_discrete(d, 0.45)) == 0
end

@testset "discrete control: continuous shunt no snap" begin
    # continuous == true ⇒ snap_to_discrete returns the clamped continuous value,
    # NOT the nearest reachable block grid point.
    block_dB = [0.05]
    cont = PowerFlows.ControlledSwitchedShunt("c", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [4], block_dB, 0.0, 0.2, zeros(Int, length(block_dB)), true, 0.0, 0.0, false)
    # 0.12 is between grid points 0.10 and 0.15; continuous must return it unchanged.
    @test PowerFlows.snap_to_discrete(cont, 0.12) == 0.12
    # clamped at the rails.
    @test PowerFlows.snap_to_discrete(cont, 0.30) == 0.2
    @test PowerFlows.snap_to_discrete(cont, -0.10) == 0.0
    # sanity: the discrete twin DOES snap 0.12 → 0.10.
    disc = PowerFlows.ControlledSwitchedShunt("d", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [4], block_dB, 0.0, 0.2, zeros(Int, length(block_dB)), false, 0.0, 0.0, false)
    @test PowerFlows.snap_to_discrete(disc, 0.12) == 0.10
end

@testset "discrete control: in-place Ybus == rebuild" begin
    sys = _make_tap_shunt_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    bus_lookup = PF.get_bus_lookup(data)
    ybus = data.power_network_matrix
    set = PowerFlows.build_controlled_device_set(sys, bus_lookup, ybus)
    @test length(set.taps) == 1
    d = set.taps[1]

    # Verify nz_offsets are now real (not all-ones stub).
    A = ybus.data
    fix = d.from_ix
    tix = d.to_ix
    @test A[fix, fix] ≈ A.nzval[d.nz_offsets[1]]
    @test A[fix, tix] ≈ A.nzval[d.nz_offsets[2]]
    @test A[tix, fix] ≈ A.nzval[d.nz_offsets[3]]
    @test A[tix, tix] ≈ A.nzval[d.nz_offsets[4]]

    # Apply a new tap ratio in-place and compare against a fresh rebuild.
    newtap = 1.05
    PowerFlows.apply_parameter!(d, data, newtap, 1)
    after = copy(A)

    # Rebuild reference: mutate sys, build fresh data2.
    txs = collect(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_tap!(txs[1], newtap)
    data2 = PowerFlowData(ACPolarPowerFlow(), sys)
    A2 = data2.power_network_matrix.data
    @test after ≈ A2

    # Parallel-branch variant: a Line paralleling the transformer aggregates
    # contributions in the same (fix,tix) off-diagonal slots.  Delta update is
    # required to preserve the parallel branch's contribution.
    @testset "discrete control: in-place Ybus == rebuild (parallel branch)" begin
        sys2 = _make_tap_shunt_system()
        txs2 = collect(PSY.get_components(PSY.TapTransformer, sys2))
        arc2 = PSY.get_arc(txs2[1])
        from_bus = PSY.get_from(arc2)
        to_bus = PSY.get_to(arc2)
        # Add a Line with distinct r,x so the off-diagonal slot accumulates two
        # different values (not a zero-sum cancellation of contributions).
        _add_simple_line!(sys2, from_bus, to_bus, 0.05, 0.15, 0.0)
        data2 = PowerFlowData(ACPolarPowerFlow(), sys2)
        bl = PF.get_bus_lookup(data2)
        ybus2 = data2.power_network_matrix
        set2 = PowerFlows.build_controlled_device_set(sys2, bl, ybus2)
        @test length(set2.taps) == 1
        d2 = set2.taps[1]
        newtap2 = 1.05
        PowerFlows.apply_parameter!(d2, data2, newtap2, 1)
        after2 = copy(ybus2.data)
        # Rebuild reference.
        PSY.set_tap!(txs2[1], newtap2)
        data2r = PowerFlowData(ACPolarPowerFlow(), sys2)
        @test after2 ≈ data2r.power_network_matrix.data
    end

    # Zero-allocation check: nzval writes only, no heap traffic.
    let
        f() = PowerFlows.apply_parameter!(d, data, 1.05, 1)
        f()                       # warm-up: force specialization
        @test (@allocated f()) == 0
    end
end

@testset "discrete control: PowerFlowData wiring" begin
    sys = _make_tap_shunt_system()
    # Default (control_discrete_devices = false): controlled_devices is nothing.
    data_off = PowerFlowData(ACPolarPowerFlow(), sys)
    @test isnothing(data_off.controlled_devices)

    # Enabled: controlled_devices is a ControlledDeviceSet with expected contents.
    data_on = PowerFlowData(ACPolarPowerFlow(; control_discrete_devices = true), sys)
    @test data_on.controlled_devices isa PowerFlows.ControlledDeviceSet
    @test length(data_on.controlled_devices.taps) == 1
    @test length(data_on.controlled_devices.shunts) == 1
end

@testset "discrete control: disabled == baseline" begin
    sys = _make_tap_shunt_system()
    a = PowerFlowData(ACPolarPowerFlow(), sys)
    b = PowerFlowData(ACPolarPowerFlow(; control_discrete_devices = false), sys)
    @test isnothing(a.controlled_devices)
    @test isnothing(b.controlled_devices)
end

@testset "discrete control: tap+shunt converges (NR)" begin
    # _make_solvable_tap_shunt_system has V_2 ~0.968 at tap=1.0; the tap has
    # full authority (dV/dp ≈ -1.04) and V_2 = vset = 1.0 is reachable at
    # tap ≈ 0.973, well inside [0.9, 1.1].  The damped steepness-homotopy
    # controller must drive the controlled bus into the vset deadband.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    t = data.controlled_devices.taps[1]
    @test t.current in t.levels
    @test abs(data.bus_magnitude[t.controlled_ix, 1] - t.vset) <=
          (t.p_max - t.p_min) / length(t.levels) + 1e-2
end

@testset "discrete control: primary-controlled tap orientation (NR)" begin
    # Exercises from-side (primary) control: the controlled bus is the tap's FROM
    # bus, so the measured plant sensitivity dV/dp has the opposite sign to the
    # usual to-side wiring, and `_control_target` must pick the orientation that
    # keeps the closed-loop gain negative. The to-side tests do NOT cover this.
    sys = _make_primary_controlled_tap_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    t = data.controlled_devices.taps[1]
    # The controlled bus is the tap's FROM bus (set via regulated_bus_number = 2).
    @test t.controlled_ix == t.from_ix
    @test t.current in t.levels
    @test abs(data.bus_magnitude[t.controlled_ix, 1] - t.vset) <=
          (t.p_max - t.p_min) / length(t.levels) + 1e-2
end

@testset "discrete control: snap + restore" begin
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    t = data.controlled_devices.taps[1]
    @test t.current in t.levels
    @test all(data.converged)
end

@testset "discrete control: refactor is no-op (disabled path)" begin
    # Verify that _ac_power_flow with no controlled devices routes through
    # _solve_with_q_limits! unchanged (pure code-motion regression check).
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    set_units_base_system!(sys, UnitSystem.SYSTEM_BASE)
    pf = ACPolarPowerFlow()
    data1 = PowerFlowData(pf, sys)
    data2 = PowerFlowData(pf, sys)
    @test isnothing(data1.controlled_devices)
    converged1 = solve_power_flow!(data1)
    converged2 = solve_power_flow!(data2)
    @test converged1
    @test converged2
    @test all(isapprox.(data1.bus_magnitude, data2.bus_magnitude; atol = 1e-12))
    @test all(isapprox.(data1.bus_angles, data2.bus_angles; atol = 1e-12))
end

@testset "discrete control: NR vs TR" begin
    # Same solvable fixture solved with NR and TR; the continuation engine must
    # drive the controlled bus into the strict vset deadband for both solvers.
    sys_nr = _make_solvable_tap_shunt_system()
    pf_nr = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data_nr = PowerFlowData(pf_nr, sys_nr)
    solve_power_flow!(data_nr)
    @test all(data_nr.converged)
    t_nr = data_nr.controlled_devices.taps[1]
    @test t_nr.current in t_nr.levels
    @test abs(data_nr.bus_magnitude[t_nr.controlled_ix, 1] - t_nr.vset) <=
          (t_nr.p_max - t_nr.p_min) / length(t_nr.levels) + 1e-2

    sys_tr = _make_solvable_tap_shunt_system()
    pf_tr = ACPolarPowerFlow{TrustRegionACPowerFlow}(; control_discrete_devices = true)
    data_tr = PowerFlowData(pf_tr, sys_tr)
    solve_power_flow!(data_tr)
    @test all(data_tr.converged)
    t_tr = data_tr.controlled_devices.taps[1]
    @test t_tr.current in t_tr.levels
    @test abs(data_tr.bus_magnitude[t_tr.controlled_ix, 1] - t_tr.vset) <=
          (t_tr.p_max - t_tr.p_min) / length(t_tr.levels) + 1e-2

    # Both solvers must agree on the controlled-bus voltage within a tight tolerance.
    @test isapprox(
        data_nr.bus_magnitude[t_nr.controlled_ix, 1],
        data_tr.bus_magnitude[t_tr.controlled_ix, 1];
        atol = 1e-4,
    )
end

@testset "discrete control: formulation-agnostic" begin
    # The outer continuation loop reads data.bus_magnitude[controlled_ix, ts]
    # after each inner solve.  For the solvable fixture the controlled bus is PQ,
    # so bus_magnitude is updated on every residual call for all three
    # formulations (polar: _update_residual_values!, rectangular CI:
    # rect_update_data!, mixed CPB: mixed_update_data!).  All three must
    # converge and satisfy the strict vset deadband.
    formulations = (
        ACPolarPowerFlow{NewtonRaphsonACPowerFlow},
        ACRectangularPowerFlow{NewtonRaphsonACPowerFlow},
        ACMixedPowerFlow{NewtonRaphsonACPowerFlow},
    )
    vmags = Float64[]
    for F in formulations
        sys = _make_solvable_tap_shunt_system()
        pf = F(; control_discrete_devices = true)
        data = PowerFlowData(pf, sys)
        solve_power_flow!(data)
        @test all(data.converged)
        t = data.controlled_devices.taps[1]
        @test t.current in t.levels
        @test abs(data.bus_magnitude[t.controlled_ix, 1] - t.vset) <=
              (t.p_max - t.p_min) / length(t.levels) + 1e-2
        push!(vmags, data.bus_magnitude[t.controlled_ix, 1])
    end
    # All three formulations must agree on the regulated voltage.
    @test maximum(vmags) - minimum(vmags) < 1e-4
end

@testset "discrete control: SVC holds weak bus to setpoint (NR)" begin
    # Ample reactive capability (100 MVA ⇒ b_lim = 1.0 p.u.): the SVC must inject
    # reactive power to drive the weak PQ bus up to its voltage_setpoint (1.0 p.u.).
    sys = _make_svc_system(; max_shunt_current = 100.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    f = data.controlled_devices.facts[1]
    @test abs(data.bus_magnitude[f.controlled_ix, 1] - f.vset) <= 5e-3
    # Regulating, not saturated at a susceptance limit.
    @test -f.b_lim < f.current < f.b_lim
end

@testset "discrete control: SVC clamps at reactive limit (NR)" begin
    # Tight reactive capability (5 MVA ⇒ b_lim = 0.05 p.u.) cannot supply the
    # bus's reactive deficit: the SVC saturates at b_lim and the bus stays below
    # setpoint — the homotopy equivalent of the PV→PQ Q-limit release.
    sys = _make_svc_system(; max_shunt_current = 5.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    f = data.controlled_devices.facts[1]
    @test data.bus_magnitude[f.controlled_ix, 1] < f.vset - 1e-3
    @test isapprox(f.current, f.b_lim; atol = 1e-3)
end

@testset "FACTS classification: regulating device is not saturated; results carry Q" begin
    # Ample capability: the device reaches setpoint with headroom, so it is not saturated
    # and never freezes short of both its limit and its setpoint.
    sys = _make_svc_system(; max_shunt_current = 100.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    f = data.controlled_devices.facts[1]
    @test !f.saturated
    at_setpoint = abs(data.bus_magnitude[f.controlled_ix, 1] - f.vset) <= 5e-3
    at_limit = abs(f.current) >= (1.0 - 1e-2) * f.b_lim
    @test at_setpoint || at_limit          # converged to setpoint OR genuinely at the bound
    @test !(f.saturated && at_setpoint)    # never both
    res = PowerFlows.get_controlled_device_results(data)
    frow = only(eachrow(res[res.family .== "FACTSControlDevice", :]))
    @test frow.saturated == false
    v_local = data.bus_magnitude[f.bus_ix, 1]
    @test frow.delivered_q_mvar ≈ f.current * v_local^2 * f.base_mva
    # Neutral columns for non-FACTS devices stay rectangular (no FACTS-only leakage).
    @test all(ismissing, res[res.family .!= "FACTSControlDevice", :delivered_q_mvar])
end

@testset "FACTS classification: clamped device is saturated and warns" begin
    # Tight capability cannot hold the setpoint: the device pins at b_lim off setpoint and is
    # reported saturated with a warning.
    sys = _make_svc_system(; max_shunt_current = 5.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @test_logs (:warn, r"saturated") match_mode = :any solve_power_flow!(data)
    f = data.controlled_devices.facts[1]
    @test f.saturated
    res = PowerFlows.get_controlled_device_results(data)
    frow = only(eachrow(res[res.family .== "FACTSControlDevice", :]))
    @test frow.saturated == true
end

@testset "FACTS write-back sets reactive_power_required" begin
    sys = _make_svc_system(; max_shunt_current = 100.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    PowerFlows.write_device_settings!(sys, data)
    fd = only(get_components(PSY.FACTSControlDevice, sys))
    # Solver-populated delivered Q = b·|V|²·base_mva at the device bus (capacitive ⇒ > 0).
    @test PSY.get_reactive_power_required(fd) > 0.0
end

@testset "discrete control: tap reads first-class PSY fields (#1684)" begin
    # Controllability set via the first-class PSY 5.12 fields (no ext); the builder
    # reads them directly.
    sys = _make_field_controlled_tap_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    t = data.controlled_devices.taps[1]
    @test t.p_min ≈ 0.85
    @test t.p_max ≈ 1.15
    @test length(t.levels) == 17
    @test t.vset ≈ 1.02
    # regulated_bus_number = 3 ⇒ remote controlled bus, not the to-bus.
    @test t.controlled_ix == PNM.get_bus_lookup(data.power_network_matrix)[3]
end

@testset "discrete control: construction guards" begin
    # Only NR/TR inner solvers are validated for the continuation.
    @test_throws ArgumentError ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}(;
        control_discrete_devices = true)
    @test_throws ArgumentError ACPolarPowerFlow{FastDecoupledACPowerFlow}(;
        control_discrete_devices = true)
    @test_throws ArgumentError ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}(;
        control_discrete_devices = true)
    @test_throws ArgumentError ACMixedPowerFlow{LevenbergMarquardtACPowerFlow}(;
        control_discrete_devices = true)
    # Construction is permissive: every device family supports time_steps>1 (taps via
    # reset-to-baseline), so a multi-step solver builds without inspecting the device mix.
    @test ACPolarPowerFlow(;
        control_discrete_devices = true, time_steps = 2) isa ACPolarPowerFlow
    # NR/TR with a single time step construct fine.
    @test ACPolarPowerFlow{TrustRegionACPowerFlow}(;
        control_discrete_devices = true) isa ACPolarPowerFlow
    # LCC systems are supported: the continuation checkpoint covers the LCC tail state.
    lcc_sys, _ = simple_lcc_system()
    @test PowerFlowData(
        ACPolarPowerFlow(; control_discrete_devices = true), lcc_sys) isa PowerFlowData
    @test PowerFlowData(
        ACPolarPowerFlow(; control_discrete_devices = true, time_steps = 2), lcc_sys) isa PowerFlowData
    # Tap-controlled transformers support multiple time steps via the reset-to-baseline
    # design (each step resets the shared Y-bus to the tap's enrollment value before
    # regulating): construction succeeds and the solve converges.
    tap_sys = _make_tap_shunt_system()
    tap_data = PowerFlowData(
        ACPolarPowerFlow(; control_discrete_devices = true, time_steps = 2), tap_sys)
    @test tap_data isa PowerFlowData
    @test solve_power_flow!(tap_data)
    # A shunt-ONLY (no taps) set supports multiple time steps and does NOT throw.
    shunt_sys = _make_multiperiod_shunt_system()
    @test PowerFlowData(
        ACPolarPowerFlow(; control_discrete_devices = true, time_steps = 2),
        shunt_sys,
    ) isa PowerFlowData
end

@testset "discrete control: FACTS enrolls without a flag; taps/shunts unaffected" begin
    sys = _make_svc_system(; max_shunt_current = 100.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @test length(data.controlled_devices.facts) == 1
end

@testset "discrete control: arc-admittance rows synced to final parameters" begin
    # Reported branch flows are computed from arc_admittance_from_to/to_from AFTER the
    # time-step loop; the post-continuation sync must bring the moved devices' rows in
    # line with their final tap, exactly matching a fresh rebuild at that tap.
    sys = _make_tap_shunt_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    set = PowerFlows.build_controlled_device_set(
        sys, PF.get_bus_lookup(data), data.power_network_matrix)
    d = set.taps[1]
    newtap = 1.05
    PowerFlows.apply_parameter!(d, data, newtap, 1)
    PowerFlows._sync_arc_admittances!(data, set)
    # Rebuild reference at the new tap.
    txs = collect(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_tap!(txs[1], newtap)
    data2 = PowerFlowData(ACPolarPowerFlow(), sys)
    @test data.power_network_matrix.arc_admittance_from_to.data ≈
          data2.power_network_matrix.arc_admittance_from_to.data
    @test data.power_network_matrix.arc_admittance_to_from.data ≈
          data2.power_network_matrix.arc_admittance_to_from.data
    # Idempotency: a second sync with no parameter change is a no-op.
    PowerFlows._sync_arc_admittances!(data, set)
    @test data.power_network_matrix.arc_admittance_from_to.data ≈
          data2.power_network_matrix.arc_admittance_from_to.data
end

@testset "discrete control: reported flows match a rebuild at the snapped tap" begin
    # End-to-end parity: solve with control, then rebuild the system at the solver's
    # final tap and re-solve WITHOUT control — voltages AND reported branch flows must
    # agree, proving the flow computation saw the same network as the inner solver.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    t = data.controlled_devices.taps[1]
    sh = data.controlled_devices.shunts[1]
    # Rebuild: same system with the tap fixed at the solved position and the shunt's
    # solved susceptance as a fixed admittance baseline.
    txs = collect(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_tap!(txs[1], t.current)
    # data_ref is built WITHOUT control_discrete_devices, so the VOLTAGE objective is
    # inert: this is a plain solve of the snapped network.
    sas = collect(PSY.get_components(PSY.SwitchedAdmittance, sys))
    PSY.set_Y!(sas[1], PSY.get_Y(sas[1]) + im * (sh.current - sh.initial))
    data_ref = PowerFlowData(ACPolarPowerFlow(), sys)
    solve_power_flow!(data_ref)
    @test all(data_ref.converged)
    @test all(isapprox.(data.bus_magnitude, data_ref.bus_magnitude; atol = 1e-6))
    @test all(
        isapprox.(
            data.arc_active_power_flow_from_to,
            data_ref.arc_active_power_flow_from_to;
            atol = 1e-5,
        ),
    )
    @test all(
        isapprox.(
            data.arc_reactive_power_flow_from_to,
            data_ref.arc_reactive_power_flow_from_to;
            atol = 1e-5,
        ),
    )
end

@testset "discrete control: solved device settings surface" begin
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    df = PowerFlows.get_controlled_device_results(data)
    @test size(df, 1) == 2
    tap_row = df[df.family .== "TapTransformer", :]
    @test tap_row.name == ["tap_1_2"]
    @test tap_row.initial == [1.0]
    t = data.controlled_devices.taps[1]
    @test tap_row.final == [t.current]
    @test tap_row.lower_limit == [t.p_min]
    @test tap_row.upper_limit == [t.p_max]
    # Disabled path: empty frame with the same schema.
    data_off = PowerFlowData(ACPolarPowerFlow(), _make_tap_shunt_system())
    df_off = PowerFlows.get_controlled_device_results(data_off)
    @test size(df_off, 1) == 0
    @test names(df_off) ==
          ["family", "name", "time_step", "lower_limit", "upper_limit", "initial",
        "final", "delivered_q_mvar", "saturated"]
end

@testset "discrete control: shunt deadband semantics" begin
    # In-band voltages hold the device (PSS/E VSWLO/VSWHI semantics); out-of-band do not.
    sh = PowerFlows.ControlledSwitchedShunt("s", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [4], [0.05], 0.0, 0.2, zeros(Int, 1), false, 0.0, 0.0, false)
    @test PowerFlows._in_deadband(sh, 1.0)
    @test PowerFlows._in_deadband(sh, 0.96)
    @test !PowerFlows._in_deadband(sh, 0.94)
    @test !PowerFlows._in_deadband(sh, 1.06)
    # Taps carry a point setpoint: never in a deadband.
    tap = PowerFlows.ControlledTap("t", 1, 2, 2, 1.0, 1.0 + 0im,
        0.0, 0.9, 1.1, collect(range(0.9, 1.1; length = 5)),
        (1, 2, 3, 4), 1.0, 1.0, 1.0)
    @test !PowerFlows._in_deadband(tap, 1.0)
    # Scale-aware settle tolerance: wide-range devices get a relative floor.
    @test PowerFlows._param_tol(tap) ≈
          max(PowerFlows.CONTROL_PARAM_TOL, PowerFlows.CONTROL_PARAM_RTOL * 0.2)
    wide = PowerFlows.ControlledSwitchedShunt("w", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [10], [1.0], 0.0, 10.0, zeros(Int, 1), false, 0.0, 0.0, false)
    @test PowerFlows._param_tol(wide) ≈ PowerFlows.CONTROL_PARAM_RTOL * 10.0
end

@testset "discrete control: inner-solve budget regression" begin
    # Iteration counts (not wall-clock) are the robust performance metric. With
    # full-step-first continuation, single-solve probes, and per-stage budgets, the
    # 2-device solvable fixture regulates in well under 150 inner solves; the pre-fix
    # engine needed >500 (16-sub-step walks per move, 2 solves per probe). Budget has
    # ~2x headroom over the measured count to absorb solver-version noise.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    n = PowerFlows.get_control_inner_solve_count(data)
    @test 0 < n < 300
    # Disabled path reports zero.
    data_off = PowerFlowData(ACPolarPowerFlow(), sys)
    solve_power_flow!(data_off)
    @test PowerFlows.get_control_inner_solve_count(data_off) == 0
end

@testset "discrete control: device-settings write-back" begin
    # A controlled solve moves devices; solve_and_store writes the solved settings back
    # into the system so the stored branch flows stay self-consistent with the device
    # state (write_power_flow_solution! recomputes flows from the system components and
    # asserts they match the stored moved-device flows). Write-back is therefore automatic
    # when controls are active.
    sys = _make_solvable_tap_shunt_system()
    tx0 = first(PSY.get_components(PSY.TapTransformer, sys))
    tap_before = PSY.get_tap(tx0)
    levels =
        PowerFlowData(
            ACPolarPowerFlow(; control_discrete_devices = true), sys,
        ).controlled_devices.taps[1].levels
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    @test solve_and_store_power_flow!(pf, sys)
    @test PSY.get_tap(tx0) != tap_before   # the fixture regulates away from tap = 1.0
    @test PSY.get_tap(tx0) in levels       # the written tap is a valid discrete level
end

@testset "write-back round-trips the API shunt convention" begin
    sys = _make_tap_shunt_system()
    sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
    # A nonzero initial_status marks the API convention (Y is the fixed base,
    # initial_status meaningful), as opposed to the parser's zeroed-status BINIT
    # convention.
    # One block already switched on at enrollment; the fixture's wide deadband holds the
    # shunt here (it never re-snaps), so this also exercises the never-snapped realizability
    # guard: block_n stays [0] while d.current reflects the pre-activated block.
    PSY.set_initial_status!(sa, [1])
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    ts = 1
    @test PowerFlows.solve_power_flow!(data)
    PowerFlows.write_device_settings!(sys, data)
    solved =
        PowerFlows.current_parameter(first(PowerFlows.get_controlled_devices(data).shunts))
    # Re-enrolling from the written-back system must reproduce the solved susceptance,
    # not double-count it.
    data2 = PowerFlowData(pf, sys)
    d2 = first(PowerFlows.get_controlled_devices(data2).shunts)
    @test PowerFlows.current_parameter(d2) ≈ solved atol = 1e-9
end

@testset "write-back hits the primary API shunt branch on a genuine snap" begin
    # _make_shunt_snap_system's weak bus needs 3 of 4 discrete steps to reach the
    # deadband, so the controlled solve snaps block_n to a nonzero, non-saturated
    # count. Unlike the never-snapped fixture above, this exercises the PRIMARY
    # (realizable) branch of write_device_settings!, not the fallback.
    sys = _make_shunt_snap_system()
    sa0 = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
    @test isempty(PSY.get_initial_status(sa0))   # API convention, not the BINIT marker
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @test PowerFlows.solve_power_flow!(data)
    d = first(PowerFlows.get_controlled_devices(data).shunts)
    @test !iszero(sum(d.block_n))
    PowerFlows.write_device_settings!(sys, data)
    sa = PSY.get_component(PSY.SwitchedAdmittance, sys, d.name)
    @test PSY.get_Y(sa) ≈ Complex(d.g0, d.b0)
    @test PSY.get_initial_status(sa) == d.block_n
    data2 = PowerFlowData(pf, sys)
    d2 = first(PowerFlows.get_controlled_devices(data2).shunts)
    @test PowerFlows.current_parameter(d2) ≈ d.current atol = 1e-9
end

@testset "sub-noise secant samples do not freeze devices" begin
    sys = _make_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    ts = 1
    @test PowerFlows._solve_with_q_limits!(pf, data, ts)
    set = PowerFlows.get_controlled_devices(data)
    d = first(set.taps)
    frozen = [false]
    dVdp = [0.05]
    # A sign-flipped but sub-floor sample must be ignored: gain unchanged, not frozen.
    PowerFlows._maybe_refresh_gain!(
        d,
        1,
        data,
        ts,
        frozen,
        dVdp,
        0.05,
        -5.0e-6 / 2.0e-5,
        5.0e-6,
    )
    @test !frozen[1]
    @test dVdp[1] == 0.05
end

@testset "discrete control: linearized plant sensitivity matches FD probe (P2)" begin
    # The linearized sensitivity dy/dp = (−J⁻¹ ∂F/∂p)[Vm(controlled)] must agree with the
    # finite-difference probe in SIGN and magnitude (the FD probe carries O(δ) truncation, so
    # the linear form is if anything more accurate). A sign error here would silently invert a
    # control's feedback direction — the highest-consequence P2 failure mode.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    PowerFlows._solve_with_q_limits!(pf, data, 1)   # converge base case only
    set = data.controlled_devices
    ctx = PowerFlows._sensitivity_context(pf, data, 1)
    @test !isnothing(ctx)
    scratch_snap = PowerFlows._snapshot_state(data, 1)
    for d in (set.taps[1], set.shunts[1])
        lin, ok_lin = PowerFlows._linear_plant_sign(d, data, 1, ctx)
        fd, ok_fd = PowerFlows._plant_sign(d, data, 1, pf, scratch_snap)
        @test ok_lin && ok_fd
        @test sign(lin) == sign(fd)
        @test isapprox(lin, fd; rtol = 1e-2)
    end
end

@testset "discrete control: P2 keeps symbolic factorization at 2 per continuation" begin
    # P1 (symbolic reuse) + P2 (the sensitivity context reuses that symbolic factor, refactors
    # numerically): the whole continuation performs exactly two counted builds — one for the
    # first `_ctrl_solve!`'s persisted `PolarNRCache` (the real KLU symbolic factorization) and
    # one for the single full `_sensitivity_context` build in the probe phase. Every subsequent
    # batched-pass gain refresh goes through `_refresh_sensitivity_context!` (values-only,
    # reusing both the KLU symbolic factor AND the persisted residual/Jacobian objects), so
    # neither count grows with the number of passes.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    @test PowerFlows.get_control_symbolic_factor_count(data) == 2
    @test PowerFlows.get_control_inner_solve_count(data) > 1
end

@testset "discrete control: batched passes keep inner solves ~flat in device count (P3)" begin
    # P3 does one inner solve per PASS (not per device). On a set of decoupled controlled
    # feeders the inner-solve count must stay ~flat as the device count grows — the sequential
    # path would scale it ~linearly. Build K feeders (REF ─tap─ PQ-load, REF ─line─ PQ-shunt).
    function _feeders(K)
        sys = System(100.0)
        ref = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
        _add_simple_source!(sys, ref, 0.0, 0.0)
        for k in 1:K
            bl = _add_simple_bus!(sys, 2k, ACBusTypes.PQ, 230, 1.0, 0.0)
            bs = _add_simple_bus!(sys, 2k + 1, ACBusTypes.PQ, 230, 1.0, 0.0)
            add_component!(
                sys,
                PowerLoad(; name = "l$k", available = true, bus = bl,
                    active_power = 0.5, reactive_power = 0.25, base_power = 100.0,
                    max_active_power = 100.0, max_reactive_power = 100.0),
            )
            add_component!(
                sys,
                PowerLoad(; name = "s$k", available = true, bus = bs,
                    active_power = 0.05, reactive_power = 0.025, base_power = 100.0,
                    max_active_power = 100.0, max_reactive_power = 100.0),
            )
            _add_simple_line!(sys, ref, bs, 1e-2, 1e-2, 0.0)
            add_component!(
                sys,
                TapTransformer(; name = "t$k", available = true,
                    active_power_flow = 0.0, reactive_power_flow = 0.0,
                    arc = Arc(; from = ref, to = bl), r = 0.01, x = 0.10,
                    primary_shunt = 0.0 + 0.0im, tap = 1.0, rating = 1.0,
                    base_power = 100.0,
                    control_objective = PSY.TransformerControlObjective.VOLTAGE),
            )
            add_component!(
                sys,
                SwitchedAdmittance(; name = "sh$k", available = true,
                    bus = bs, Y = 0.0 + 0.0im, initial_status = [0], number_of_steps = [4],
                    Y_increase = [0.0 + 0.05im], admittance_limits = (min = 0.9, max = 1.1),
                ),
            )
        end
        return sys
    end
    function _inner(K)
        pf = ACPolarPowerFlow(; control_discrete_devices = true)
        data = PowerFlowData(pf, _feeders(K))
        solve_power_flow!(data)
        @test all(data.converged)
        return PowerFlows.get_control_inner_solve_count(data)
    end
    n1, n8 = _inner(1), _inner(8)   # 2 vs 16 devices
    # Batched: n8 ≈ n1 (one solve/pass, flat). Sequential would give n8 ≳ 8·n1. Allow generous
    # headroom for the extra per-pass work while still failing loudly if batching regresses.
    @test n8 < 2 * n1 + 20
end

@testset "linear plant sign matches FD probe at PV controlled bus" begin
    sys = _make_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    ts = 1
    # Converge the base state, then build the sensitivity context the probes use.
    @test PowerFlows._solve_with_q_limits!(pf, data, ts)
    ctx = PowerFlows._sensitivity_context(pf, data, ts)
    @test !isnothing(ctx)
    set = PowerFlows.get_controlled_devices(data)
    for d in set.shunts
        cbus = PowerFlows.controlled_bus_ix(d)
        # Force the controlled bus PV (the must_be_PV production case).
        original_bt = data.bus_type[cbus, ts]
        data.bus_type[cbus, ts] = PSY.ACBusTypes.PV
        s_lin, ok = PowerFlows._linear_plant_sign(d, data, ts, ctx)
        data.bus_type[cbus, ts] = original_bt
        @test ok
        # Voltage at a PV bus is pinned: the true sensitivity is exactly zero.
        @test iszero(s_lin)
    end
end

@testset "_restore_one! early-return restores state on a failed solve" begin
    sys = _make_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    ts = 1
    @test PowerFlows._solve_with_q_limits!(pf, data, ts)
    set = PowerFlows.get_controlled_devices(data)
    d = first(set.taps)
    pre_v = copy(data.bus_magnitude[:, ts])
    pre_q = copy(data.bus_reactive_power_injections[:, ts])
    # `_run_power_flow_method`'s loop guard is `i < maxIterations` with i starting at 1,
    # so maxIterations=1 (or 0) takes ZERO real steps — maxIterations=2 is the smallest
    # budget that runs exactly one real NR step (mutating bus_magnitude away from the
    # poisoned start) while still failing to converge from such a bad warm start.
    data.bus_magnitude[:, ts] .= 0.05
    snapshot_v = copy(data.bus_magnitude[:, ts])
    # The forced non-convergence emits an @error at finalization; capture it with @test_logs so
    # it does not trip run_tests()'s zero-Logging.Error-events assertion (full suite).
    scratch_snap = PowerFlows._snapshot_state(data, ts)
    ok =
        @test_logs (:error, r"failed to converge") match_mode = :any PowerFlows._restore_one!(
            d, data, ts, PowerFlows.current_parameter(d), pf, scratch_snap;
            maxIterations = 2)
    @test !ok
    # On failure the pre-call state must be untouched (no diverged iterate left).
    @test data.bus_magnitude[:, ts] == snapshot_v
    # Restore sanity for later testsets.
    data.bus_magnitude[:, ts] .= pre_v
    data.bus_reactive_power_injections[:, ts] .= pre_q
end

@testset "snap holds never-moved off-grid devices" begin
    sys = _make_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    ts = 1
    @test PowerFlows._solve_with_q_limits!(pf, data, ts)
    set = PowerFlows.get_controlled_devices(data)
    d = first(set.shunts)
    # Simulate a deadband-held device whose baseline is OFF the prefix chain: current
    # equals initial (never moved) and sits between two chain points.
    off_chain = d.initial + 0.4 * minimum(abs, d.block_dB)
    d.initial = off_chain
    PowerFlows.apply_parameter!(d, data, off_chain, ts)
    PowerFlows._snap_device_group!([d], data, ts)
    @test PowerFlows.current_parameter(d) == off_chain   # held, not snapped
end

@testset "probe restores VSC DC-network state" begin
    # A VSC system passes the construction guard (only LCC is rejected), so it can enter the
    # continuation with a controlled tap. `_read_vsc_state!` writes the solver iterate into
    # dcn.p_c/q_c/node_vdc on every residual evaluation; a probe that fails to restore them
    # would leak a diverged DC state into the next warm start.
    #
    # The "to" converter must NOT pin P via DC_POWER for this to be observable: with a fixed P
    # setpoint, p_c/node_vdc are pure DC-side algebra (P_from + P_to = losses(Vdc), Vdc[slack]
    # fixed) and are exactly invariant to any AC-only perturbation regardless of restore
    # correctness. Giving the "to" converter AC-voltage control (ControlPVac: P pinned, Q free)
    # makes q_c a genuine AC-network-coupled unknown — moving under a nearby tap probe and
    # needing a real restore — while p_c/node_vdc stay checked too (they should never move).
    sys = _build_vsc_pq_system(;
        ac_control_to = PSY.VSCACControlModes.AC_VOLTAGE,
        ac_setpoint_to = 1.0,
    )
    pq = sort!(
        collect(
            PSY.get_components(
                b -> PSY.get_bustype(b) == PSY.ACBusTypes.PQ,
                PSY.ACBus,
                sys,
            ),
        );
        by = PSY.get_number,
    )
    _add_control_tap!(sys, pq[2], pq[3])   # incident to the AC-voltage-controlled VSC bus
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        control_discrete_devices = true,
        solver_settings = VSC_SETTINGS,
    )
    data = PowerFlowData(pf, sys)
    ts = 1
    @test PowerFlows._solve_with_q_limits!(pf, data, ts)
    dcn = PowerFlows.get_dc_network(data)
    @test PowerFlows.has_dc_network(dcn)
    pre_p = copy(dcn.p_c[:, ts])
    pre_q = copy(dcn.q_c[:, ts])
    pre_v = copy(dcn.node_vdc[:, ts])
    set = PowerFlows.get_controlled_devices(data)
    @test !isempty(set.taps)
    d = first(set.taps)
    scratch_snap = PowerFlows._snapshot_state(data, ts)
    PowerFlows._plant_sign(d, data, ts, pf, scratch_snap)   # probe solves, then must restore
    @test dcn.p_c[:, ts] == pre_p
    @test dcn.node_vdc[:, ts] == pre_v
    @test dcn.q_c[:, ts] == pre_q
end

@testset "continuation reuses symbolic factorization and bounded inner solves" begin
    # Task 13(b): the sensitivity context (residual + Jacobian + numeric refactor) must be
    # built ONCE per continuation and REFRESHED (values only) thereafter, not rebuilt from
    # scratch on every converged batched pass. `get_control_symbolic_factor_count` is the
    # regression metric: a full `_sensitivity_context` build counts, a values-only
    # `_refresh_sensitivity_context!` does not.
    sys = _make_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @test PowerFlows.solve_power_flow!(data)
    @test PowerFlows.get_control_symbolic_factor_count(data) <= 2
    n_inner = PowerFlows.get_control_inner_solve_count(data)
    @test 0 < n_inner < 200
end

@testset "STATCOM |V|-dependent limit: Q_solved ≈ V·Imax at saturation" begin
    # Deep-sag case (+60% load, bus-9 shunt OOS — the calibrated 14-bus stress anchor) with an
    # undersized current limit: this system's PV buses give bus 14 enough native support that
    # a ~6.5 Mvar-capable device (shmx_mva=15) already reaches vset unsaturated (verified by a
    # direct sweep of shmx_mva at this stress point) — shmx_mva=5 is comfortably below that
    # natural need, so the STATCOM saturates at its |V|-dependent current bound before reaching
    # vset. `_facts_b_limit`'s STATCOM branch is current-limited: |Q| <= V·Imax ⇒ b <= Imax/V,
    # so at saturation Q = b_lim·V² = (Imax/V)·V² = V·Imax exactly (mva_cap=9999, non-binding).
    sys = build_ieee14_facts_system(;
        shmx_mva = 5.0, mva_cap = 9999.0, vset = 1.0, stress = 1.6, shunt9_off = true)
    pf = ACPowerFlow(; control_discrete_devices = true)
    data = PowerFlows.PowerFlowData(pf, sys)
    @test PowerFlows.solve_power_flow!(data)
    @test all(data.converged)
    res = PowerFlows.get_controlled_device_results(data)
    row = only(eachrow(res[res.family .== "FACTSControlDevice", :]))
    bl = PowerFlows.get_bus_lookup(data)
    v14 = data.bus_magnitude[bl[14], 1]
    base_mva = PSY.get_base_power(sys)
    q_mvar = row.final * v14^2 * base_mva
    f = only(data.controlled_devices.facts)
    @test isapprox(row.final, f.b_lim; atol = 1e-3)   # riding the |V|-dependent limit
    @test data.bus_magnitude[f.controlled_ix, 1] < f.vset - 1e-3   # setpoint unreached
    @test isapprox(q_mvar, v14 * 5.0; atol = 0.5)   # Q ≈ V·Imax at the current limit
end

@testset "STATCOM riding its limit converges without oscillation chatter" begin
    # Regression for the chatter-freeze guard: refreshing `b_lim` from the measured voltage
    # every pass must not itself induce direction reversals in the damped target (a "b_lim
    # chases V, V chases b_lim" feedback loop) — the device should settle cleanly against its
    # limit, never trip `CONTROL_OSCILLATION_LIMIT`, and stay well under the per-stage pass
    # budget (`MAX_CONTROL_PASSES_PER_STAGE` per steepness stage). Same saturating fixture as
    # the test above.
    sys = build_ieee14_facts_system(;
        shmx_mva = 5.0, mva_cap = 9999.0, vset = 1.0, stress = 1.6, shunt9_off = true)
    pf = ACPowerFlow(; control_discrete_devices = true)
    data = PowerFlows.PowerFlowData(pf, sys)
    tl = Test.TestLogger(; min_level = Logging.Warn)
    converged = Logging.with_logger(tl) do
        PowerFlows.solve_power_flow!(data)
    end
    @test converged
    @test all(data.converged)
    @test !any(occursin("oscillat", r.message) for r in tl.logs)
    n_inner = PowerFlows.get_control_inner_solve_count(data)
    @test 0 < n_inner < 300
end

# ── PSS/E voltage band [VMI1, VMA1] for voltage-controlling taps ────────────────────────────
# `_make_solvable_tap_shunt_system` puts bus 2 at |V| ≈ 0.97 with the tap at 1.0, and the tap
# has full authority over it (lowering the tap raises |V₂|). `remote` adds a load bus 4 behind
# bus 2 and regulates it instead (regulated_bus_number = 4).
function _banded_tap_system(vmi, vma; remote::Bool = false, tap = 1.0)
    sys = _make_solvable_tap_shunt_system()
    tx = only(get_components(TapTransformer, sys))
    if remote
        b2 = get_bus(sys, 2)
        b4 = _add_simple_bus!(sys, 4, ACBusTypes.PQ, 230, 1.0, 0.0)
        _add_simple_line!(sys, b2, b4, 1e-2, 5e-2, 0.0)
        _add_simple_load!(sys, b4, 0.2, 0.1)
        set_regulated_bus_number!(tx, 4)
    end
    # What the PSS/E parser stores: the band in `ext`, its midpoint as the setpoint.
    get_ext(tx)["VMI1"] = vmi
    get_ext(tx)["VMA1"] = vma
    set_voltage_setpoint!(tx, (vmi + vma) / 2)
    set_tap!(tx, tap)
    return sys, tx
end

function _solve_banded(vmi, vma; remote::Bool = false, tap_voltage_band::Bool = true)
    sys, _ = _banded_tap_system(vmi, vma; remote = remote)
    pf = ACPolarPowerFlow(;
        control_discrete_devices = true, tap_voltage_band = tap_voltage_band)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    d = only(PF.get_controlled_devices(data).taps)
    return d, data.bus_magnitude[d.controlled_ix, 1]
end

# |V| at the regulated bus with the tap fixed at `tap` (no control).
function _v_at_fixed_tap(vmi, vma, tap; remote::Bool = false)
    sys, _ = _banded_tap_system(vmi, vma; remote = remote, tap = tap)
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    @test solve_power_flow!(data)
    reg = remote ? 4 : 2
    return data.bus_magnitude[PF.get_bus_lookup(data)[reg], 1]
end

@testset "tap band: enrollment reads [VMI1, VMA1] only while the setpoint is its midpoint" begin
    sys, tx = _banded_tap_system(0.95, 1.05)
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    bl = PF.get_bus_lookup(data)
    d = only(PowerFlows.build_controlled_device_set(sys, bl, data.power_network_matrix).taps)
    @test d.band
    @test (d.v_lo, d.v_hi) == (0.95, 1.05)
    @test d.vset ≈ 1.0
    # Option off: the midpoint point setpoint.
    d_mid = only(
        PowerFlows.build_controlled_device_set(sys, bl, data.power_network_matrix;
            tap_voltage_band = false).taps,
    )
    @test !d_mid.band
    # A setpoint moved off the midpoint after parsing wins over the stale ext band.
    set_voltage_setpoint!(tx, 1.03)
    d_moved = only(
        PowerFlows.build_controlled_device_set(sys, bl, data.power_network_matrix).taps)
    @test !d_moved.band
    @test d_moved.vset ≈ 1.03
    # API-built taps (no ext band) keep their point setpoint.
    sys_api = _make_solvable_tap_shunt_system()
    data_api = PowerFlowData(ACPolarPowerFlow(), sys_api)
    @test !only(
        PowerFlows.build_controlled_device_set(sys_api, PF.get_bus_lookup(data_api),
            data_api.power_network_matrix).taps,
    ).band
    # The pf option defaults to the band.
    @test ACPolarPowerFlow().tap_voltage_band
    @test ACRectangularPowerFlow().tap_voltage_band
    @test ACMixedPowerFlow().tap_voltage_band
    @test !ACPolarPowerFlow(; tap_voltage_band = false).tap_voltage_band
end

@testset "tap band: latch, deadband and directional snap (unit)" begin
    levels = collect(range(0.9, 1.1; length = 33))
    d = PowerFlows.ControlledTap("t", 1, 2, 2, 1.0, 1.0 / (0.01 + 0.1im), 0.0, 0.9, 1.1,
        levels, (1, 1, 1, 1), 1.0, 1.0, 1.0, nothing, true, 0.97, 1.03, 0, 1.0)
    @test PowerFlows._in_deadband(d, 1.0)
    PowerFlows._update_band_state!(d, 1.0)
    @test d.side == 0
    PowerFlows._update_band_state!(d, 0.96)              # below VMI1: latch, regulate to it
    @test d.side == -1
    @test d.anchor == 1.0
    @test PowerFlows.control_setpoint(d) == 0.97
    @test !PowerFlows._in_deadband(d, 0.98)              # latched: inside does not hold
    # The tap moved down (dV/dp < 0): it may not retreat above its anchor.
    @test PowerFlows._band_clamp(d, 1.02, -1.0) == 1.0
    @test PowerFlows._band_clamp(d, 0.98, -1.0) == 0.98
    # Snap goes on in the direction moved: 0.9810 lies between levels 0.98125 and 0.975.
    d.current = 0.981
    @test PowerFlows.snap_to_discrete(d, 0.981) ≈ 0.975
    @test PowerFlows.snap_to_discrete(d, 0.85) == 0.9     # tap limit holds
    d.current = 0.975
    @test PowerFlows._next_level(d, -1) ≈ 0.96875
    d.current = 0.9
    @test isnothing(PowerFlows._next_level(d, -1))
    # Back at its anchor with the voltage inside: released and held.
    d.current = 1.0
    PowerFlows._update_band_state!(d, 0.99)
    @test d.side == 0
    @test PowerFlows._in_deadband(d, 0.99)
    PowerFlows._update_band_state!(d, 1.04)               # above VMA1
    @test d.side == 1
    @test PowerFlows.control_setpoint(d) == 1.03
    # A point-setpoint tap snaps to the nearest level and has no deadband.
    p = PowerFlows.ControlledTap("p", 1, 2, 2, 1.0, 1.0 + 0im, 0.0, 0.9, 1.1, levels,
        (1, 1, 1, 1), 1.0, 1.0, 1.0)
    @test !p.band
    @test !PowerFlows._in_deadband(p, 1.0)
    @test PowerFlows.snap_to_discrete(p, 0.981) ≈ 0.98125
end

@testset "tap band: a tap whose voltage is inside its band does not move" begin
    v0 = _v_at_fixed_tap(0.95, 1.05, 1.0)
    @test 0.95 < v0 < 1.0                       # inside the band, below its midpoint
    d, v = _solve_banded(0.95, 1.05)
    @test d.current == d.initial == 1.0
    @test v ≈ v0 atol = 1e-8
    # The midpoint option still drives it to 1.0.
    d_mid, v_mid = _solve_banded(0.95, 1.05; tap_voltage_band = false)
    @test d_mid.current < 1.0
    @test abs(v_mid - 1.0) < 0.01
end

@testset "tap band: below the band, steps toward it and stops at the first position inside" begin
    vmi, vma = 0.99, 1.03
    @test _v_at_fixed_tap(vmi, vma, 1.0) < vmi
    d, v = _solve_banded(vmi, vma)
    p = d.current
    @test p in d.levels
    @test p < d.initial                         # lowering the tap raises |V₂|
    @test vmi - PowerFlows.CONTROL_TAP_BAND_TOL <= v <= vma
    # The first step inside: one position back toward the initial tap is still below VMI1.
    step = d.levels[2] - d.levels[1]
    @test _v_at_fixed_tap(vmi, vma, p + step) < vmi
    @test v ≈ _v_at_fixed_tap(vmi, vma, p) atol = 1e-8
    # The midpoint option drives further, to |V| ≈ 1.01.
    d_mid, v_mid = _solve_banded(vmi, vma; tap_voltage_band = false)
    @test d_mid.current < p
    @test abs(v_mid - 1.01) < 0.01
end

@testset "tap band: above the band, steps toward it and stops at the first position inside" begin
    vmi, vma = 0.90, 0.95
    @test _v_at_fixed_tap(vmi, vma, 1.0) > vma
    d, v = _solve_banded(vmi, vma)
    p = d.current
    @test p in d.levels
    @test p > d.initial
    @test vmi <= v <= vma + PowerFlows.CONTROL_TAP_BAND_TOL
    step = d.levels[2] - d.levels[1]
    @test _v_at_fixed_tap(vmi, vma, p - step) > vma
end

@testset "tap band: an unreachable band leaves the tap at its limit" begin
    vmi, vma = 1.12, 1.14
    @test _v_at_fixed_tap(vmi, vma, 0.9) < vmi    # even the lowest tap cannot reach it
    d, v = _solve_banded(vmi, vma)
    @test d.current == d.p_min == 0.9
    @test v < vmi
    vmi, vma = 0.80, 0.82
    @test _v_at_fixed_tap(vmi, vma, 1.1) > vma
    d, v = _solve_banded(vmi, vma)
    @test d.current == d.p_max == 1.1
end

@testset "tap band: a remote regulated bus is held to the band the same way" begin
    vmi, vma = 0.97, 1.03
    @test _v_at_fixed_tap(vmi, vma, 1.0; remote = true) < vmi
    d, v = _solve_banded(vmi, vma; remote = true)
    @test d.controlled_ix != d.to_ix
    p = d.current
    @test p < d.initial
    @test vmi - PowerFlows.CONTROL_TAP_BAND_TOL <= v <= vma
    step = d.levels[2] - d.levels[1]
    @test _v_at_fixed_tap(vmi, vma, p + step; remote = true) < vmi
    # Inside the band at the initial tap: held.
    vmi, vma = 0.80, 1.20
    d, _ = _solve_banded(vmi, vma; remote = true)
    @test d.current == d.initial
end

# Several taps whose regulated buses already sit inside [0.92, 1.08] (|V| ≈ 0.96–0.98 at tap
# 1.0): with the band nothing has to move and the continuation settles in its first pass of
# every stage; with the midpoint rule every tap chases 1.0 and the passes do real work.
function _multi_banded_system(n; vmi = 0.92, vma = 1.08)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    for k in 1:n
        bk = _add_simple_bus!(sys, 10 + k, ACBusTypes.PQ, 230, 1.0, 0.0)
        _add_simple_load!(sys, bk, 40.0, 20.0)   # MW, MVAr
        tx = TapTransformer(; name = "t$k", available = true,
            active_power_flow = 0.0, reactive_power_flow = 0.0,
            arc = Arc(; from = b1, to = bk), r = 0.005, x = 0.06 + 0.02k,
            primary_shunt = 0.0 + 0.0im, tap = 1.0, rating = 1.0, base_power = 100.0,
            control_objective = PSY.TransformerControlObjective.VOLTAGE,
            voltage_setpoint = (vmi + vma) / 2,
            ext = Dict{String, Any}("VMI1" => vmi, "VMA1" => vma))
        add_component!(sys, tx)
        # Neighbouring feeders are tied, so the taps interact.
        k > 1 && _add_simple_line!(sys, get_bus(sys, 10 + k - 1), bk, 0.01, 0.2, 0.0)
    end
    return sys
end

@testset "tap band: a multi-tap system inside its bands settles at once; the midpoint rule moves" begin
    n = 6
    function run(band)
        pf = ACPolarPowerFlow(; control_discrete_devices = true, tap_voltage_band = band)
        data = PowerFlowData(pf, _multi_banded_system(n))
        @test solve_power_flow!(data)
        set = PF.get_controlled_devices(data)
        v = [data.bus_magnitude[d.controlled_ix, 1] for d in set.taps]
        return data, set, v
    end
    data_b, set_b, v_b = run(true)
    @test all(0.92 .<= v_b .<= 0.99)                 # inside the band, below its midpoint
    @test all(d.current == d.initial for d in set_b.taps)
    inner_band = PF.get_control_inner_solve_count(data_b)
    data_m, set_m, v_m = run(false)
    @test count(d -> d.current != d.initial, set_m.taps) == n
    @test all(abs.(v_m .- 1.0) .< 0.01)
    inner_mid = PF.get_control_inner_solve_count(data_m)
    # Band: the base solve and the snap's solve, no pass moves anything (2 inner solves here);
    # the midpoint rule ran 45.
    @test inner_band <= 3
    @test inner_mid > 10 * inner_band
end

# REF(1) ─tap B (1→2)─ 2 ─line─ 3 ─tap A (3→4)─ 4, loads at 2 and 4. A regulates bus 4 and sits
# inside its band [VMI1, VMA1] at its parsed ratio; B regulates bus 2, below its band. B's moves
# carry |V₄| out of A's band and back during the continuation, so A latches, steps, and retreats
# to within `_param_tol` of its anchor (1.99e-5 off its ratio here before GAP-057's fix).
function _in_band_drift_system(; tA = 1.0055)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    b4 = _add_simple_bus!(sys, 4, ACBusTypes.PQ, 115, 1.0, 0.0)
    _add_simple_line!(sys, b2, b3, 0.01, 0.2146, 0.0)
    _add_simple_load!(sys, b2, 12.4407, 31.5854)
    _add_simple_load!(sys, b4, 44.2138, 9.1698)
    for (nm, f, t, tap, lo, hi, x, reg) in (
        ("A", b3, b4, tA, 0.9751, 1.0081, 0.0531, 4),
        ("B", b1, b2, 0.919, 0.9994, 1.0194, 0.0858, 2))
        add_component!(sys, TapTransformer(; name = nm, available = true,
            active_power_flow = 0.0, reactive_power_flow = 0.0,
            arc = Arc(; from = f, to = t), r = 0.002, x = x, primary_shunt = 0.0im,
            tap = tap, rating = 1.0, base_power = 100.0,
            control_objective = PSY.TransformerControlObjective.VOLTAGE,
            voltage_setpoint = (lo + hi) / 2, regulated_bus_number = reg,
            ext = Dict{String, Any}("VMI1" => lo, "VMA1" => hi)))
    end
    return sys
end

@testset "tap band: an in-band tap ends bit-equal to its parsed ratio (GAP-057)" begin
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, _in_band_drift_system())
    @test solve_power_flow!(data)
    taps = Dict(d.name => d for d in PF.get_controlled_devices(data).taps)
    A, B = taps["A"], taps["B"]
    @test B.current != B.initial                       # B regulated
    @test A.current === A.initial === 1.0055           # A held exactly at its parsed ratio
    vA = data.bus_magnitude[A.controlled_ix, 1]
    @test A.v_lo - PowerFlows.CONTROL_TAP_BAND_TOL <= vA <= A.v_hi
    # The solved network is the one with A at its parsed ratio and B where it ended.
    sys_fixed = _in_band_drift_system()
    set_tap!(get_component(TapTransformer, sys_fixed, "B"), B.current)
    fixed = PowerFlowData(ACPolarPowerFlow(), sys_fixed)
    @test solve_power_flow!(fixed)
    @test data.bus_magnitude[:, 1] ≈ fixed.bus_magnitude[:, 1] atol = 1e-8
    # Unit: a sub-tolerance drift is undone by the snap, and the Y-bus block goes back.
    data2 = PowerFlowData(pf, _in_band_drift_system())
    set2 = PF.get_controlled_devices(data2)
    d = only(filter(t -> t.name == "A", set2.taps))
    A0 = copy(data2.power_network_matrix.data.nzval[collect(d.nz_offsets)])
    PowerFlows.apply_parameter!(d, data2, d.initial + 0.5 * PowerFlows._param_tol(d), 1)
    cont = PowerFlows._snap_device_group!(set2.taps, data2, 1)
    @test d.current === d.initial
    @test cont[findfirst(==(d), set2.taps)] != d.initial    # the restore path keeps the drift
    @test data2.power_network_matrix.data.nzval[collect(d.nz_offsets)] ≈ A0 rtol = 1e-14
end

# ── PSS/E voltage band [VSWLO, VSWHI] for switched shunts ───────────────────────────────────
# REF(1) ─ line(0.01 + j0.2) ─ PQ(3) with a 30 MW / 100q MVAr load and a switched shunt at bus 3
# (|V₃| ≈ 0.930 with no shunt at q = 0.3; each 0.05 p.u. of capacitance adds ≈ 0.01 p.u.). The
# shunt is built as the PSS/E parser builds it: `Y` = BINIT, `initial_status` zeroed, `ext`
# carrying MODSW and ADJM. `remote` adds a load bus 4 behind bus 3 and regulates it instead.
function _banded_shunt_system(vlo, vhi; steps = [4, 2], dB = [0.05, 0.1], binit = 0.0,
    q = 0.3, remote::Bool = false, adjm::Int = 0, continuous::Bool = false)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_line!(sys, b1, b3, 0.01, 0.2, 0.0)
    _add_simple_load!(sys, b3, 30.0, 100q)
    reg = 0
    if remote
        b4 = _add_simple_bus!(sys, 4, ACBusTypes.PQ, 230, 1.0, 0.0)
        _add_simple_line!(sys, b3, b4, 0.01, 0.1, 0.0)
        _add_simple_load!(sys, b4, 10.0, 10.0)
        reg = 4
    end
    mode = if continuous
        PSY.SwitchedAdmittanceControlMode.CONTINUOUS_VOLTAGE
    else
        PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE
    end
    add_component!(sys, SwitchedAdmittance(; name = "sh3", available = true, bus = b3,
        Y = binit * im, initial_status = zeros(Int, length(steps)),
        number_of_steps = steps, Y_increase = dB .* im,
        admittance_limits = (min = vlo, max = vhi), control_mode = mode,
        regulated_bus_number = reg,
        ext = Dict{String, Any}("MODSW" => continuous ? 2 : 1, "ADJM" => adjm)))
    return sys
end

function _solve_banded_shunt(vlo, vhi; shunt_voltage_band::Bool = true, kw...)
    sys = _banded_shunt_system(vlo, vhi; kw...)
    pf = ACPolarPowerFlow(;
        control_discrete_devices = true, shunt_voltage_band = shunt_voltage_band)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    d = only(PF.get_controlled_devices(data).shunts)
    return d, data.bus_magnitude[d.controlled_ix, 1], data
end

# |V| at the regulated bus with the shunt fixed at susceptance `b` (no control).
function _v_at_fixed_shunt(vlo, vhi, b; remote::Bool = false, kw...)
    sys = _banded_shunt_system(vlo, vhi; kw..., remote = remote, binit = b)
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    @test solve_power_flow!(data)
    return data.bus_magnitude[PF.get_bus_lookup(data)[remote ? 4 : 3], 1]
end

# The level one step from `d.current` toward `d.initial`.
function _level_back(d)
    k = findfirst(==(d.current), d.levels)
    return d.levels[k + (d.current > d.initial ? -1 : 1)]
end

@testset "shunt band: level tables follow PSS/E's block order (ADJM 0 and 1)" begin
    # ADJM 0: capacitor and reactor chains by sign, each in input order, a reactor block listed
    # after a capacitor block included.
    levels, blocks = PowerFlows._shunt_levels(0.0, [2, 1, 3], [-0.05, 0.05, -0.1], 0)
    @test levels ≈ [-0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05]
    @test blocks[1, :] == [2, 0, 3]
    @test blocks[3, :] == [2, 0, 1]
    @test blocks[5, :] == [1, 0, 0]
    @test blocks[6, :] == [0, 0, 0]
    @test blocks[7, :] == [0, 1, 0]
    # Input order matters for ADJM 0: block 2 (0.03) switches on only after block 1 (0.1).
    levels0, _ = PowerFlows._shunt_levels(0.0, [1, 1], [0.1, 0.03], 0)
    @test levels0 ≈ [0.0, 0.1, 0.13]
    # ADJM 1: every total, the next higher or lower admittance; fewest steps per total.
    levels1, blocks1 = PowerFlows._shunt_levels(0.0, [1, 1], [0.1, 0.03], 1)
    @test levels1 ≈ [0.0, 0.03, 0.1, 0.13]
    @test blocks1[2, :] == [0, 1]
    levels2, blocks2 = PowerFlows._shunt_levels(0.0, [2, 1], [0.05, 0.1], 1)
    @test levels2 ≈ [0.0, 0.05, 0.1, 0.15, 0.2]
    @test blocks2[3, :] == [0, 1]                 # 0.1 by one step, not two
    levelsm, _ = PowerFlows._shunt_levels(0.0, [1, 1], [-0.1, 0.05], 1)
    @test levelsm ≈ [-0.1, -0.05, 0.0, 0.05]
    # Over the combination cap: ADJM 0 with a warning.
    big_steps = fill(9, 8)
    @test_logs (:warn, r"ADJM 0") PowerFlows._shunt_levels(0.0, big_steps, fill(0.01, 8), 1)
end

@testset "shunt band: latch, deadband and directional snap (unit)" begin
    d = PowerFlows.ControlledSwitchedShunt("s", 3, 3, 1.0, 0.98, 1.02, 0.0, 0.0,
        [4, 2], [0.05, 0.1], 0.0, 0.4, zeros(Int, 2), false, 0.0, 0.0, false)
    @test !d.band                                 # the positional form is the midpoint rule
    d.band = true
    @test d.levels ≈ [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    @test PowerFlows._in_deadband(d, 1.0)
    PowerFlows._update_band_state!(d, 0.95)       # below VSWLO: latch, regulate to it
    @test d.side == -1
    @test PowerFlows.control_setpoint(d) == 0.98
    @test !PowerFlows._in_deadband(d, 0.99)       # latched: inside does not hold
    @test PowerFlows._band_clamp(d, -0.1, 0.2) == 0.0   # never back past the anchor
    # Snap goes on upward: 0.21 lies between 0.2 and 0.3.
    @test PowerFlows.snap_to_discrete(d, 0.21) ≈ 0.3
    @test d.block_n == [4, 1]
    @test PowerFlows.snap_to_discrete(d, 0.5) ≈ 0.4
    @test d.block_n == [4, 2]
    d.current = 0.2
    @test PowerFlows._next_level(d, 1) ≈ 0.3
    d.current = 0.3
    PowerFlows._sync_blocks!(d)
    @test d.block_n == [4, 1]
    # The midpoint rule snaps to the nearest level and holds once inside.
    d.band = false
    d.side = 0
    @test PowerFlows.snap_to_discrete(d, 0.21) ≈ 0.2
    @test PowerFlows._in_deadband(d, 0.99)
    @test (@allocated PowerFlows.snap_to_discrete(d, 0.21)) == 0
end

@testset "shunt band: enrollment and options" begin
    sys = _banded_shunt_system(0.98, 1.02; adjm = 1)
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    bl = PF.get_bus_lookup(data)
    d = only(PowerFlows.build_controlled_device_set(sys, bl, data.power_network_matrix).shunts)
    @test d.band
    @test (d.vset_lo, d.vset_hi) == (0.98, 1.02)
    @test d.levels ≈ [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]   # ADJM 1 totals
    d_off = only(
        PowerFlows.build_controlled_device_set(sys, bl, data.power_network_matrix;
            shunt_voltage_band = false).shunts,
    )
    @test !d_off.band
    @test d_off.levels ≈ [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]          # ADJM 0 chain
    # A zero-width band is a setpoint: no band latch.
    sys0 = _banded_shunt_system(0.99, 0.99)
    d0 = only(PowerFlows.build_controlled_device_set(sys0, PF.get_bus_lookup(data),
        PowerFlowData(ACPolarPowerFlow(), sys0).power_network_matrix).shunts)
    @test !d0.band
    @test d0.vset == 0.99
    @test ACPolarPowerFlow().shunt_voltage_band
    @test ACRectangularPowerFlow().shunt_voltage_band
    @test ACMixedPowerFlow().shunt_voltage_band
    @test !ACPolarPowerFlow(; shunt_voltage_band = false).shunt_voltage_band
end

@testset "shunt band: a shunt whose voltage is inside its band does not move" begin
    v0 = _v_at_fixed_shunt(0.92, 0.95, 0.0)
    @test 0.92 < v0 < 0.95
    d, v, _ = _solve_banded_shunt(0.92, 0.95)
    @test d.current == d.initial == 0.0
    @test v ≈ v0 atol = 1e-8
end

@testset "shunt band: below the band, steps up and stops at the first step inside" begin
    vlo, vhi = 0.98, 1.02
    @test _v_at_fixed_shunt(vlo, vhi, 0.0) < vlo
    d, v, _ = _solve_banded_shunt(vlo, vhi)
    @test d.current in d.levels
    @test d.current ≈ 0.3
    @test d.block_n == [4, 1]                     # block 1 full, then one step of block 2
    @test vlo <= v <= vhi
    @test _v_at_fixed_shunt(vlo, vhi, _level_back(d)) < vlo
    @test v ≈ _v_at_fixed_shunt(vlo, vhi, d.current) atol = 1e-8
    # The midpoint rule held it where its voltage crossed in and snapped to the nearest
    # level, here one step short: the bus ends below the band.
    d_mid, v_mid, _ = _solve_banded_shunt(vlo, vhi; shunt_voltage_band = false)
    @test d_mid.current ≈ 0.2
    @test v_mid < vlo
end

@testset "shunt band: above the band, switches reactors in and stops at the first step inside" begin
    vlo, vhi = 0.90, 0.92
    kw = (steps = [3, 2], dB = [-0.05, 0.1])
    @test _v_at_fixed_shunt(vlo, vhi, 0.0; kw...) > vhi
    d, v, _ = _solve_banded_shunt(vlo, vhi; kw...)
    @test d.current ≈ -0.1
    @test d.block_n == [2, 0]
    @test vlo <= v <= vhi
    @test _v_at_fixed_shunt(vlo, vhi, _level_back(d); kw...) > vhi
    d_mid, v_mid, _ = _solve_banded_shunt(vlo, vhi; shunt_voltage_band = false, kw...)
    @test v_mid > vhi                             # midpoint rule: one step short again
end

@testset "shunt band: block order, reactors off before capacitors on; ADJM 1" begin
    # BINIT has one reactor step in; the voltage is below the band, so the reactor comes out
    # first, then the capacitor block's steps go in, in order.
    kw = (steps = [2, 2], dB = [-0.05, 0.1], binit = -0.05)
    vlo, vhi = 0.97, 1.0
    d, v, _ = _solve_banded_shunt(vlo, vhi; kw...)
    @test d.levels ≈ [-0.1, -0.05, 0.0, 0.1, 0.2]
    @test d.current ≈ 0.2
    @test d.block_n == [0, 2]
    @test vlo <= v <= vhi
    @test _v_at_fixed_shunt(vlo, vhi, 0.1; kw...) < vlo
    # Two capacitor blocks, 0.15 then 0.1: ADJM 0 reaches 0.15 before 0.1 + 0.15, ADJM 1 also
    # has 0.1 alone, the first total inside [0.95, 1.0].
    kw1 = (steps = [1, 1], dB = [0.15, 0.1])
    vlo, vhi = 0.95, 1.0
    @test _v_at_fixed_shunt(vlo, vhi, 0.1; kw1...) >= vlo
    d0, v0, _ = _solve_banded_shunt(vlo, vhi; adjm = 0, kw1...)
    @test d0.current ≈ 0.15
    @test d0.block_n == [1, 0]
    d1, v1, _ = _solve_banded_shunt(vlo, vhi; adjm = 1, kw1...)
    @test d1.current ≈ 0.1
    @test d1.block_n == [0, 1]
    @test vlo <= v1 <= vhi
end

@testset "shunt band: a remote regulated bus (SWREG) is held to the band the same way" begin
    vlo, vhi = 0.93, 0.96
    @test _v_at_fixed_shunt(vlo, vhi, 0.0; remote = true) < vlo
    d, v, _ = _solve_banded_shunt(vlo, vhi; remote = true)
    @test d.controlled_ix != d.bus_ix
    @test d.current ≈ 0.2
    @test vlo <= v <= vhi
    @test _v_at_fixed_shunt(vlo, vhi, _level_back(d); remote = true) < vlo
    # Inside at the initial susceptance: held.
    d, _, _ = _solve_banded_shunt(0.85, 0.95; remote = true)
    @test d.current == d.initial
end

# Six feeders off one REF bus, tied in a chain, each with a load and a capacitor bank that
# starts out (|V| ≈ 0.91–0.94, below [0.98, 1.02]).
function _multi_banded_shunt_system(n; vlo = 0.98, vhi = 1.02, levels = nothing)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    prev = b1
    for k in 1:n
        bk = _add_simple_bus!(sys, 10 + k, ACBusTypes.PQ, 230, 1.0, 0.0)
        _add_simple_line!(sys, b1, bk, 0.01, 0.15 + 0.03k, 0.0)
        k > 1 && _add_simple_line!(sys, prev, bk, 0.01, 0.2, 0.0)
        _add_simple_load!(sys, bk, 20.0, 25.0)
        b = isnothing(levels) ? 0.0 : levels[k]
        add_component!(sys, SwitchedAdmittance(; name = "sh$k", available = true, bus = bk,
            Y = b * im, initial_status = [0, 0], number_of_steps = [4, 2],
            Y_increase = [0.05im, 0.1im], admittance_limits = (min = vlo, max = vhi),
            control_mode = PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE,
            ext = Dict{String, Any}("MODSW" => 1, "ADJM" => 0)))
        prev = bk
    end
    return sys
end

@testset "shunt band: several interacting shunts settle inside their bands" begin
    n = 6
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, _multi_banded_shunt_system(n))
    @test solve_power_flow!(data)
    set = PF.get_controlled_devices(data)
    v = [data.bus_magnitude[d.controlled_ix, 1] for d in set.shunts]
    @test all(0.98 .<= v .<= 1.02)
    @test all(d.current in d.levels for d in set.shunts)
    @test PF.get_control_batched_fallback_count(data) == 0
    @test PF.get_control_inner_solve_count(data) <= 80
    # No shunt can step back: with any one of them one level back toward where it started, the
    # others as solved, some bus is below the band (its own, or a neighbour's: the shunts
    # couple, so "first step inside" is minimal over single steps, not per bus).
    solved = [d.current for d in set.shunts]
    for k in 1:n
        lv = copy(solved)
        lv[k] = _level_back(set.shunts[k])
        d_fix = PowerFlowData(ACPolarPowerFlow(), _multi_banded_shunt_system(n; levels = lv))
        @test solve_power_flow!(d_fix)
        bl = PF.get_bus_lookup(d_fix)
        @test any(d_fix.bus_magnitude[bl[10 + j], 1] < 0.98 for j in 1:n)
    end
end

@testset "continuous shunt: a zero-width band ends on its setpoint, or exactly at a limit" begin
    kw = (steps = [4, 2], dB = [-0.05, 0.1], continuous = true)   # range [-0.2, 0.2]
    for vset in (0.92, 0.95)
        d, v, _ = _solve_banded_shunt(vset, vset; kw...)
        @test d.b_min < d.current < d.b_max
        @test abs(v - vset) < 1e-6
    end
    d, v, _ = _solve_banded_shunt(0.99, 0.99; kw...)    # beyond even 0.2 p.u. of capacitance
    @test d.current == d.b_max
    @test v < 0.99
    d, v, _ = _solve_banded_shunt(0.88, 0.88; kw...)    # beyond 0.2 p.u. of reactance
    @test d.current == d.b_min
    @test v > 0.88
    # A banded continuous shunt above its band ends on the violated edge.
    d, v, _ = _solve_banded_shunt(0.90, 0.92; kw...)
    @test abs(v - 0.92) < 1e-6
    # Discrete shunts are not finished: a zero-width discrete one snaps to the nearest step.
    d, v, _ = _solve_banded_shunt(0.95, 0.95; steps = [4, 2], dB = [-0.05, 0.1])
    @test d.current in d.levels
end

# ── FACTS: rail snap, exact finish, warm start ──────────────────────────────────────────────
@testset "FACTS: a device that cannot reach vset ends exactly on its rail, saturated" begin
    # STATCOM at IEEE 14 bus 14, 5 MVA, loads ×1.6, bus-9 shunt off: the setpoint is out of
    # reach. PSS/E holds such a device at |Q| = SHMX·V; it ends there exactly.
    sys = build_ieee14_facts_system(; shmx_mva = 5.0, stress = 1.6, shunt9_off = true)
    data = PowerFlowData(ACPolarPowerFlow(; control_discrete_devices = true), sys)
    tl = Test.TestLogger(; min_level = Logging.Warn)
    ok = Logging.with_logger(tl) do
        solve_power_flow!(data)
    end
    @test ok
    @test !any(occursin("did not settle", r.message) for r in tl.logs)
    f = only(PF.get_controlled_devices(data).facts)
    v = data.bus_magnitude[f.controlled_ix, 1]
    @test f.vset - v > PowerFlows.CONTROL_FACTS_SETPOINT_BAND
    @test f.current == f.b_lim
    @test f.b_lim ≈ PowerFlows._facts_b_limit(f, v) rtol = 1e-9
    @test f.saturated
    # STATCOM law: delivered Q = b·V² = rating·V (here the device regulates its own bus).
    @test f.current * v^2 ≈ f.rating * v rtol = 1e-9
end

@testset "FACTS: a device free in range ends on its setpoint" begin
    sys = build_ieee14_facts_system(; shmx_mva = 25.0)
    data = PowerFlowData(ACPolarPowerFlow(; control_discrete_devices = true), sys)
    @test solve_power_flow!(data)
    f = only(PF.get_controlled_devices(data).facts)
    @test abs(data.bus_magnitude[f.controlled_ix, 1] - f.vset) < 1e-6
    @test abs(f.current) < f.b_lim
    @test !f.saturated
end

@testset "FACTS: warm start from reactive_power_required" begin
    # Unit: b = Q / (base·|V|²) at the device bus's stored |V|, clamped to ±rating.
    sys = _make_svc_system(; max_shunt_current = 100.0)
    fd = only(get_components(FACTSControlDevice, sys))
    set_reactive_power_required!(fd, 42.0)
    set_magnitude!(get_bus(fd), 1.05)
    @test PowerFlows._facts_warm_start(fd, 100.0, 1.0) ≈ 0.42 / 1.05^2
    set_reactive_power_required!(fd, 500.0)
    @test PowerFlows._facts_warm_start(fd, 100.0, 1.0) == 1.0
    set_reactive_power_required!(fd, 0.0)
    @test PowerFlows._facts_warm_start(fd, 100.0, 1.0) == 0.0
    # The store and the withdrawal agree: the warm b is debited from the bus's constant-Z
    # reactive withdrawal, so a solve with control off sees the stored Q.
    sys = _make_svc_system(; max_shunt_current = 100.0)
    set_reactive_power_required!(only(get_components(FACTSControlDevice, sys)), 0.0)
    cold = PowerFlowData(ACPolarPowerFlow(; control_discrete_devices = true), sys)
    @test solve_power_flow!(cold)
    n_cold = PF.get_control_inner_solve_count(cold)
    f_cold = only(PF.get_controlled_devices(cold).facts)
    @test abs(cold.bus_magnitude[f_cold.controlled_ix, 1] - f_cold.vset) < 1e-6
    # Store the solved Q and |V| (as solve_and_store does), rebuild: it starts at the answer.
    PowerFlows.write_device_settings!(sys, cold)
    bus = get_bus(only(get_components(FACTSControlDevice, sys)))
    set_magnitude!(bus, cold.bus_magnitude[f_cold.bus_ix, 1])
    warm = PowerFlowData(ACPolarPowerFlow(; control_discrete_devices = true), sys)
    f_warm = only(PF.get_controlled_devices(warm).facts)
    @test f_warm.initial ≈ f_cold.current rtol = 1e-9
    ix = f_warm.bus_ix
    # Built warm, the withdrawal already carries b, as the cold solve's did once it ended.
    @test warm.bus_reactive_power_constant_impedance_withdrawals[ix, 1] ≈
          cold.bus_reactive_power_constant_impedance_withdrawals[ix, 1]
    @test solve_power_flow!(warm)
    @test abs(warm.bus_magnitude[f_warm.controlled_ix, 1] - f_warm.vset) < 1e-6
    @test f_warm.current ≈ f_cold.current rtol = 1e-6
    @test PF.get_control_inner_solve_count(warm) < n_cold
end

# ── Shared regulated bus (GAP-056) ──────────────────────────────────────────────────────────
# REF(1) ─ G(2) ─ X(3) ─ Y(4). A generator at G holds its own terminal (PV, |V| = 1.0); with
# `ireg` its `ext` names X as the bus it regulates in PSS/E (IREG), which PowerSystems cannot
# represent. Continuous MODSW 2 shunts with a point setpoint `vset` (VSWLO = VSWHI): "X_Z2" at X
# (`strong`) and "Y_Z2" at Y regulating X remotely (`weak`), as the PSS/E parser builds them.
function _shared_bus_system(; vset = 1.0, ireg = false, strong = true, weak = true,
    bx = 0.2, by = 0.1)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    bg = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230, 1.0, 0.0)
    bx_ = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    by_ = _add_simple_bus!(sys, 4, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_line!(sys, b1, bg, 0.005, 0.05, 0.0)
    _add_simple_line!(sys, bg, bx_, 0.005, 0.05, 0.0)
    _add_simple_line!(sys, bx_, by_, 0.01, 0.3, 0.0)
    _add_simple_load!(sys, bx_, 60.0, 40.0)
    _add_simple_load!(sys, by_, 10.0, 5.0)
    g = _add_simple_thermal_standard!(sys, bg, 0.5, 0.0)
    set_reactive_power_limits!(g, (min = -3.0, max = 3.0))
    ireg && (get_ext(g)["IREG"] = 3)
    for (nm, bus, b, reg, on) in (("X_Z2", bx_, bx, 0, strong), ("Y_Z2", by_, by, 3, weak))
        on || continue
        add_component!(sys, SwitchedAdmittance(; name = nm, available = true, bus = bus,
            Y = b * im, initial_status = [0], number_of_steps = [1], Y_increase = [0.5im],
            admittance_limits = (min = vset, max = vset),
            control_mode = PSY.SwitchedAdmittanceControlMode.CONTINUOUS_VOLTAGE,
            regulated_bus_number = reg,
            ext = Dict{String, Any}("MODSW" => 2, "ADJM" => 0, "RMPCT" => 100.0)))
    end
    return sys
end

# |V_X| and the generator's Q with every shunt at its stored susceptance (no control).
function _shared_bus_stored(; kw...)
    data = PowerFlowData(ACPolarPowerFlow(), _shared_bus_system(; kw...))
    @test solve_power_flow!(data)
    bl = PF.get_bus_lookup(data)
    return data.bus_magnitude[bl[3], 1], data.bus_reactive_power_injections[bl[2], 1]
end

# Remote regulation off: GAP-056's band is the stopgap for an IREG generator PowerFlows does not
# model; with it on, the generator holds X (test_remote_voltage_control.jl).
function _shared_bus_solve(; kw...)
    data = PowerFlowData(
        ACPolarPowerFlow(; control_discrete_devices = true, remote_voltage_control = false),
        _shared_bus_system(; kw...))
    @test solve_power_flow!(data)
    bl = PF.get_bus_lookup(data)
    sh = Dict(d.name => d for d in PF.get_controlled_devices(data).shunts)
    return sh, data.bus_magnitude[bl[3], 1], data.bus_reactive_power_injections[bl[2], 1]
end

@testset "shared regulated bus: enrollment gives a shared point setpoint a hold band (GAP-056)" begin
    δ = PowerFlows.CONTROL_SHARED_SETPOINT_BAND
    for (kw, shared) in (((; weak = false), false), ((; weak = false, ireg = true), true),
        ((;), true))
        sys = _shared_bus_system(; kw...)
        data = PowerFlowData(ACPolarPowerFlow(), sys)
        set = PowerFlows.build_controlled_device_set(sys, PF.get_bus_lookup(data),
            data.power_network_matrix)
        for d in set.shunts
            @test (d.vset_hi - d.vset_lo ≈ 2δ) == shared
            @test d.band == shared
            @test d.vset == 1.0
        end
    end
    g = only(get_components(ThermalStandard, _shared_bus_system(; ireg = true)))
    @test PowerFlows._generator_remote_bus(g) == 3
    get_ext(g)["IREG"] = 2                      # its own bus: local
    @test isnothing(PowerFlows._generator_remote_bus(g))
    delete!(get_ext(g), "IREG")
    get_ext(g)["psse"] = Dict{String, Any}("ireg" => 3)
    @test PowerFlows._generator_remote_bus(g) == 3
end

@testset "shared regulated bus: a shunt beside a remote-regulating generator keeps its setting (GAP-056)" begin
    v0, q0 = _shared_bus_stored(; strong = false)
    vset = round(v0 + 2e-4; digits = 6)     # the stored state misses the setpoint by 2e-4
    # Sole regulator: it ends on its setpoint, as before, taking the generator's part.
    sh, v, q = _shared_bus_solve(; strong = false, vset = vset)
    @test abs(v - vset) < 1e-6
    @test sh["Y_Z2"].current > sh["Y_Z2"].initial
    # The generator regulates X by IREG: held at its stored susceptance, generator Q unchanged.
    sh, v, q = _shared_bus_solve(; strong = false, ireg = true, vset = vset)
    @test sh["Y_Z2"].current == sh["Y_Z2"].initial
    @test v ≈ v0 atol = 1e-8
    @test q ≈ q0 atol = 1e-8
end

@testset "shared regulated bus: two shunts on one bus keep their split, and act outside the band (GAP-056)" begin
    δ = PowerFlows.CONTROL_SHARED_SETPOINT_BAND
    v0, q0 = _shared_bus_stored()
    # On setpoint in the stored state: neither moves (before, both went to one common
    # susceptance, 5 MVAr each).
    sh, v, _ = _shared_bus_solve(; vset = round(v0; digits = 6))
    @test all(d.current == d.initial for d in values(sh))
    # 2δ below the setpoint: they regulate, and end inside the band (on an edge: the
    # continuation's early, shallow stages carry a band this narrow across it).
    vset = round(v0 + 2δ; digits = 6)
    sh, v, _ = _shared_bus_solve(; vset = vset)
    @test all(d.current > d.initial for d in values(sh))
    @test vset - δ - 1e-6 <= v <= vset + δ + 1e-6
    @test min(abs(v - (vset - δ)), abs(v - (vset + δ))) < 1e-6
end

# T152: the exact finish solves at CONTROL_FINISH_SOLVE_TOL, not at the caller's tolerance. At a
# coarse tolerance a finish step whose own mismatch |Δb|·V² is below it was accepted without a
# Newton iteration, so the next round read the voltage from before the step, and a lone continuous
# shunt with a point setpoint ended off it by the order of dV/dQ · tol. (Offsets the continuation
# carries without freezing the device: a frozen device is not finished at all.)
@testset "exact finish: a continuous shunt reaches its point setpoint under a coarse solve tolerance (T152)" begin
    v0, _ = _shared_bus_stored(; weak = false)
    for off in (1e-2, -1e-2, -3e-3)
        vset = round(v0 + off; digits = 6)
        data = PowerFlowData(
            ACPolarPowerFlow(; control_discrete_devices = true, remote_voltage_control = false,
                solver_settings = Dict{Symbol, Any}(:tol => 5e-3)),
            _shared_bus_system(; weak = false, vset = vset))
        tl = Test.TestLogger(; min_level = Logging.Warn)
        @test Logging.with_logger(() -> solve_power_flow!(data), tl)
        @test !any(occursin("freezing device", r.message) for r in tl.logs)
        sh = only(PF.get_controlled_devices(data).shunts)
        @test !sh.band && sh.b_min < sh.current < sh.b_max
        @test abs(data.bus_magnitude[PF.get_bus_lookup(data)[3], 1] - vset) < 1e-8
    end
    @test PowerFlows.CONTROL_FINISH_SOLVE_TOL <= 1e-8
end

# ── Band settle beside a railed shunt (GAP-030) ─────────────────────────────────────────────
# REF(1) ─ X(2) ─ Y(3), loads at X and Y. Discrete (MODSW 1) shunts: capacitor A at X, one
# 0.1 p.u. block, and B at Y, a reactor whose blocks `b_blocks` (one −0.5 p.u. block: levels
# −0.5 and 0) sit at 0. Bands are set from solved voltages: A is inside its band at 0 and at
# one block; B is below its band either way.
function _railed_neighbour_system(; a_binit = 0.0, b_blocks = [-0.5],
    a_band = (0.5, 1.5), b_band = (0.5, 1.5))
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    bx = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    by = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_line!(sys, b1, bx, 0.01, 0.15, 0.0)
    _add_simple_line!(sys, bx, by, 0.01, 0.05, 0.0)
    _add_simple_load!(sys, bx, 20.0, 15.0)
    _add_simple_load!(sys, by, 20.0, 15.0)
    for (nm, bus, binit, dB, band) in (("A", bx, a_binit, [0.1], a_band),
        ("B", by, 0.0, b_blocks, b_band))
        add_component!(sys, SwitchedAdmittance(; name = nm, available = true, bus = bus,
            Y = binit * im, initial_status = zeros(Int, length(dB)),
            number_of_steps = ones(Int, length(dB)), Y_increase = dB .* im,
            admittance_limits = (min = band[1], max = band[2]),
            control_mode = PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE,
            regulated_bus_number = 0, ext = Dict{String, Any}("MODSW" => 1, "ADJM" => 0)))
    end
    return sys
end

function _railed_neighbour_voltages(a_binit)
    data = PowerFlowData(ACPolarPowerFlow(), _railed_neighbour_system(; a_binit = a_binit))
    @test solve_power_flow!(data)
    bl = PF.get_bus_lookup(data)
    return data.bus_magnitude[bl[2], 1], data.bus_magnitude[bl[3], 1]
end

# A latched below its band at one block above its anchor 0 (as the continuation leaves it),
# B latched below its band at 0; then the band settle after a converged solve.
function _railed_neighbour_settle(b_blocks; settle::Bool = true)
    vx0, _ = _railed_neighbour_voltages(0.0)
    vx1, vy1 = _railed_neighbour_voltages(0.1)
    @test vx1 > vx0
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    sys = _railed_neighbour_system(; b_blocks = b_blocks,
        a_band = (vx0 - 0.005, vx1 + 0.02), b_band = (vy1 + 1e-3, vy1 + 0.03))
    data = PowerFlowData(pf, sys)
    set = PF.get_controlled_devices(data)
    sh = Dict(d.name => d for d in set.shunts)
    a, b = sh["A"], sh["B"]
    PowerFlows._band_apply!(a, data, 0.1, 1)
    a.side, a.anchor = -1, 0.0
    b.side, b.anchor = -1, 0.0
    @test PowerFlows._solve_with_q_limits!(pf, data, 1)
    @test PowerFlows._outside_latched_side(b, PowerFlows.measured_value(b, data, 1))
    settle || return a, b, set, data
    PowerFlows._band_settle!(pf, data, set, 1, PowerFlows._snapshot_state(data, 1))
    va = PowerFlows.measured_value(a, data, 1)
    return a, b, va
end

@testset "shunt band: a railed neighbour outside its band does not block a back-off (GAP-030)" begin
    # B has no level above 0: it cannot raise its voltage, so A steps back to its anchor, the
    # first position inside its band.
    a, b, va = _railed_neighbour_settle([-0.5])
    @test isnothing(PowerFlows._next_level(b, 1))
    @test a.current == 0.0
    @test a.vset_lo <= va <= a.vset_hi
    @test b.current == 0.0
    # B with a capacitor block above 0 can still correct: latched outside, it counts against a
    # back-off ...
    _, b, set, data = _railed_neighbour_settle([-0.5, 0.1]; settle = false)
    @test !isnothing(PowerFlows._next_level(b, 1))
    @test PowerFlows._any_latched_outside(set, data, 1)
    # ... and the settle's forward walk steps it up, the direction that corrects it (GAP-068:
    # before, a latched device still at its anchor took no forward step and vetoed A's back-off).
    # Railed there, it no longer blocks A's step back to its anchor.
    a, b, va = _railed_neighbour_settle([-0.5, 0.1])
    @test b.current == 0.1
    @test isnothing(PowerFlows._next_level(b, 1))
    @test a.current == 0.0
    @test a.vset_lo <= va <= a.vset_hi
end

@testset "band settle: which latched devices can still correct (GAP-030, unit)" begin
    levels = collect(range(0.9, 1.1; length = 33))
    t = PowerFlows.ControlledTap("t", 1, 2, 2, 1.0, 1.0 / (0.01 + 0.1im), 0.0, 0.9, 1.1,
        levels, (1, 1, 1, 1), 1.1, 1.1, 1.1, nothing, true, 0.97, 1.03, -1, 1.1)
    # Below its band at the top tap: with dV/dtap > 0 it would need to go up, with < 0 down.
    @test !PowerFlows._can_correct(t, 0.5)
    @test PowerFlows._can_correct(t, -0.5)
    @test PowerFlows._can_correct(t, nothing)          # no gain: counted, the rule before
    @test PowerFlows._can_correct(t, 0.0)
end
