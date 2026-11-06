# GAP-068 (T96): banded voltage taps whose regulated voltage starts (or ends) in band must not walk
# away from it. Synthetic two-tap systems only. Before the fix:
# - `_update_band_state!` re-latched a tap on the far side of its band when its own low-steepness
#   move (S = 100) carried its voltage across the whole band, anchoring it at that off-grid
#   continuous position;
# - the snap then took the nearest level, behind the new anchor, and the band settle's forward walk,
#   stepping in sign(current − anchor), walked the tap away from its band to its limit;
# - the back-off refused any level more than 1e-9 of the range behind the anchor, so a tap whose
#   stored ratio sits a few 1e-6 off its level was never returned to it.
# The expectations are what PSS/E does: the first position inside the band, and an in-band tap held.

function _tbw_bus!(sys, n, t, kv, v = 1.0)
    b = ACBus(; number = n, name = "tbw_b$n", available = true, bustype = t, angle = 0.0,
        magnitude = v, voltage_limits = (0.0, 2.0), base_voltage = Float64(kv))
    add_component!(sys, b)
    return b
end
_tbw_load!(sys, b, p, q) = add_component!(sys, PowerLoad(; name = "tbw_ld$(get_number(b))",
    available = true, bus = b, active_power = p, reactive_power = q, base_power = 100.0,
    max_active_power = p, max_reactive_power = q))
function _tbw_tap!(sys, f, t; name, tap, lims, ntp, band, reg, r, x)
    tx = TapTransformer(; name = name, available = true, active_power_flow = 0.0,
        reactive_power_flow = 0.0, arc = Arc(; from = f, to = t), r = r, x = x,
        primary_shunt = 0.0im, tap = tap, rating = 1.0, base_power = 100.0,
        control_objective = PSY.TransformerControlObjective.VOLTAGE,
        voltage_setpoint = sum(band) / 2, regulated_bus_number = reg,
        ext = Dict{String, Any}("VMI1" => band[1], "VMA1" => band[2],
            "psse" => Dict{String, Any}("CONT1" => reg)))
    add_component!(sys, tx)
    set_tap_limits!(tx, (min = lims[1], max = lims[2]))
    set_number_of_tap_positions!(tx, ntp)
    return tx
end

const _TBW_T_LIMS = (0.9, 1.1)
_tbw_t_level(k) = _TBW_T_LIMS[1] + k * (_TBW_T_LIMS[2] - _TBW_T_LIMS[1]) / 32
const _TBW_T_BAND = (1.0245, 1.037)
const _TBW_U_BAND = (0.965, 0.975)

"""REF(1) ←U— 2 (69 kV, a load) ←T— 3 (25 kV, a 5.2 MW load). U regulates its own winding-1 bus 2
with a 10 mV band; T regulates its own winding-1 bus 3 (radial, CONT1 = its from bus, so
dV3/dT > 0), 33 positions each. Both sit far from mid-range, so the first steepness stage moves a
latched one across its band. With `upair`, U is two equal units in parallel."""
function _tbw_system(; utap, ttap, vref = 0.9837 / utap, upair::Bool = false)
    sys = System(100.0)
    b1 = _tbw_bus!(sys, 1, ACBusTypes.REF, 500, vref)
    add_component!(sys, Source(; name = "tbw_src", available = true, bus = b1,
        active_power = 0.0, reactive_power = 0.0, R_th = 1e-5, X_th = 1e-5))
    b2 = _tbw_bus!(sys, 2, ACBusTypes.PQ, 69)
    b3 = _tbw_bus!(sys, 3, ACBusTypes.PQ, 25)
    _tbw_load!(sys, b2, 0.3, 0.1)
    _tbw_load!(sys, b3, 0.052, 0.02)
    units = upair ? (("U", 0.004, 0.1), ("U2", 0.004, 0.1)) : (("U", 0.002, 0.05),)
    for (nm, r, x) in units
        _tbw_tap!(sys, b2, b1; name = nm, tap = utap, lims = (0.9, 1.1), ntp = 33,
            band = _TBW_U_BAND, reg = 2, r = r, x = x)
    end
    _tbw_tap!(sys, b3, b2; name = "T", tap = ttap, lims = _TBW_T_LIMS, ntp = 33,
        band = _TBW_T_BAND, reg = 3, r = 0.02, x = 0.65)
    return sys
end

function _tbw_solve(sys; tap_direction = :cont_sign)
    pf = ACPolarPowerFlow(; control_discrete_devices = true, tap_direction = tap_direction)
    data = PowerFlowData(pf, sys)
    ok = with_logger(NullLogger()) do
        solve_power_flow!(data)
    end
    taps = Dict(d.name => d for d in PF.get_controlled_devices(data).taps)
    return ok, taps, data.bus_magnitude[:, 1], data
end

"""Every tap locked at `taps` (name => ratio): the bus voltages."""
function _tbw_locked(sys, taps)
    for (n, t) in taps
        set_tap!(get_component(TapTransformer, sys, n), t)
    end
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    with_logger(NullLogger()) do
        solve_power_flow!(data)
    end || error("locked solve failed")
    return data.bus_magnitude[:, 1]
end
_tbw_steps(d) =
    round(Int, (d.current - d.initial) / ((d.p_max - d.p_min) / (length(d.levels) - 1)))
_tbw_inband(v, band) = band[1] <= v <= band[2]

@testset "tap band walk: a tap carried across its band keeps its first edge (GAP-068, unit)" begin
    levels = collect(range(0.9, 1.1; length = 33))
    d = PowerFlows.ControlledTap("t", 1, 2, 2, 1.0, 1.0 / (0.01 + 0.1im), 0.0, 0.9, 1.1,
        levels, (1, 1, 1, 1), 1.0, 1.0, 1.0, nothing, true, 0.97, 1.03, 0, 1.0)
    PowerFlows._update_band_state!(d, 0.96)          # below: latched −1 at its stored tap
    @test (d.side, d.anchor) == (-1, 1.0)
    d.current = 1.04                                 # its own move carried it above the band
    PowerFlows._update_band_state!(d, 1.035)
    @test (d.side, d.anchor) == (-1, 1.0)            # no flip, no off-grid anchor
    d.current = 1.0                                  # back at its anchor, still above: reversal
    PowerFlows._update_band_state!(d, 1.035)
    @test (d.side, d.anchor) == (1, 1.0)
    # The snap of a latched tap exactly at its anchor goes the correcting way, −side·sign(g).
    d.side, d.anchor, d.current = 1, 1.0045, 1.0045  # above its band, off grid
    @test PowerFlows.snap_to_discrete(d, 1.0045) ≈ 1.00625          # no gain: nearest
    @test PowerFlows.snap_to_discrete(d, 1.0045, 0.5) ≈ 1.0         # dV/dtap > 0: down
    @test PowerFlows.snap_to_discrete(d, 1.0045, -0.5) ≈ 1.00625    # dV/dtap < 0: up
    @test PowerFlows._correcting_direction(d, 0.5) == -1
    @test PowerFlows._correcting_direction(d, -0.5) == 1
    d.current = 1.0125
    @test PowerFlows._correcting_direction(d, nothing) == 1          # no gain: as it moved
end

@testset "tap band walk: a tap latched across its band does not walk to its limit (GAP-068)" begin
    for tap_direction in (:cont_sign, :sensitivity)
        # U starts just above its band: PSS/E steps it once, to the first position inside.
        sys = _tbw_system(; utap = 1.0625, ttap = _tbw_t_level(27))
        v0 = _tbw_locked(deepcopy(sys), Dict("U" => 1.0625, "T" => _tbw_t_level(27)))
        @test v0[2] > _TBW_U_BAND[2] && v0[2] - _TBW_U_BAND[2] < 1e-3
        ok, taps, v, data = _tbw_solve(sys; tap_direction)
        u = taps["U"]
        @test ok
        @test _tbw_steps(u) == -1
        @test _tbw_inband(v[2], _TBW_U_BAND)
        @test _tbw_inband(v[3], _TBW_T_BAND)
        @test any(abs(u.current - lv) < 1e-12 for lv in u.levels)
        @test PF.get_control_batched_fallback_count(data) == 0
    end
end

@testset "tap band walk: an in-band tap a hair off its level returns to it (GAP-068)" begin
    for δ in (2e-6, 3e-6, -3e-6, 0.0)
        # T stored δ off level 28, its voltage in band; U steps once (legitimately), and with U
        # there and T at its stored ratio V3 is still in band, so PSS/E leaves T alone.
        t0 = _tbw_t_level(28) + δ
        sys = _tbw_system(; utap = 1.075, ttap = t0)
        ok, taps, v, _ = _tbw_solve(deepcopy(sys))
        u, t = taps["U"], taps["T"]
        @test ok
        @test _tbw_steps(u) == -1
        @test _tbw_inband(v[2], _TBW_U_BAND)
        vchk = _tbw_locked(deepcopy(sys), Dict("U" => u.current, "T" => t0))
        @test _tbw_inband(vchk[3], _TBW_T_BAND)       # T's stored ratio is in band there
        @test t.current == t0                          # returned to its stored ratio exactly
        @test _tbw_inband(v[3], _TBW_T_BAND)
    end
end

@testset "tap band walk: a parallel pair carried across its band moves together (GAP-068)" begin
    sys = _tbw_system(; utap = 1.0625, ttap = _tbw_t_level(27), upair = true)
    ok, taps, v, _ = _tbw_solve(sys)
    @test ok
    @test _tbw_steps(taps["U"]) == _tbw_steps(taps["U2"]) == -1
    @test taps["U"].current == taps["U2"].current
    @test _tbw_inband(v[2], _TBW_U_BAND)
end

# T161 (GAP-080): the band settle judged a banded tap's band on the network as the snap left it,
# with every continuous device still at the continuation's fixed point, off its setpoint by the
# sigmoid law's residue, and the exact finish that puts them on their setpoints came after it.
# REF(1) ─ 2 ←T─ 3 (radial, 69 kV side 3): a continuous MODSW 2 shunt at 2 holds 2 at a point
# setpoint; T regulates its winding-1 bus 3 inside a band whose top lies `m` above where bus 3
# ends with T at its stored ratio. T's bus is inside its band throughout in PSS/E's terms, but
# the shunt's residue (bus 2 above its setpoint) lifted bus 3 past the top at the snap: T latched and
# stepped down, and the step back failed the same test. The shunt's own bus hardly moves with T,
# so once it is finished first the back-off (or no latch at all) keeps T where it was.
function _tbw_hv_shunt_system(; hi, vset2 = 1.0, b0 = 0.15)
    sys = System(100.0)
    b1 = _tbw_bus!(sys, 1, ACBusTypes.REF, 230)
    add_component!(sys, Source(; name = "tbw_src", available = true, bus = b1,
        active_power = 0.0, reactive_power = 0.0, R_th = 1e-5, X_th = 1e-5))
    b2 = _tbw_bus!(sys, 2, ACBusTypes.PQ, 230)
    b3 = _tbw_bus!(sys, 3, ACBusTypes.PQ, 69)
    add_component!(sys, Line(; name = "tbw_l12", available = true, active_power_flow = 0.0,
        reactive_power_flow = 0.0, arc = Arc(; from = b1, to = b2), r = 0.01, x = 0.1,
        b = (from = 0.0, to = 0.0), rating = 1.0, angle_limits = (min = -pi / 2, max = pi / 2)))
    _tbw_load!(sys, b2, 0.5, 0.3)
    _tbw_load!(sys, b3, 0.1, 0.05)
    _tbw_tap!(sys, b3, b2; name = "T", tap = 1.0, lims = (0.9, 1.1), ntp = 201,
        band = (hi - 0.1, hi), reg = 3, r = 0.002, x = 0.08)
    add_component!(sys, SwitchedAdmittance(; name = "S2", available = true, bus = b2,
        Y = b0 * im, initial_status = [0], number_of_steps = [1], Y_increase = [3.0im],
        admittance_limits = (min = vset2, max = vset2),
        control_mode = PSY.SwitchedAdmittanceControlMode.CONTINUOUS_VOLTAGE,
        regulated_bus_number = 0,
        ext = Dict{String, Any}("MODSW" => 2, "ADJM" => 0, "RMPCT" => 100.0)))
    return sys
end

@testset "band settle reads the network with continuous devices finished: an in-band tap keeps its position (T161, GAP-080)" begin
    # Where the network ends with the shunt at 0.3 and T at its stored ratio: no control.
    ref = PowerFlowData(ACPolarPowerFlow(), _tbw_hv_shunt_system(; hi = 2.0, b0 = 0.3))
    @test with_logger(() -> solve_power_flow!(ref), NullLogger())
    bl = PF.get_bus_lookup(ref)
    v2, v3 = ref.bus_magnitude[bl[2], 1], ref.bus_magnitude[bl[3], 1]
    for m in (5e-5, 1e-4, 2e-4)
        hi = round(v3 + m; digits = 7)
        ok, taps, V, data = _tbw_solve(
            _tbw_hv_shunt_system(; hi = hi, vset2 = round(v2; digits = 7)))
        @test ok
        t = taps["T"]
        s = only(PF.get_controlled_devices(data).shunts)
        @test abs(V[bl[2]] - s.vset) < 1e-8                      # the shunt on its setpoint
        @test t.current == t.initial                              # T held, as PSS/E holds it
        @test hi - 0.1 < V[bl[3]] <= hi                           # its bus inside the band
        @test V[bl[3]] ≈ hi - m atol = 2e-6
    end
end
