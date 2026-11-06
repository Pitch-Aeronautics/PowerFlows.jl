abstract type AbstractControlledDevice end
abstract type AbstractBranchControl <: AbstractControlledDevice end
abstract type AbstractShuntControl <: AbstractControlledDevice end

"""Voltage-controlling tap transformer. `nz_offsets` are the 4 cached
`nzval` linear indices of the (from,to)×(from,to) Y-bus block. The control
orientation is NOT stored: it comes from the measured plant sensitivity dV/dp
(see `_control_target`), which is correct for any wiring of the controlled bus.

Impedance correction (PSS/E tables, `PSY.ImpedanceCorrectionData`): PNM's `Ybus` scales the
series impedance by the table's factor at the build-time tap. `correction` holds a
`TAP_RATIO`-mode table, re-evaluated at every tap the controller applies, so a step is sized
on, and the Y-bus re-stamped with, the corrected impedance at the new tap (see
[`_series_admittance`](@ref)). A `PHASE_SHIFT_ANGLE`-mode table does not depend on the tap
(the angle of a `TapTransformer` is its fixed winding-group shift): its factor is folded into
`yt` at enrollment and `correction` is `nothing`.

Voltage band (PSS/E `[VMI1, VMA1]`): when `band` is set the tap is held while its regulated
voltage lies in `[v_lo, v_hi]`, and one outside the band regulates toward the violated edge
and stops at the first discrete step inside it (see `_update_band_state!`, `snap_to_discrete`
and `_band_settle!`). `side` is the latched violation (−1 below `v_lo`, +1 above `v_hi`, 0 not
regulating) and `anchor` the tap where it latched; both are continuation scratch, reset at the
start of every `_control_continuation!`. Without a band the tap is driven to `vset`.

Orientation: the continuation orients a tap by its measured dV/dp. With `orientation` ±1
(`tap_direction = :cont_sign`, from PSS/E's CONT1) the sign of dV/dp is fixed instead, as PSS/E
adjusts the ratio: a regulated bus on the winding-2 (to) side, where raising the ratio lowers
its voltage, is −1, one on the winding-1 (from) side +1. A terminal bus is on its own side; a
remote bus is placed by CONT1's sign, > 0 the winding-2 side, < 0 the winding-1 side. The
measured magnitude still sizes each step. `aligned` is set when parallel taps regulating one
bus measure opposite signs, and `offset` keeps a parallel unit's ratio offset from its group, so
the units move by the same amount (see `_align_parallel_taps!`)."""
mutable struct ControlledTap <: AbstractBranchControl
    name::String
    from_ix::Int
    to_ix::Int
    controlled_ix::Int
    vset::Float64
    yt::ComplexF64                   # 1/(r+jx), divided by any tap-independent correction factor
    alpha::Float64                   # winding-group phase shift (PSY.get_α)
    p_min::Float64
    p_max::Float64
    levels::Vector{Float64}          # discrete tap ratios
    nz_offsets::NTuple{4, Int}       # nzval idx for Y11,Y12,Y21,Y22
    initial::Float64                 # enrollment-time tap (reporting)
    synced::Float64                  # tap reflected in the arc-admittance rows
    current::Float64
    correction::Union{Nothing, IS.PiecewiseLinearData}   # TAP_RATIO table, or nothing
    band::Bool                       # hold inside [v_lo, v_hi] (PSS/E); false ⇒ drive to vset
    v_lo::Float64                    # VMI1
    v_hi::Float64                    # VMA1
    side::Int                        # latched violation: −1 below, +1 above, 0 none
    anchor::Float64                  # tap where the current latch began
    orientation::Int                 # sign of dV/dp fixed by PSS/E's CONT sign; 0 ⇒ measured
    aligned::Int                     # continuation scratch: sign a parallel group imposed
    offset::Float64                  # continuation scratch: ratio offset within a parallel group
end

# Positional forms without the newer fields: no impedance correction table, point setpoint,
# orientation by measured sensitivity.
ControlledTap(
    name, from_ix, to_ix, controlled_ix, vset, yt, alpha, p_min, p_max, levels, nz_offsets,
    initial, synced, current, correction = nothing,
) = ControlledTap(
    name, from_ix, to_ix, controlled_ix, vset, yt, alpha, p_min, p_max, levels, nz_offsets,
    initial, synced, current, correction, false, vset, vset, 0, current,
)
ControlledTap(
    name, from_ix, to_ix, controlled_ix, vset, yt, alpha, p_min, p_max, levels, nz_offsets,
    initial, synced, current, correction, band, v_lo, v_hi, side, anchor,
) = ControlledTap(
    name, from_ix, to_ix, controlled_ix, vset, yt, alpha, p_min, p_max, levels, nz_offsets,
    initial, synced, current, correction, band, v_lo, v_hi, side, anchor, 0, 0, 0.0,
)

"""Active-power-controlling phase shifter (PSS/E COD1 = 3, GAP-029): a `PhaseShiftingTransformer`
whose angle `α` (the parameter, radians) steps over `levels` (NTP1 positions over
[RMI1, RMA1]; empty for a continuous shifter) to hold its active power flow inside the band
`[v_lo, v_hi]` (VMI1/VMA1 in MW over the system base, so p.u.). The flow is the winding-1 →
winding-2 active power measured at the metered end (`metered` 1, the default, or 2 from PSS/E's
NMETR): at winding 1 the power entering the transformer from the `from` bus, at winding 2 the
power leaving it into the `to` bus. The ratio `tap` is fixed. Its Y-bus block and arc rows are
delta-updated as a tap's are, with `t = tap·e^{jα}` varying in angle (see `_branch_terms`).

The band, latch, directional snap, settle and parallel-unit alignment are the tap family's
(`BandedDevice`, `_align_parallel_taps!`). The orientation always comes from the measured
dP/dα (`orientation` is 0). A unit of a parallel group (same terminals) measures the group's
mean flow (`group`, continuation scratch), so the units regulate one quantity, as parallel taps
regulate one bus, and move by equal steps with their angle offsets kept (`offset`).

`correction` holds a `PHASE_SHIFT_ANGLE`-mode impedance correction table, re-evaluated at every
angle (in degrees, as PNM evaluates it); a `TAP_RATIO`-mode table depends only on the fixed
ratio and is folded into `yt`."""
mutable struct ControlledPhaseShifter <: AbstractBranchControl
    name::String
    from_ix::Int
    to_ix::Int
    metered::Int                     # 1: flow measured at winding 1 (from), 2: at winding 2
    tap::Float64                     # fixed ratio
    yt::ComplexF64                   # 1/(r+jx), divided by any angle-independent factor
    y_shunt::ComplexF64              # primary (magnetizing) shunt, in the measured flow at winding 1
    vset::Float64                    # band midpoint (p.u. MW); the target without a band
    p_min::Float64                   # RMI1 (rad)
    p_max::Float64                   # RMA1 (rad)
    levels::Vector{Float64}          # discrete angles (rad); empty ⇒ continuous
    nz_offsets::NTuple{4, Int}
    initial::Float64
    synced::Float64
    current::Float64
    correction::Union{Nothing, IS.PiecewiseLinearData}   # PHASE_SHIFT_ANGLE table, or nothing
    band::Bool
    v_lo::Float64                    # VMI1 / base (p.u.)
    v_hi::Float64                    # VMA1 / base (p.u.)
    side::Int
    anchor::Float64
    orientation::Int                 # always 0: oriented by the measured dP/dα
    aligned::Int
    offset::Float64
    group::Vector{ControlledPhaseShifter}   # continuation scratch: parallel members, self included
end

# Tap-like branch devices: the parameter enters the 2×2 branch block through `t`.
const BranchDevice = Union{ControlledTap, ControlledPhaseShifter}

"""The sign of dV/dp the continuation must use for device `d`, or 0 to use the measured one:
a tap's PSS/E CONT-sign orientation (`tap_direction = :cont_sign`), else the sign its parallel
group was aligned to (`_align_parallel_taps!`)."""
_forced_sign(::AbstractControlledDevice) = 0
_forced_sign(d::BranchDevice) = iszero(d.orientation) ? d.aligned : d.orientation

"""Slope of PNM's correction-table interpolation (`PNM._interpolate_correction_factor`) at
`x`: the slope of the segment that interpolation uses there, and zero outside the table (the
end values are held)."""
function _correction_factor_slope(curve::IS.PiecewiseLinearData, x::Float64)
    points = IS.get_points(curve)
    (x < points[1].x || x > points[end].x) && return 0.0
    @inbounds for i in 1:(length(points) - 1)
        if x <= points[i + 1].x
            dx = points[i + 1].x - points[i].x
            iszero(dx) && return 0.0
            return (points[i + 1].y - points[i].y) / dx
        end
    end
    return 0.0
end

"""Series admittance of tap device `d` at tap ratio `p`: `yt`, divided by the impedance
correction factor at `p` when a `TAP_RATIO` table is attached. PNM evaluates the same table at
`abs(tap)` with the same interpolation, so at the enrollment tap this is the Y-bus's own
corrected admittance."""
@inline function _series_admittance(d::ControlledTap, p::Float64)
    curve = d.correction
    isnothing(curve) && return d.yt
    return d.yt / PNM._interpolate_correction_factor(curve, abs(p))
end

"""`(yt(p), dyt/dp)` for tap device `d`: with a correction table `yt(p) = yt/F(p)`, so
`dyt/dp = −yt(p)·F′(p)/F(p)`; without one the derivative is zero."""
@inline function _series_admittance_and_slope(d::ControlledTap, p::Float64)
    curve = d.correction
    isnothing(curve) && return d.yt, zero(ComplexF64)
    F = PNM._interpolate_correction_factor(curve, abs(p))
    y = d.yt / F
    return y, -y * _correction_factor_slope(curve, abs(p)) * sign(p) / F
end

"""Series admittance of phase shifter `d` at angle `α` (rad): `yt`, divided by the
`PHASE_SHIFT_ANGLE` correction factor at `α` in degrees when a table is attached (PNM's own
evaluation, so at the enrollment angle this is the Y-bus's corrected admittance)."""
@inline function _series_admittance(d::ControlledPhaseShifter, α::Float64)
    curve = d.correction
    isnothing(curve) && return d.yt
    return d.yt / PNM._interpolate_correction_factor(curve, rad2deg(α))
end

"""`(yt(α), dyt/dα)` for phase shifter `d` (α in rad; the table's x is degrees)."""
@inline function _series_admittance_and_slope(d::ControlledPhaseShifter, α::Float64)
    curve = d.correction
    isnothing(curve) && return d.yt, zero(ComplexF64)
    x = rad2deg(α)
    F = PNM._interpolate_correction_factor(curve, x)
    y = d.yt / F
    return y, -y * _correction_factor_slope(curve, x) * (180 / π) / F
end

"""Voltage-controlling switched shunt (PSS/E MODSW 1 discrete, 2 continuous).

`levels` are the realizable total susceptances in ascending order and `level_blocks[k, :]`
the per-block step counts that realize `levels[k]` (see [`_shunt_levels`](@ref)): with PSS/E
`ADJM = 0` the cumulative block-activation chains (capacitor blocks switch on in listed order,
reactor blocks likewise, each off in reverse), with `ADJM = 1` every combination of steps, so a
step goes to the next higher or lower total admittance.

Voltage band (PSS/E `[VSWLO, VSWHI]`, `vset_lo`/`vset_hi`): the shunt is held while its
regulated voltage is inside. With `band` set, one outside latches the violated edge (`side`
−1 below, +1 above, `anchor` the susceptance where it latched), regulates toward it, and a
discrete shunt stops at the first level inside (directional snap, then `_band_settle!`); a
continuous one ends `finish_margin` inside the edge (0, on the edge, for a PSS/E band; half the
band for GAP-056's hold band, `_share_regulated_setpoints!`). Without `band`
(`shunt_voltage_band = false`, the rule before the band was implemented) it regulates toward
`vset`, the band midpoint, is held as soon as its voltage crosses into the band, and snaps to the
nearest level."""
mutable struct ControlledSwitchedShunt <: AbstractShuntControl
    name::String
    bus_ix::Int
    controlled_ix::Int
    vset::Float64
    vset_lo::Float64                 # VSWLO/VSWHI deadband: held anywhere inside
    vset_hi::Float64
    g0::Float64                      # real(get_Y)
    b0::Float64                      # fixed (non-switchable) susceptance base
    block_steps::Vector{Int}         # number_of_steps per block
    block_dB::Vector{Float64}        # imag(Y_increase) per block
    b_min::Float64
    b_max::Float64
    block_n::Vector{Int}             # per-block step counts of the last snap (reporting)
    continuous::Bool                 # MODSW==2 ⇒ continuous regulation (no discrete snap)
    initial::Float64                 # enrollment-time susceptance (reporting)
    current::Float64                 # current total susceptance b
    psse_convention::Bool            # true ⇒ PSS/E parser convention (Y=BINIT total,
    # initial_status zeroed); false ⇒ PSY API convention (Y is the fixed base,
    # initial_status meaningful). Determines how write_device_settings! sources Y/status.
    levels::Vector{Float64}          # realizable totals, ascending
    level_blocks::Matrix{Int}        # level_blocks[k, j]: steps of block j at levels[k]
    band::Bool                       # PSS/E band: first level inside; false ⇒ midpoint rule
    side::Int                        # latched violation: −1 below, +1 above, 0 none
    anchor::Float64                  # susceptance where the current latch began
    finish_margin::Float64           # continuous: how far inside a latched edge it finishes
end

# Positional form without the finish margin (a PSS/E band: a continuous device ends on the edge).
ControlledSwitchedShunt(
    name, bus_ix, controlled_ix, vset, vset_lo, vset_hi, g0, b0, block_steps, block_dB,
    b_min, b_max, block_n, continuous, initial, current, psse_convention, levels,
    level_blocks, band, side, anchor,
) = ControlledSwitchedShunt(
    name, bus_ix, controlled_ix, vset, vset_lo, vset_hi, g0, b0, block_steps, block_dB,
    b_min, b_max, block_n, continuous, initial, current, psse_convention, levels,
    level_blocks, band, side, anchor, 0.0,
)

# Positional form without the level table and band (unit tests, older callers): the PSS/E
# ADJM = 0 chains, and the midpoint rule.
ControlledSwitchedShunt(
    name, bus_ix, controlled_ix, vset, vset_lo, vset_hi, g0, b0, block_steps, block_dB,
    b_min, b_max, block_n, continuous, initial, current, psse_convention,
) = ControlledSwitchedShunt(
    name, bus_ix, controlled_ix, vset, vset_lo, vset_hi, g0, b0, block_steps, block_dB,
    b_min, b_max, block_n, continuous, initial, current, psse_convention,
    _shunt_levels(b0, block_steps, block_dB, 0)..., false, 0, current, 0.0,
)

"""`(levels, level_blocks)` of a switched shunt with fixed base `b0`: the realizable total
susceptances in ascending order and, row by row, the per-block step counts that realize them.

- `adjm = 0` (PSS/E ADJM 0, the only method before PSS/E 32): blocks switch on in input order
  and off in reverse. Capacitor blocks (dB > 0) and reactor blocks (dB < 0) form two chains
  stepping away from the all-off base, so the totals are `b0`, `b0` plus each capacitor
  prefix and `b0` plus each reactor prefix. PSS/E lists reactor blocks first, then capacitor
  blocks; the chains are taken by sign, so a bank listed otherwise keeps each sign's order.
- `adjm = 1` (PSS/E ADJM 1): steps and blocks switch so that the next higher (or lower) total
  admittance is reached: every combination of steps, each distinct total once, realized by the
  combination with the fewest steps switched on. Above `CONTROL_SHUNT_MAX_COMBINATIONS`
  combinations it falls back to the ADJM 0 chains with a warning."""
function _shunt_levels(b0::Float64, steps::Vector{Int}, dB::Vector{Float64}, adjm::Int;
    name::String = "")
    nb = length(steps)
    if adjm == 1
        ncomb = prod(big.(steps) .+ 1; init = big(1))
        if ncomb <= CONTROL_SHUNT_MAX_COMBINATIONS
            return _shunt_levels_by_total(b0, steps, dB)
        end
        @warn "ControlledSwitchedShunt \"$name\": ADJM 1 with $ncomb step combinations \
            (over $CONTROL_SHUNT_MAX_COMBINATIONS); switching blocks in input order (ADJM 0)."
    end
    rows = Vector{Vector{Int}}()
    totals = Float64[]
    for positive in (false, true)
        n = zeros(Int, nb)
        total = b0
        for k in 1:nb
            (positive ? dB[k] > 0.0 : dB[k] < 0.0) || continue
            for _ in 1:steps[k]
                n[k] += 1
                total += dB[k]
                push!(totals, total)
                push!(rows, copy(n))
            end
        end
    end
    push!(totals, b0)
    push!(rows, zeros(Int, nb))
    return _sorted_levels(totals, rows, nb)
end

function _shunt_levels_by_total(b0::Float64, steps::Vector{Int}, dB::Vector{Float64})
    nb = length(steps)
    best = Dict{Float64, Vector{Int}}()
    n = zeros(Int, nb)
    while true
        total = b0 + sum(n .* dB; init = 0.0)
        key = round(total; digits = 12)
        cur = get(best, key, nothing)
        if isnothing(cur) || sum(n) < sum(cur)
            best[key] = copy(n)
        end
        # Odometer over n[k] ∈ 0:steps[k].
        k = 1
        while k <= nb && n[k] == steps[k]
            n[k] = 0
            k += 1
        end
        k > nb && break
        n[k] += 1
    end
    rows = collect(values(best))
    totals = [b0 + sum(r .* dB; init = 0.0) for r in rows]
    return _sorted_levels(totals, rows, nb)
end

function _sorted_levels(totals::Vector{Float64}, rows::Vector{Vector{Int}}, nb::Int)
    perm = sortperm(totals)
    levels = totals[perm]
    blocks = zeros(Int, length(levels), nb)
    for (i, p) in enumerate(perm)
        blocks[i, :] .= rows[p]
    end
    return levels, blocks
end

"""Continuous shunt SVC/STATCOM (`FACTSControlDevice`). Holds `vset` at `controlled_ix`
(the regulated bus, possibly remote via FCREG) by varying a symmetric shunt susceptance
`b ∈ [-b_lim, b_lim]` (negative = inductive, positive = capacitive); the injected reactive
power is `b·|V|²`. Applied through the constant-Z reactive-withdrawal slot (never the
Y-bus), so a step does not invalidate a fast-decoupled B″ factorization. `svc` selects the
limit law (`_facts_b_limit`); `b_lim` is the current effective |b| bound, refreshed each
outer iteration from the measured controlled-bus voltage. At a susceptance limit the clamp
holds it there — the homotopy equivalent of the PV→PQ Q-limit release."""
mutable struct ControlledFACTS <: AbstractShuntControl
    name::String
    bus_ix::Int
    controlled_ix::Int
    vset::Float64
    svc::Bool          # true ⇒ SVC (susceptance-bounded); false ⇒ STATCOM (current-limited)
    rating::Float64    # SHMX/base_mva: SVC susceptance-at-unity bound, STATCOM current limit
    q_cap::Float64     # independent MVA ceiling/base_mva (get_max_reactive_power)
    b_lim::Float64     # current effective |b| bound (refreshed per iteration)
    base_mva::Float64  # system base, to report delivered Q (b·|V|²) in MVA
    initial::Float64   # enrollment-time susceptance (reporting)
    current::Float64   # current susceptance b
    saturated::Bool    # post-solve classification: held at the V-dependent |b| bound, off setpoint
end

struct ControlledDeviceSet
    taps::Vector{ControlledTap}
    shunts::Vector{ControlledSwitchedShunt}
    facts::Vector{ControlledFACTS}
    # Per-time-step scalar state [device, ts]; `load_device_state!`/`save_device_state!` swap it
    # in/out of the device scratch. Taps mutate the SHARED Y-bus, so their store is reconciled via
    # `apply_parameter!` (reset to `d.initial` each step), not a scalar swap.
    shunt_current::Matrix{Float64}
    shunt_block_n::Matrix{Vector{Int}}
    facts_current::Matrix{Float64}
    facts_b_lim::Matrix{Float64}
    facts_saturated::Matrix{Bool}
    tap_current::Matrix{Float64}
    # Tap device index by component name (`PSY.TapTransformer` names are unique), for
    # post-processing that recomputes a moved tap's branch flow at its solved ratio.
    tap_index::Dict{String, Int}
    # Phase shifters (COD1 = 3), their per-ts angle store and index by component name: the
    # fourth family, after FACTS in the continuation's [taps; shunts; facts; phase shifters]
    # indexing. Like taps they mutate the shared Y-bus (reset-to-baseline per time step).
    phase_shifters::Vector{ControlledPhaseShifter}
    pst_current::Matrix{Float64}
    pst_index::Dict{String, Int}
    # Perf counters for the last `_control_continuation!` (counts, not wall-clock; the regression
    # harness). symbolic_factors stays O(1) per continuation with PolarNRCache reuse. See getters.
    inner_solves::Base.RefValue{Int}
    symbolic_factors::Base.RefValue{Int}
    numeric_refactors::Base.RefValue{Int}
    # Passes that wanted the batched path but ran the sequential per-device path (a failed joint
    # solve or sensitivity refresh), and sensitivity-context rebuilds after a bus-type change.
    batched_fallbacks::Base.RefValue{Int}
    context_rebuilds::Base.RefValue{Int}
    # The control solve's time limit (`control_time_limit`, see `_ac_power_flow_with_area_relax!`):
    # the `time()` past which no further inner solve starts (Inf: none), and whether the last
    # solve stopped there.
    deadline::Base.RefValue{Float64}
    time_limit_reached::Base.RefValue{Bool}
    # The control solve's inner-solve limit (`control_solve_limit`): the inner solves the armed
    # call may start (`typemax(Int)`: none), how many it has started, and whether the last solve
    # stopped there. A count, so where it stops does not depend on the machine.
    solve_limit::Base.RefValue{Int}
    solves_started::Base.RefValue{Int}
    solve_limit_reached::Base.RefValue{Bool}
end
function ControlledDeviceSet(
    taps::Vector{ControlledTap},
    shunts::Vector{ControlledSwitchedShunt},
    facts::Vector{ControlledFACTS},
    n_time_steps::Int = 1;
    phase_shifters::Vector{ControlledPhaseShifter} = ControlledPhaseShifter[],
)
    shunt_current = [d.current for d in shunts, _ in 1:n_time_steps]
    shunt_block_n = [copy(d.block_n) for d in shunts, _ in 1:n_time_steps]
    facts_current = [d.current for d in facts, _ in 1:n_time_steps]
    facts_b_lim = [d.b_lim for d in facts, _ in 1:n_time_steps]
    facts_saturated = [d.saturated for d in facts, _ in 1:n_time_steps]
    tap_current = [d.current for d in taps, _ in 1:n_time_steps]
    return ControlledDeviceSet(
        taps,
        shunts,
        facts,
        shunt_current,
        shunt_block_n,
        facts_current,
        facts_b_lim,
        facts_saturated,
        tap_current,
        Dict{String, Int}(d.name => i for (i, d) in enumerate(taps)),
        phase_shifters,
        [d.current for d in phase_shifters, _ in 1:n_time_steps],
        Dict{String, Int}(d.name => i for (i, d) in enumerate(phase_shifters)),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(Inf),
        Ref(false),
        Ref(typemax(Int)),
        Ref(0),
        Ref(false),
    )
end
function Base.isempty(s::ControlledDeviceSet)
    return isempty(s.taps) && isempty(s.shunts) && isempty(s.facts) &&
           isempty(s.phase_shifters)
end

"""Load time step `ts`'s persisted state into the device scratch `_control_continuation!` mutates.
Shunt/FACTS is a plain scalar swap; taps instead reset to their enrollment baseline (`d.initial`)
via a Y-bus delta-update, since they mutate the shared Y-bus and every step must regulate from the
same baseline network (reset-to-baseline design; a no-op on step 1)."""
function load_device_state!(set::ControlledDeviceSet, data, ts::Int)
    for (i, d) in enumerate(set.shunts)
        d.current = set.shunt_current[i, ts]
        d.block_n .= set.shunt_block_n[i, ts]
    end
    for (i, d) in enumerate(set.facts)
        d.current = set.facts_current[i, ts]
        d.b_lim = set.facts_b_lim[i, ts]
        d.saturated = set.facts_saturated[i, ts]
    end
    for d in set.taps
        apply_parameter!(d, data, d.initial, ts)
    end
    for d in set.phase_shifters
        apply_parameter!(d, data, d.initial, ts)
    end
    return
end
load_device_state!(::Nothing, data, ts::Int) = nothing

"""Persist the device scratch for time step `ts` into the per-ts store after `_control_continuation!`
runs for that step. Taps persist their converged position for reporting; the next `load_device_state!`
resets the Y-bus to baseline before the next step regulates."""
function save_device_state!(set::ControlledDeviceSet, data, ts::Int)
    for (i, d) in enumerate(set.shunts)
        set.shunt_current[i, ts] = d.current
        set.shunt_block_n[i, ts] .= d.block_n
    end
    for (i, d) in enumerate(set.facts)
        set.facts_current[i, ts] = d.current
        set.facts_b_lim[i, ts] = d.b_lim
        set.facts_saturated[i, ts] = d.saturated
    end
    for (i, d) in enumerate(set.taps)
        set.tap_current[i, ts] = d.current
    end
    for (i, d) in enumerate(set.phase_shifters)
        set.pst_current[i, ts] = d.current
    end
    return
end
save_device_state!(::Nothing, data, ts::Int) = nothing

"""Width (number of time steps) of the per-time-step device store; all store matrices share
it by construction."""
store_time_steps(set::ControlledDeviceSet) = size(set.shunt_current, 2)

"""Guard the per-ts store width against the data horizon at the solve seam: without it a set built
for a different horizon would only surface as a `BoundsError` deep inside the solve."""
function validate_device_store_width(set::ControlledDeviceSet, n_time_steps::Int)
    width = store_time_steps(set)
    if width != n_time_steps
        error(
            "ControlledDeviceSet per-time-step store has width $width but the " *
            "PowerFlowData horizon is $n_time_steps time steps. Rebuild the set with " *
            "`build_controlled_device_set(...; n_time_steps = $n_time_steps)`.",
        )
    end
    return
end
validate_device_store_width(::Nothing, ::Int) = nothing

"""Number of inner `_solve_with_q_limits!` calls the last discrete-control continuation
performed (0 when the data was built without discrete control)."""
function get_control_inner_solve_count(data)
    isnothing(data.controlled_devices) && return 0
    return data.controlled_devices.inner_solves[]
end

"""Whether the last discrete-control solve stopped because its time limit
(`control_time_limit`) passed: it then returned not converged, with no inner solve started after
the limit. `false` when the data was built without discrete control."""
function get_control_time_limit_reached(data)
    isnothing(data.controlled_devices) && return false
    return data.controlled_devices.time_limit_reached[]
end

"""Whether the last discrete-control solve stopped because it had started as many inner solves as
its limit (`control_solve_limit`) allows: it then returned not converged. `false` when the data was
built without discrete control."""
function get_control_solve_limit_reached(data)
    isnothing(data.controlled_devices) && return false
    return data.controlled_devices.solve_limit_reached[]
end

"""Number of inner solves the last discrete-control solve (`_ac_power_flow_with_area_relax!`)
started, over every continuation it ran (area relaxes included); 0 when the data was built
without discrete control."""
function get_control_solves_started(data)
    isnothing(data.controlled_devices) && return 0
    return data.controlled_devices.solves_started[]
end

# Has the armed inner-solve limit been used up, or the armed time limit passed? Records which on
# the set; one comparison, and one `time()` call when a time limit is armed, made only between
# inner solves.
@inline function _control_time_up!(set::ControlledDeviceSet)
    if set.solves_started[] >= set.solve_limit[]
        set.solve_limit_reached[] = true
        return true
    end
    isinf(set.deadline[]) && return false
    time() >= set.deadline[] || return false
    set.time_limit_reached[] = true
    return true
end

# Did the running control solve stop at either limit?
@inline _control_stopped(set::ControlledDeviceSet) =
    set.time_limit_reached[] || set.solve_limit_reached[]

"""Number of KLU/AA SYMBOLIC factorizations performed inside the last discrete-control
continuation. With `PolarNRCache` symbolic reuse this stays O(1) per continuation even as
`inner_solves` grows; without it, it tracks `inner_solves`. 0 when built without control."""
function get_control_symbolic_factor_count(data)
    isnothing(data.controlled_devices) && return 0
    return data.controlled_devices.symbolic_factors[]
end

# Hot-path instrumentation. The `nothing` (non-control) case — every ordinary AC solve — is a
# single branch and early return, mirroring `_ctrl_solve!`. Only the continuation path counts.
@inline function _count_symbolic_factor!(data)
    cd = data.controlled_devices
    isnothing(cd) || (cd.symbolic_factors[] += 1)
    return
end
@inline function _count_numeric_refactor!(data)
    cd = data.controlled_devices
    isnothing(cd) || (cd.numeric_refactors[] += 1)
    return
end

"""Number of passes of the last discrete-control continuation that wanted the batched (one joint
solve) path but ran the sequential per-device path instead (0 when built without control, or
when the continuation never batched: a non-polar formulation, or no sensitivity context)."""
function get_control_batched_fallback_count(data)
    isnothing(data.controlled_devices) && return 0
    return data.controlled_devices.batched_fallbacks[]
end

"""Number of times the last discrete-control continuation rebuilt its sensitivity context after
a bus-type change (a Q-limit PV→PQ flip in an inner solve)."""
function get_control_context_rebuild_count(data)
    isnothing(data.controlled_devices) && return 0
    return data.controlled_devices.context_rebuilds[]
end

"""Number of per-NR-iteration NUMERIC refactorizations performed inside the last
discrete-control continuation. 0 when the data was built without discrete control."""
function get_control_numeric_refactor_count(data)
    isnothing(data.controlled_devices) && return 0
    return data.controlled_devices.numeric_refactors[]
end

controlled_bus_ix(d::AbstractControlledDevice) = d.controlled_ix

# Results reporting reads the PERSISTED per-ts store, not the live device scalar (which
# reflects only the solve loop's last-processed time step once time_steps>1). Dispatch, not
# an `isa` branch, over the mixed device loop.
stored_current(set::ControlledDeviceSet, ::ControlledTap, i::Int, ts::Int) =
    set.tap_current[i, ts]
stored_current(set::ControlledDeviceSet, ::ControlledSwitchedShunt, i::Int, ts::Int) =
    set.shunt_current[i, ts]
stored_current(set::ControlledDeviceSet, ::ControlledFACTS, i::Int, ts::Int) =
    set.facts_current[i, ts]
stored_current(set::ControlledDeviceSet, ::ControlledPhaseShifter, i::Int, ts::Int) =
    set.pst_current[i, ts]

# Result columns meaningful only for FACTS; taps/shunts report the neutral value so the
# frame stays rectangular.
stored_delivered_q_mvar(
    ::ControlledDeviceSet,
    ::AbstractControlledDevice,
    ::Int,
    data,
    ::Int,
) =
    missing
stored_delivered_q_mvar(
    set::ControlledDeviceSet, d::ControlledFACTS, i::Int, data, ts::Int,
) = delivered_q_mvar(set.facts_current[i, ts], data.bus_magnitude[d.bus_ix, ts], d.base_mva)
stored_saturated(::ControlledDeviceSet, ::AbstractControlledDevice, ::Int, ::Int) = false
stored_saturated(set::ControlledDeviceSet, ::ControlledFACTS, i::Int, ts::Int) =
    set.facts_saturated[i, ts]

# FACTS's |b| bound is voltage-dependent and refreshed per-ts (`b_lim`); taps/shunts have a
# fixed parameter range, so their stored limits equal `parameter_limits(d)` regardless of ts.
stored_parameter_limits(::ControlledDeviceSet, d::AbstractControlledDevice, ::Int, ::Int) =
    parameter_limits(d)
stored_parameter_limits(set::ControlledDeviceSet, ::ControlledFACTS, i::Int, ts::Int) =
    (-set.facts_b_lim[i, ts], set.facts_b_lim[i, ts])

voltage_setpoint(d::AbstractControlledDevice) = d.vset

# The families with a PSS/E band and a discrete grid: taps ([VMI1, VMA1] in voltage), switched
# shunts ([VSWLO, VSWHI]) and phase shifters ([VMI1, VMA1] in MW). The latch, clamp, directional
# snap and band settle below are shared.
const BandedDevice = Union{ControlledTap, ControlledSwitchedShunt, ControlledPhaseShifter}
band_edges(d::BranchDevice) = (d.v_lo, d.v_hi)
band_edges(d::ControlledSwitchedShunt) = (d.vset_lo, d.vset_hi)

# A banded device regulates toward the edge it latched on; otherwise toward `vset`.
function control_setpoint(d::BandedDevice)
    d.band || return d.vset
    lo, hi = band_edges(d)
    d.side < 0 && return lo
    d.side > 0 && return hi
    return d.vset
end
parameter_limits(d::BranchDevice) = (d.p_min, d.p_max)
parameter_limits(d::ControlledSwitchedShunt) = (d.b_min, d.b_max)
parameter_limits(d::ControlledFACTS) = (-d.b_lim, d.b_lim)
current_parameter(d::AbstractControlledDevice) = d.current

# The continuation drives `measured_value(d, data, ts)` toward `control_setpoint(d)`. Every
# supported family regulates bus voltage, so these default to the controlled-bus magnitude and
# `vset` on the supertype — the only quantity-specific seam, overridable per family for a future
# non-voltage device; the rest of the continuation engine is agnostic to what is being regulated.
measured_value(d::AbstractControlledDevice, data, ts::Int) =
    data.bus_magnitude[controlled_bus_ix(d), ts]
control_setpoint(d::AbstractControlledDevice) = voltage_setpoint(d)

# Seam: future implicit embedding dispatches here. Never called by the outer loop.
stamp_control!(d::AbstractControlledDevice, args...) =
    error("implicit embedding not implemented for $(typeof(d))")

# PSS/E deadband semantics: a banded tap ([VMI1, VMA1]) or switched shunt ([VSWLO, VSWHI]) is
# held while its regulated voltage is inside its band, unless it is already regulating toward
# an edge (see `_update_band_state!`). A shunt without `band` (the midpoint rule) is held
# whenever its voltage is inside, including once it has crossed in. FACTS devices, and taps
# without a band, carry a point setpoint and always regulate.
_in_deadband(::AbstractControlledDevice, ::Float64) = false
function _in_deadband(d::ControlledSwitchedShunt, y::Float64)
    d.band && d.side != 0 && return false
    return d.vset_lo <= y <= d.vset_hi
end
_in_deadband(d::BranchDevice, y::Float64) = d.band && d.side == 0 && d.v_lo <= y <= d.v_hi

# Band latch, updated from the measured regulated voltage `y` before each damped move.
# Below the band ⇒ latch −1 (regulate up to the low edge), above ⇒ +1 (down to the high edge),
# recording the parameter as `anchor`. A latched device keeps regulating to its edge after
# crossing into the band, so the continuation settles at the edge and the snap can take the
# first step inside (PSS/E stops at the first tap position, or shunt step, whose voltage is in
# the band). It is released, and held from then on, when its voltage is inside the band while
# it still sits at its anchor: the band was reached without it (other devices moved the
# voltage), or `_band_clamp` stopped its retreat there.
# A latched device re-latches on the opposite side only from its anchor (GAP-068): when its own
# move carried its voltage across the whole band (the low-steepness stage's fixed point lies
# ln(pos/(1−pos))/S past the edge for a device far from mid-range), it keeps regulating toward its
# first edge, whose law brings it back as S grows, and the clamp stops it at its anchor; there, a
# voltage still on the far side is a real reversal and re-latches. So every anchor is a position
# the device actually held (its stored setting, or a level), never an off-grid continuous value.
# This applies to the devices that snap to a grid (`_band_capable`); a continuous banded shunt
# (MODSW 2, GAP-056's shared band) re-latches as before and settles on the far edge.
_update_band_state!(::AbstractControlledDevice, ::Float64) = nothing
function _update_band_state!(d::BandedDevice, y::Float64)
    d.band || return
    lo, hi = band_edges(d)
    away = abs(d.current - d.anchor) >= _param_tol(d)
    keep = away && _band_capable(d)
    if y < lo
        (d.side == 1 && keep) || d.side == -1 || (d.side = -1; d.anchor = d.current)
    elseif y > hi
        (d.side == -1 && keep) || d.side == 1 || (d.side = 1; d.anchor = d.current)
    elseif d.side != 0 && !away
        d.side = 0
    end
    return
end

# A banded device that snaps to a discrete grid (the band settle's devices).
_band_capable(d::ControlledTap) = d.band
_band_capable(d::ControlledPhaseShifter) = d.band && !isempty(d.levels)
_band_capable(d::ControlledSwitchedShunt) = d.band && !d.continuous

# A latched tap never retreats past its anchor: the edge it regulates to is approached from
# outside the band, so moving back beyond where it latched would leave the band on the other
# side of a position at which the voltage was already outside. `dvdp` orients the move.
_band_clamp(::AbstractControlledDevice, p::Float64, ::Float64) = p
function _band_clamp(d::BandedDevice, p::Float64, dvdp::Float64)
    (d.band && d.side != 0 && !iszero(dvdp)) || return p
    # Sign of the tap movement that moves the voltage from the violated side into the band.
    if -d.side * sign(dvdp) > 0
        return max(p, d.anchor)
    end
    return min(p, d.anchor)
end

# A frozen device holds its parameter; a frozen banded device is not regulating toward an edge.
_release_band!(::AbstractControlledDevice) = nothing
_release_band!(d::BandedDevice) = (d.side = 0; nothing)

function _nz_index(A::SparseArrays.SparseMatrixCSC, row::Int, col::Int)
    @inbounds for k in SparseArrays.nzrange(A, col)
        A.rowval[k] == row && return k
    end
    error("Ybus has no stored entry at ($row,$col); structural zero")
end

function _ybus_block_offsets(ybus, i::Int, j::Int)
    A = ybus.data
    return (
        _nz_index(A, i, i),
        _nz_index(A, i, j),
        _nz_index(A, j, i),
        _nz_index(A, j, j),
    )
end

# Y11/Y12/Y21 are delta-updated (new−old) so parallel branches' contributions
# in the shared nzval slots are preserved. Y22 = yt is tap-independent unless an impedance
# correction table re-evaluates at the new tap; its delta is exactly zero otherwise.
# Correctness of the running sum relies on `old_tap` coming from `d.current` (the last
# applied value), never read back from the nzval.
function apply_parameter!(d::BranchDevice, data, p::Float64, ::Int)
    A = data.power_network_matrix.data
    ff_old, ft_old, tf_old, tt_old = _branch_terms(d, d.current)
    ff_new, ft_new, tf_new, tt_new = _branch_terms(d, p)
    o = d.nz_offsets
    @inbounds begin
        A.nzval[o[1]] += ff_new - ff_old
        A.nzval[o[2]] += ft_new - ft_old
        A.nzval[o[3]] += tf_new - tf_old
        A.nzval[o[4]] += tt_new - tt_old
    end
    d.current = p
    return
end

# Delta-update (`+=`, not `=`): `_get_withdrawals!` accumulates all constant-Z devices on
# this bus into one slot, so overwriting would drop co-located contributions. Only
# susceptance is controlled; g0 is constant and stays in the baseline. Raising the
# (capacitive) susceptance lowers the bus's reactive withdrawal, injecting Q and raising the
# voltage. Shared by switched shunts and FACTS shunt compensators.
function apply_parameter!(d::AbstractShuntControl, data, b::Float64, ts::Int)
    data.bus_reactive_power_constant_impedance_withdrawals[d.bus_ix, ts] +=
        d.current - b
    d.current = b
    return
end

@inline function _sigmoid(lo::Float64, hi::Float64, S::Float64,
    x::Float64, xset::Float64)
    return (hi - lo) / (1.0 + exp(S * (x - xset))) + lo
end

# Nearest discrete tap, except for a banded tap regulating toward an edge: the continuation left
# it at the tap that puts its voltage on the edge, so the snap goes on in the direction it moved
# (away from `anchor`) to the first level at or beyond it, the first position inside the band.
# A latched tap still exactly at its anchor (the clamp held it there) goes on in the direction
# that corrects its voltage, −side·sign(g), given the continuation's gain `g` (GAP-068); without a
# gain it takes the nearest level, as before.
function snap_to_discrete(d::BranchDevice, p::Float64,
    g::Union{Nothing, Float64} = nothing)
    pc = clamp(p, d.p_min, d.p_max)
    isempty(d.levels) && return pc       # a continuous phase shifter (NTP1 < 2)
    dir = 0
    if d.band && d.side != 0
        dir = Int(sign(pc - d.anchor))
        if dir == 0 && !isnothing(g) && !iszero(g)
            dir = -d.side * Int(sign(g))
        end
    end
    if dir != 0
        return _directional_level(d, pc, dir)
    end
    best = d.levels[1]
    @inbounds for lv in d.levels
        abs(lv - pc) < abs(best - pc) && (best = lv)
    end
    return best
end

# The first level at or beyond `p` in direction `dir` (+1 up, −1 down), within a hair of `p`;
# the end level when none lies beyond (the parameter limits hold). `levels` ascend.
_directional_level(d::BandedDevice, p::Float64, dir::Int) =
    d.levels[_directional_level_index(d, p, dir)]
function _directional_level_index(d::BandedDevice, p::Float64, dir::Int)
    lo, hi = parameter_limits(d)
    eps = 1e-9 * (hi - lo)
    if dir > 0
        @inbounds for k in eachindex(d.levels)
            d.levels[k] >= p - eps && return k
        end
        return length(d.levels)
    end
    @inbounds for k in length(d.levels):-1:1
        d.levels[k] <= p + eps && return k
    end
    return 1
end

# The next level strictly beyond the device's current level in direction `dir`, or `nothing`
# at a limit.
function _next_level(d::BandedDevice, dir::Int)
    lo, hi = parameter_limits(d)
    eps = 1e-9 * (hi - lo)
    if dir > 0
        @inbounds for lv in d.levels
            lv > d.current + eps && return lv
        end
    else
        @inbounds for k in length(d.levels):-1:1
            d.levels[k] < d.current - eps && return d.levels[k]
        end
    end
    return nothing
end

# Whether a banded device's regulated voltage `y` is still outside its band on the side it
# latched.
function _outside_latched_side(d::BandedDevice, y::Float64)
    lo, hi = band_edges(d)
    d.side < 0 && return y < lo - CONTROL_TAP_BAND_TOL
    d.side > 0 && return y > hi + CONTROL_TAP_BAND_TOL
    return false
end

# Discrete shunt: a level of `levels` (`_shunt_levels`: the PSS/E ADJM 0 block chains, or every
# total for ADJM 1), recording its per-block step counts in `block_n`. A banded shunt regulating
# toward an edge goes on in the direction it moved (away from `anchor`) to the first level at or
# beyond its continuous value, the first level inside the band; otherwise the nearest level.
# Continuous (MODSW 2) shunts clamp. Allocation-free.
function snap_to_discrete(d::ControlledSwitchedShunt, b::Float64)
    d.continuous && return clamp(b, d.b_min, d.b_max)
    target = clamp(b, d.b_min, d.b_max)
    dir = 0
    if d.band && d.side != 0
        dir = Int(sign(target - d.anchor))
    end
    k = if dir != 0
        _directional_level_index(d, target, dir)
    else
        _nearest_level_index(d.levels, target)
    end
    _record_blocks!(d, k)
    return d.levels[k]
end

function _nearest_level_index(levels::Vector{Float64}, x::Float64)
    best = 1
    @inbounds for k in eachindex(levels)
        abs(levels[k] - x) < abs(levels[best] - x) && (best = k)
    end
    return best
end

@inline function _record_blocks!(d::ControlledSwitchedShunt, k::Int)
    @inbounds for j in eachindex(d.block_n)
        d.block_n[j] = d.level_blocks[k, j]
    end
    return
end

# After a band-settle step: the shunt sits on a level, so record that level's blocks.
_sync_blocks!(::AbstractControlledDevice) = nothing
function _sync_blocks!(d::ControlledSwitchedShunt)
    d.continuous && return
    eps = 1e-9 * (d.b_max - d.b_min)
    @inbounds for k in eachindex(d.levels)
        if abs(d.levels[k] - d.current) <= eps
            _record_blocks!(d, k)
            return
        end
    end
    return
end

# Effective |b| bound at the measured controlled-bus voltage (Q = b·V²). SVC is
# susceptance-bounded at `rating`, with an independent MVA ceiling `q_cap` that tightens the
# bound as |Q|=b·V² approaches q_cap. STATCOM is current-limited (|Q|<=V·I_max => b<=I_max/V),
# subject to the same MVA ceiling.
function _facts_b_limit(d::ControlledFACTS, vmag::Float64)
    v = max(vmag, 1e-3)
    if d.svc
        return min(d.rating, d.q_cap / v^2)
    else
        return min(d.rating / v, d.q_cap / v^2)
    end
end

# Continuous devices: no discrete grid, just clamp into the refreshed voltage-dependent band.
snap_to_discrete(d::ControlledFACTS, b::Float64) = clamp(b, -d.b_lim, d.b_lim)

# Delivered reactive power Q = b·|V|² in MVA, evaluated at the device's own (sending) bus
# voltage `vmag` — the susceptance sits there, not at the regulated bus (which may be remote).
# The 3-arg form takes an explicit susceptance so results-reporting can read a stored
# (possibly non-live, past-time-step) value without touching `d.current`.
delivered_q_mvar(b::Float64, vmag::Float64, base_mva::Float64) = b * vmag^2 * base_mva
delivered_q_mvar(d::ControlledFACTS, vmag::Float64) =
    delivered_q_mvar(d.current, vmag, d.base_mva)

# Post-solve classification of one FACTS endpoint against its final (voltage-dependent) bound
# and setpoint. `saturated` = pinned at the |b| bound while off setpoint — the device cannot
# supply the bus's reactive deficit (the homotopy analogue of a PV→PQ Q-limit release). A
# device that reaches setpoint with headroom, or is at its limit AND at setpoint (exactly
# enough), is not saturated. Mutates `d.saturated`; returns it. `@warn`s only on genuine
# saturation so it stays quiet on the common regulating case.
function classify_facts_saturation!(d::ControlledFACTS, vmag::Float64)
    at_limit = abs(d.current) >= (1.0 - CONTROL_FACTS_LIMIT_RTOL) * d.b_lim
    at_setpoint = abs(vmag - d.vset) <= CONTROL_FACTS_SETPOINT_BAND
    d.saturated = at_limit && !at_setpoint
    if d.saturated
        law = d.svc ? "SVC" : "STATCOM"
        @warn "ControlledFACTS \"$(d.name)\" saturated: |V−vset|=$(abs(vmag - d.vset)) at \
            its $law susceptance limit (|b|=$(abs(d.current)) ≥ b_lim=$(d.b_lim)); the bus \
            cannot be held to $(d.vset)."
    end
    return d.saturated
end

# The four series terms (ff, ft, tf, tt) of a tap device's 2×2 branch block at tap `p`, in its
# own arc orientation, without the primary shunt (which is tap-independent and sits in ff). tt
# (= yt) depends on the tap only through an impedance correction table.
@inline function _branch_terms(d::ControlledTap, p::Float64)
    t = p * cis(d.alpha)
    yt = _series_admittance(d, p)
    return (yt / abs2(t), -yt / conj(t), -yt / t, yt)
end

# The four series terms of a phase shifter's branch block at angle `α`: a tap's with the ratio
# fixed and the angle varying, `t = tap·e^{jα}` (PNM's stamp).
@inline function _branch_terms(d::ControlledPhaseShifter, α::Float64)
    t = d.tap * cis(α)
    yt = _series_admittance(d, α)
    return (yt / abs2(t), -yt / conj(t), -yt / t, yt)
end

# Active power entering the branch at terminal i, P_i = Re(V_i·conj(Y_ii·V_i + Y_ij·V_j)), and
# its partials with respect to (|V_i|, |V_j|, θ_i, θ_j) and, through (dY_ii, dY_ij), the angle.
@inline function _terminal_p_partials(Vi::ComplexF64, Vj::ComplexF64, Yii::ComplexF64,
    Yij::ComplexF64, dYii::ComplexF64, dYij::ComplexF64)
    vi, vj = abs(Vi), abs(Vj)
    A = conj(Yij) * Vi * conj(Vj)
    G = real(Yii)
    P = vi^2 * G + real(A)
    dvi = 2 * vi * G + (iszero(vi) ? 0.0 : real(A) / vi)
    dvj = iszero(vj) ? 0.0 : real(A) / vj
    dθi = -imag(A)
    dθj = imag(A)
    dα = vi^2 * real(dYii) + real(conj(dYij) * Vi * conj(Vj))
    return P, dvi, dvj, dθi, dθj, dα
end

# One unit's winding-1 → winding-2 active power at its metered end, and its partials: at winding 1
# P_f (the primary shunt included, as the reported flow), at winding 2 −P_t. Returns
# (P, ∂/∂|V_f|, ∂/∂|V_t|, ∂/∂θ_f, ∂/∂θ_t, ∂/∂α).
function _metered_flow_partials(d::ControlledPhaseShifter, data, ts::Int)
    f, t = d.from_ix, d.to_ix
    Vf = data.bus_magnitude[f, ts] * cis(data.bus_angles[f, ts])
    Vt = data.bus_magnitude[t, ts] * cis(data.bus_angles[t, ts])
    ff, ft, tf, tt = _branch_terms(d, d.current)
    dff, dft, dtf, dtt = _branch_term_slopes(d, d.current)
    if d.metered == 2
        P, dvt, dvf, dθt, dθf, dα = _terminal_p_partials(Vt, Vf, tt, tf, dtt, dtf)
        return -P, -dvf, -dvt, -dθf, -dθt, -dα
    end
    return _terminal_p_partials(Vf, Vt, ff + d.y_shunt, ft, dff, dft)
end

# ∂/∂α of the four branch terms (ff, ft, tf, tt) at α: with t = a·e^{jα}, ∂(1/conj(t))/∂α =
# j/conj(t) and ∂(1/t)/∂α = −j/t, so ∂ft = j·ft and ∂tf = −j·tf, plus yt's correction slope.
@inline function _branch_term_slopes(d::ControlledPhaseShifter, α::Float64)
    t = d.tap * cis(α)
    yt, dyt = _series_admittance_and_slope(d, α)
    ft = -yt / conj(t)
    tf = -yt / t
    return (dyt / abs2(t), im * ft - dyt / conj(t), -im * tf - dyt / t, dyt)
end

_metered_flow(d::ControlledPhaseShifter, data, ts::Int) =
    first(_metered_flow_partials(d, data, ts))

# A phase shifter regulates its metered MW flow; a member of a parallel group the group's mean.
function measured_value(d::ControlledPhaseShifter, data, ts::Int)
    isempty(d.group) && return _metered_flow(d, data, ts)
    return sum(m -> _metered_flow(m, data, ts), d.group) / length(d.group)
end

# Delta-update the from/to arc-admittance rows to `d.current` (parallel branches sharing the arc
# row are preserved; `d.synced` tracks the reflected value). Reported flows are computed from
# these matrices right after each time step, so an unsynced branch would report flows at its
# original tap.
function _sync_branch_arc_rows!(
    Yft::SparseArrays.SparseMatrixCSC,
    Ytf::SparseArrays.SparseMatrixCSC,
    d::BranchDevice,
    arc_row::Dict{Tuple{Int, Int}, Int},
    ix_to_number::Dict{Int, Int},
)
    d.current == d.synced && return
    fb = ix_to_number[d.from_ix]
    tb = ix_to_number[d.to_ix]
    ff_new, ft_new, tf_new, tt_new = _branch_terms(d, d.current)
    ff_old, ft_old, tf_old, tt_old = _branch_terms(d, d.synced)
    corrected = !isnothing(d.correction)
    r = get(arc_row, (fb, tb), 0)
    if !iszero(r)
        @inbounds begin
            Yft.nzval[_nz_index(Yft, r, d.from_ix)] += ff_new - ff_old
            Yft.nzval[_nz_index(Yft, r, d.to_ix)] += ft_new - ft_old
            Ytf.nzval[_nz_index(Ytf, r, d.from_ix)] += tf_new - tf_old
            # Ytf(r, to_ix) holds the tt term (= yt): tap-dependent only under correction.
            corrected && (Ytf.nzval[_nz_index(Ytf, r, d.to_ix)] += tt_new - tt_old)
        end
    else
        # The branch may be stored under the reversed arc orientation; the from/to roles
        # of the four terms swap accordingly (its "from side" is our to bus).
        r = get(arc_row, (tb, fb), 0)
        if iszero(r)
            @warn "discrete control: arc ($fb, $tb) of device \"$(d.name)\" not found \
                in the arc-admittance axes; reported flows on that branch reflect its \
                original parameter." maxlog = 1
            return
        end
        @inbounds begin
            Yft.nzval[_nz_index(Yft, r, d.from_ix)] += tf_new - tf_old
            Ytf.nzval[_nz_index(Ytf, r, d.to_ix)] += ft_new - ft_old
            Ytf.nzval[_nz_index(Ytf, r, d.from_ix)] += ff_new - ff_old
            # Yft(r, to_ix) is the reversed arc's from-side self term (= yt).
            corrected && (Yft.nzval[_nz_index(Yft, r, d.to_ix)] += tt_new - tt_old)
        end
    end
    d.synced = d.current
    return
end

"""One-shot post-continuation sync: bring the arc-admittance rows of every moved tap device
in line with its final parameter so the branch flows reported by `solve_power_flow!` match
the network the voltages were solved on. Shunt-side devices never touch the arc matrices.
No-op when the arc admittance matrices were not built."""
function _sync_arc_admittances!(data, set::ControlledDeviceSet)
    Yft = data.power_network_matrix.arc_admittance_from_to
    Ytf = data.power_network_matrix.arc_admittance_to_from
    (isnothing(Yft) || isnothing(Ytf)) && return
    isempty(set.taps) && isempty(set.phase_shifters) && return
    bus_lookup = PNM.get_bus_lookup(Yft)
    ix_to_number = Dict{Int, Int}(v => k for (k, v) in bus_lookup)
    arc_row = get_arc_lookup(data)   # (from_no, to_no) => arc row index
    for d in set.taps
        _sync_branch_arc_rows!(Yft.data, Ytf.data, d, arc_row, ix_to_number)
    end
    for d in set.phase_shifters
        _sync_branch_arc_rows!(Yft.data, Ytf.data, d, arc_row, ix_to_number)
    end
    return
end
