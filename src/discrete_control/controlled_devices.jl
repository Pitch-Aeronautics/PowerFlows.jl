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
start of every `_control_continuation!`. Without a band the tap is driven to `vset`."""
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
end

# Positional forms without the newer fields: no impedance correction table, point setpoint.
ControlledTap(
    name, from_ix, to_ix, controlled_ix, vset, yt, alpha, p_min, p_max, levels, nz_offsets,
    initial, synced, current, correction = nothing,
) = ControlledTap(
    name, from_ix, to_ix, controlled_ix, vset, yt, alpha, p_min, p_max, levels, nz_offsets,
    initial, synced, current, correction, false, vset, vset, 0, current,
)

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

"""Voltage-controlling switched shunt, snapped onto the PSS/E cumulative
block-activation chain (blocks switch on in listed order, off in reverse)."""
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
    # Perf counters for the last `_control_continuation!` (counts, not wall-clock; the regression
    # harness). symbolic_factors stays O(1) per continuation with PolarNRCache reuse. See getters.
    inner_solves::Base.RefValue{Int}
    symbolic_factors::Base.RefValue{Int}
    numeric_refactors::Base.RefValue{Int}
    # Passes that wanted the batched path but ran the sequential per-device path (a failed joint
    # solve or sensitivity refresh), and sensitivity-context rebuilds after a bus-type change.
    batched_fallbacks::Base.RefValue{Int}
    context_rebuilds::Base.RefValue{Int}
end
function ControlledDeviceSet(
    taps::Vector{ControlledTap},
    shunts::Vector{ControlledSwitchedShunt},
    facts::Vector{ControlledFACTS},
    n_time_steps::Int = 1,
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
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
    )
end
function Base.isempty(s::ControlledDeviceSet)
    return isempty(s.taps) && isempty(s.shunts) && isempty(s.facts)
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
# A banded tap regulates toward the edge it latched on; otherwise toward `vset`.
function control_setpoint(d::ControlledTap)
    d.band || return d.vset
    d.side < 0 && return d.v_lo
    d.side > 0 && return d.v_hi
    return d.vset
end
parameter_limits(d::ControlledTap) = (d.p_min, d.p_max)
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

# PSS/E deadband semantics: a switched shunt is held while the controlled voltage is
# anywhere INSIDE [VSWLO, VSWHI]; only excursions outside the band trigger switching.
# A banded tap is held inside [VMI1, VMA1] unless it is already regulating toward an edge
# (see `_update_band_state!`). FACTS devices carry a point setpoint and always regulate.
_in_deadband(::AbstractControlledDevice, ::Float64) = false
_in_deadband(d::ControlledSwitchedShunt, y::Float64) = d.vset_lo <= y <= d.vset_hi
_in_deadband(d::ControlledTap, y::Float64) = d.band && d.side == 0 && d.v_lo <= y <= d.v_hi

# Banded-tap latch, updated from the measured regulated voltage `y` before each damped move.
# Below the band ⇒ latch −1 (regulate up to VMI1), above ⇒ +1 (down to VMA1), recording the tap
# as `anchor`. A latched tap keeps regulating to its edge after crossing into the band, so the
# continuation settles at the edge and the snap can take the first step inside (PSS/E stops at
# the first tap position whose voltage is in the band). It is released, and held from then on,
# when its voltage is inside the band while it still sits at its anchor: the band was reached
# without it (other devices moved the voltage), or `_band_clamp` stopped its retreat there.
_update_band_state!(::AbstractControlledDevice, ::Float64) = nothing
function _update_band_state!(d::ControlledTap, y::Float64)
    d.band || return
    if y < d.v_lo
        d.side == -1 || (d.side = -1; d.anchor = d.current)
    elseif y > d.v_hi
        d.side == 1 || (d.side = 1; d.anchor = d.current)
    elseif d.side != 0 && abs(d.current - d.anchor) < _param_tol(d)
        d.side = 0
    end
    return
end

# A latched tap never retreats past its anchor: the edge it regulates to is approached from
# outside the band, so moving back beyond where it latched would leave the band on the other
# side of a position at which the voltage was already outside. `dvdp` orients the move.
_band_clamp(::AbstractControlledDevice, p::Float64, ::Float64) = p
function _band_clamp(d::ControlledTap, p::Float64, dvdp::Float64)
    (d.band && d.side != 0 && !iszero(dvdp)) || return p
    # Sign of the tap movement that moves the voltage from the violated side into the band.
    if -d.side * sign(dvdp) > 0
        return max(p, d.anchor)
    end
    return min(p, d.anchor)
end

# A frozen device holds its parameter; a frozen tap is not regulating toward an edge.
_release_band!(::AbstractControlledDevice) = nothing
_release_band!(d::ControlledTap) = (d.side = 0; nothing)

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
function apply_parameter!(d::ControlledTap, data, p::Float64, ::Int)
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
function snap_to_discrete(d::ControlledTap, p::Float64)
    pc = clamp(p, d.p_min, d.p_max)
    dir = 0
    if d.band && d.side != 0
        dir = Int(sign(pc - d.anchor))
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
# the end level when none lies beyond (the tap limits hold).
function _directional_level(d::ControlledTap, p::Float64, dir::Int)
    eps = 1e-9 * (d.p_max - d.p_min)
    if dir > 0
        @inbounds for lv in d.levels
            lv >= p - eps && return lv
        end
        return d.levels[end]
    end
    @inbounds for k in length(d.levels):-1:1
        d.levels[k] <= p + eps && return d.levels[k]
    end
    return d.levels[1]
end

# The next level strictly beyond the tap's current level in direction `dir`, or `nothing` at a
# tap limit.
function _next_level(d::ControlledTap, dir::Int)
    eps = 1e-9 * (d.p_max - d.p_min)
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

# Whether a banded tap's regulated voltage `y` is still outside its band on the side it latched.
function _outside_latched_side(d::ControlledTap, y::Float64)
    d.side < 0 && return y < d.v_lo - CONTROL_TAP_BAND_TOL
    d.side > 0 && return y > d.v_hi + CONTROL_TAP_BAND_TOL
    return false
end

# PSS/E mixed banks: capacitor blocks (dB>0) switch on cumulatively in listed order,
# reactor blocks (dB<0) likewise — two independent chains stepping away from the
# all-off base, NOT one serial chain, so a mixed bank reaches both signs. Realizable
# totals = b0 ∪ {b0 + capacitor prefixes} ∪ {b0 + reactor prefixes}. Same-sign banks
# reduce to the previous single-chain walk. O(Σ steps), allocation-free.
function snap_to_discrete(d::ControlledSwitchedShunt, b::Float64)
    d.continuous && return clamp(b, d.b_min, d.b_max)
    target = clamp(b, d.b_min, d.b_max)
    best = d.b0
    best_steps = 0
    best_positive = true
    @inbounds for positive in (true, false)
        total = d.b0
        steps_taken = 0
        for k in eachindex(d.block_steps, d.block_dB)
            dB = d.block_dB[k]
            on_side = if positive
                dB > 0.0
            else
                dB < 0.0
            end
            on_side || continue
            for _ in 1:d.block_steps[k]
                total += dB
                steps_taken += 1
                if abs(total - target) < abs(best - target)
                    best = total
                    best_steps = steps_taken
                    best_positive = positive
                end
            end
        end
    end
    # Record the winning side's prefix in block_n; the other side's blocks are 0.
    fill!(d.block_n, 0)
    remaining = best_steps
    @inbounds for k in eachindex(d.block_steps)
        dB = d.block_dB[k]
        on_winning_side = if best_positive
            dB > 0.0
        else
            dB < 0.0
        end
        if on_winning_side
            n = min(remaining, d.block_steps[k])
            d.block_n[k] = n
            remaining -= n
        end
    end
    return best
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

# Delta-update the from/to arc-admittance rows to `d.current` (parallel branches sharing the arc
# row are preserved; `d.synced` tracks the reflected value). Reported flows are computed from
# these matrices right after each time step, so an unsynced branch would report flows at its
# original tap.
function _sync_branch_arc_rows!(
    Yft::SparseArrays.SparseMatrixCSC,
    Ytf::SparseArrays.SparseMatrixCSC,
    d::ControlledTap,
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
    isempty(set.taps) && return
    bus_lookup = PNM.get_bus_lookup(Yft)
    ix_to_number = Dict{Int, Int}(v => k for (k, v) in bus_lookup)
    arc_row = get_arc_lookup(data)   # (from_no, to_no) => arc row index
    for d in set.taps
        _sync_branch_arc_rows!(Yft.data, Ytf.data, d, arc_row, ix_to_number)
    end
    return
end
