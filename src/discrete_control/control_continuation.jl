# Target slope of the local iteration map near the fixed point (0<m<1 ⇒ monotone,
# non-oscillatory). 0.5 trades settling speed for a 2× margin on the worst-case gain bound.
# It also bounds the relaxation factor itself: ω = (1−θ)/(1+gbound) ≤ 1−θ = 0.5.
const CONTROL_CONTRACTION = 0.5

# State a rolled-back trial must restore. `bus_type` is here because the inner solve's Q-limit
# logic switches buses PV→PQ and back (`_check_q_limit_bounds!`). An absent VSC or LCC
# subsystem is carried as empty vectors, which is what the capture/restore guards test.
struct ControlStateSnapshot
    vmag::Vector{Float64}
    vang::Vector{Float64}
    btype::Vector{PSY.ACBusTypes}
    pinj::Vector{Float64}
    qinj::Vector{Float64}
    dc_p::Vector{Float64}
    dc_q::Vector{Float64}
    dc_v::Vector{Float64}
    lcc_rectifier_tap::Vector{Float64}
    lcc_inverter_tap::Vector{Float64}
    lcc_rectifier_angle::Vector{Float64}
    lcc_inverter_angle::Vector{Float64}
    lcc_i_dc::Vector{Float64}
    # Remote-regulation members' Q-limit status (GAP-015): the inner solve's Q-limit check fixes
    # and frees them like the PV↔PQ switches above. Empty without remote-regulation groups.
    rv_status::Vector{Int8}
end

@inline function _dc_state_cols(data, ts::Int)
    dcn = get_dc_network(data)
    if has_dc_network(dcn)
        return dcn.p_c[:, ts], dcn.q_c[:, ts], dcn.node_vdc[:, ts]
    end
    return Float64[], Float64[], Float64[]
end
@inline function _lcc_state_cols(data, ts::Int)
    if get_lcc_count(data) > 0
        lcc = data.lcc
        return lcc.rectifier.tap[:, ts], lcc.inverter.tap[:, ts],
        lcc.rectifier.thyristor_angle[:, ts], lcc.inverter.thyristor_angle[:, ts],
        lcc.i_dc[:, ts]
    end
    return Float64[], Float64[], Float64[], Float64[], Float64[]
end
@inline function _snapshot_state(data, ts::Int)
    dc_p, dc_q, dc_v = _dc_state_cols(data, ts)
    lcc_rt, lcc_it, lcc_ra, lcc_ia, lcc_idc = _lcc_state_cols(data, ts)
    return ControlStateSnapshot(
        data.bus_magnitude[:, ts],
        data.bus_angles[:, ts],
        data.bus_type[:, ts],
        data.bus_active_power_injections[:, ts],
        data.bus_reactive_power_injections[:, ts],
        dc_p, dc_q, dc_v,
        lcc_rt, lcc_it, lcc_ra, lcc_ia, lcc_idc,
        _rv_status_column(data.remote_voltage_control, ts),
    )
end
@inline function _capture_state!(snap::ControlStateSnapshot, data, ts::Int)
    snap.vmag .= view(data.bus_magnitude, :, ts)
    snap.vang .= view(data.bus_angles, :, ts)
    snap.btype .= view(data.bus_type, :, ts)
    snap.pinj .= view(data.bus_active_power_injections, :, ts)
    snap.qinj .= view(data.bus_reactive_power_injections, :, ts)
    if !isempty(snap.dc_v)
        dcn = get_dc_network(data)
        snap.dc_p .= view(dcn.p_c, :, ts)
        snap.dc_q .= view(dcn.q_c, :, ts)
        snap.dc_v .= view(dcn.node_vdc, :, ts)
    end
    if !isempty(snap.lcc_i_dc)
        lcc = data.lcc
        snap.lcc_rectifier_tap .= view(lcc.rectifier.tap, :, ts)
        snap.lcc_inverter_tap .= view(lcc.inverter.tap, :, ts)
        snap.lcc_rectifier_angle .= view(lcc.rectifier.thyristor_angle, :, ts)
        snap.lcc_inverter_angle .= view(lcc.inverter.thyristor_angle, :, ts)
        snap.lcc_i_dc .= view(lcc.i_dc, :, ts)
    end
    isempty(snap.rv_status) ||
        (snap.rv_status .= view(data.remote_voltage_control.status, :, ts))
    return
end
@inline function _restore_state!(data, ts::Int, snap::ControlStateSnapshot)
    data.bus_magnitude[:, ts] .= snap.vmag
    data.bus_angles[:, ts] .= snap.vang
    data.bus_type[:, ts] .= snap.btype
    data.bus_active_power_injections[:, ts] .= snap.pinj
    data.bus_reactive_power_injections[:, ts] .= snap.qinj
    if !isempty(snap.dc_v)
        dcn = get_dc_network(data)
        dcn.p_c[:, ts] .= snap.dc_p
        dcn.q_c[:, ts] .= snap.dc_q
        dcn.node_vdc[:, ts] .= snap.dc_v
    end
    if !isempty(snap.lcc_i_dc)
        lcc = data.lcc
        lcc.rectifier.tap[:, ts] .= snap.lcc_rectifier_tap
        lcc.inverter.tap[:, ts] .= snap.lcc_inverter_tap
        lcc.rectifier.thyristor_angle[:, ts] .= snap.lcc_rectifier_angle
        lcc.inverter.thyristor_angle[:, ts] .= snap.lcc_inverter_angle
        lcc.i_dc[:, ts] .= snap.lcc_i_dc
        # Re-derive phi/branch_admittances at the restored state.
        _update_ybus_lcc!(data, ts)
    end
    isempty(snap.rv_status) ||
        (view(data.remote_voltage_control.status, :, ts) .= snap.rv_status)
    return
end

# Counting wrapper around the inner solve: iteration counts are the repo's robust
# performance metric, and the continuation's cost IS its inner-solve count. Past the control
# solve's time limit (`control_time_limit`), or once it has started as many inner solves as its
# limit (`control_solve_limit`) allows, no inner solve starts: it reports a failed solve, which
# every caller already rolls back, and the loops below stop at their next check.
@inline function _ctrl_solve!(pf, data, ts::Int; kwargs...)
    cd = data.controlled_devices
    if !isnothing(cd)
        _control_time_up!(cd) && return false
        cd.inner_solves[] += 1
        cd.solves_started[] += 1
    end
    return _solve_with_q_limits!(pf, data, ts; kwargs...)
end

# Sign-corrected sigmoid law: orientation comes from the MEASURED dy/dp (not the
# device's primary/secondary wiring) so the closed-loop gain σ'(y)·dy/dp ≤ 0
# (negative feedback) regardless of wiring. `_sigmoid(lo,hi,…)` decreases in y when hi>lo.
# `y` is the regulated quantity — bus voltage for every currently supported device family.
#
# The sigmoid maps the regulated quantity to an absolute parameter in [lo, hi], so two devices
# with the same limits are driven to the same parameter. A unit of a parallel tap group maps
# into its window shifted by its offset from the group (`_sigmoid_offset`), so the units move by
# the same amount and keep their ratio offset (clamped to their own limits).
@inline function _control_target(d, y::Float64, S::Float64, dydp::Float64)
    lo, hi = parameter_limits(d)
    w = _sigmoid_offset(d)
    return if dydp > 0.0
        clamp(_sigmoid(lo + w, hi + w, S, y, control_setpoint(d)), lo, hi)
    else
        clamp(_sigmoid(hi + w, lo + w, S, y, control_setpoint(d)), lo, hi)
    end
end
_sigmoid_offset(::AbstractControlledDevice) = 0.0
_sigmoid_offset(d::BranchDevice) = d.offset

# Scale-aware settle tolerance: 1e-5 absolute is 5e-5 relative for a 0.2-wide tap band
# but only 1e-6 relative for a 10 p.u. shunt; a relative floor lets wide-range devices
# settle in comparable pass counts. Sub-grid precision is wasted anyway — final values
# snap to discrete grids.
@inline function _param_tol(d)
    lo, hi = parameter_limits(d)
    return max(CONTROL_PARAM_TOL, CONTROL_PARAM_RTOL * (hi - lo))
end

# Measure dy/dp (sensitivity of the regulated quantity to the parameter) by a small
# parameter perturbation. The probe is definitionally a temporary excursion: the full
# pre-probe state (incl. any Q-limit flips the probe solve caused) is restored afterward,
# which also makes the second re-converging solve unnecessary. `reliable=false` (the
# probe solve failed) ⇒ orientation unknown, so the caller freezes the device rather
# than stepping it with an unknown sign.
function _plant_sign(
    d, data, ts::Int, pf, snap::ControlStateSnapshot; kwargs...,
)::Tuple{Float64, Bool}
    p0 = current_parameter(d)
    y0 = measured_value(d, data, ts)
    _capture_state!(snap, data, ts)
    lo, hi = parameter_limits(d)
    δ = 1e-3 * (hi - lo)
    if δ <= 0.0
        δ = 1e-6
    end
    p1 = clamp(p0 + δ, lo, hi)
    p1 == p0 && (p1 = clamp(p0 - δ, lo, hi))
    h = p1 - p0
    apply_parameter!(d, data, p1, ts)
    ok = _ctrl_solve!(pf, data, ts; kwargs...)
    y1 = measured_value(d, data, ts)
    apply_parameter!(d, data, p0, ts)
    _restore_state!(data, ts, snap)
    reliable = ok && !iszero(h)
    dVdp = 0.0
    if reliable
        dVdp = (y1 - y0) / h
    end
    return dVdp, reliable
end

# ── Linearized plant sensitivities ───────────────────────────────────────────────────────
# Differentiating F(x,p)=0 at the converged base: dx/dp = −J⁻¹·(∂F/∂p), and dy/dp is the
# controlled-bus voltage component — no perturbation, one triangular solve per device on the base
# NR solve's reused factorization instead of a full nonlinear solve. Polar + voltage-device (tap/shunt/FACTS)
# only (state layout x[2b−1]=Vm, x[2b]=Va and ∂F/∂p are polar-specific); rect/mixed formulations
# fall back to the FD `_plant_sign`. Signs are validated against the FD probe in the tests.

# Sensitivity context: residual+Jacobian built at the CURRENT converged base state and numerically
# factored (reusing the base NR solve's persisted symbolic factorization). Built ONCE per
# continuation (probe phase) and thereafter kept current by `_refresh_sensitivity_context!`
# (values-only) after each batched-pass joint solve, or rebuilt in place when that solve changed
# the bus types. `nothing` ⇒ the linear path is unavailable (non-polar formulation, or the base
# Jacobian is singular) and the caller uses FD probes. Mutable so a rebuild can replace its
# objects while the continuation keeps holding the same context.
mutable struct _SensitivityContext{C, R, JT}
    lin_cache::C
    residual::R             # persisted; re-evaluated in place by _refresh_sensitivity_context!
    J::JT                   # persisted; re-evaluated in place by _refresh_sensitivity_context!
    rhs::Vector{Float64}    # ∂F/∂p scratch, refilled per device
    sol::Vector{Float64}    # J⁻¹·∂F/∂p scratch
    # Snapshot of bus_type at build time. `ACPowerFlowResidual`'s subnetworks/slack-participation
    # factors/validate_indices are computed ONCE at construction from bus_type and are NOT
    # recomputed by the residual functor — a Q-limit PV→PQ flip (or PQ→PV return) elsewhere in
    # the network after this ctx was built silently stales that structure.
    # `_refresh_sensitivity_context!` checks this snapshot and rebuilds a ctx whose bus-type
    # pattern has since changed.
    bus_type::Vector{PSY.ACBusTypes}
end

_sensitivity_context(::AbstractACPowerFlow, data, ts::Int; kwargs...) = nothing
function _sensitivity_context(pf::ACPolarPowerFlow, data, ts::Int; kwargs...)
    ctx = _build_sensitivity_context(data, ts; kwargs...)
    # One full sensitivity-context build (fresh residual + Jacobian structure + refactor) per
    # continuation. `_refresh_sensitivity_context!` re-evaluates VALUES into these same objects
    # on every subsequent batched pass, or rebuilds them after a bus-type change, and does NOT
    # count here: the Jacobian sparsity (and thus the persisted symbolic factorization this
    # counter tracks) is invariant under the PV↔PQ switches that cause a rebuild, so only this one
    # build should ever register.
    isnothing(ctx) || _count_symbolic_factor!(data)
    return ctx
end

function _build_sensitivity_context(data, ts::Int; kwargs...)
    backend = resolve_linear_solver_backend(get(kwargs, :linear_solver, nothing))
    residual = ACPowerFlowResidual(data, ts)
    x = calculate_x0(data, ts)
    residual(x, ts)                       # evaluate at current state; fills P_net/Q_net
    J = ACPowerFlowJacobian(residual, ts)
    J(ts)                                 # Jacobian values at current state
    lin_cache =
        _nr_linear_solver_cache!(
            data, J, backend, residual.bus_slack_participation_factors, ts)
    try
        numeric_refactor!(lin_cache, J.Jv)
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        return                    # singular base Jacobian ⇒ fall back to FD probes
    end
    n = length(residual.Rv)
    return _SensitivityContext(
        lin_cache, residual, J, zeros(n), zeros(n), copy(view(data.bus_type, :, ts)))
end

# Rebuild `ctx` in place at the current converged state (after a bus-type change); `false` when
# the rebuilt Jacobian is singular, leaving `ctx` as it was.
function _rebuild_sensitivity_context!(
    ctx::_SensitivityContext, data, ts::Int; kwargs...,
)::Bool
    fresh = _build_sensitivity_context(data, ts; kwargs...)
    isnothing(fresh) && return false
    ctx.lin_cache = fresh.lin_cache
    ctx.residual = fresh.residual
    ctx.J = fresh.J
    ctx.rhs = fresh.rhs
    ctx.sol = fresh.sol
    ctx.bus_type = fresh.bus_type
    cd = data.controlled_devices
    isnothing(cd) || (cd.context_rebuilds[] += 1)
    return true
end

# Values-only refresh at a new converged base state: the Jacobian sparsity depends only on
# topology/REF layout (invariant across the continuation), so re-evaluating values + numeric
# refactor into the persisted objects replaces a full rebuild — UNLESS a PV↔PQ flip has changed
# the slack-participation/subnetwork layout the persisted `residual` baked in at construction
# (see `_SensitivityContext.bus_type`). Then the context is rebuilt in place from the current
# state (`_rebuild_sensitivity_context!`: a fresh residual and Jacobian, the shared symbolic
# factorization reused and the pivots refreshed for the new types). Only a singular refactor
# or rebuild returns `false`, sending the caller to the sequential per-device path. Before, a
# flip returned `false` here, and since the inner solves' Q-limit flips were then one-way every
# later pass of the continuation ran the per-device path: one full solve per device per pass. A
# PQ→PV return (a released unit whose voltage crossed back past its setpoint) changes the types
# the same way and is rebuilt the same way.
#
# `P_net`/`Q_net`/`P_net_set` and the four constant-I/Z withdrawal vectors are all captured from
# `data` ONCE at `ACPowerFlowResidual` construction and are NOT independently re-read by the
# residual functor: `_update_residual_values!`'s PQ-bus case does `P_net[ix] += ...` (a
# TELESCOPING correction from the bus's voltage at the residual's LAST evaluation to its new
# value), which is only correct when the SAME residual object is the one driving every
# evaluation of a single NR solve. Here `ctx.residual` is evaluated once per PASS while `data`'s
# voltage is moved BETWEEN passes by the joint solve's OWN, unrelated residual object — so
# `P_net`/`Q_net` must be rebuilt from `data` (exactly like the constructor) before every
# evaluation, or they silently accumulate a wrong, ever-growing correction. Likewise
# `apply_parameter!` for switched shunts/FACTS writes device moves directly into
# `data.bus_reactive_power_constant_impedance_withdrawals`, so the four constant-I/Z vectors need
# the same re-sync.
function _refresh_sensitivity_context!(
    ctx::_SensitivityContext, data, ts::Int; kwargs...,
)::Bool
    view(data.bus_type, :, ts) == ctx.bus_type ||
        return _rebuild_sensitivity_context!(ctx, data, ts; kwargs...)
    residual = ctx.residual
    copyto!(
        residual.bus_active_constant_I,
        view(data.bus_active_power_constant_current_withdrawals, :, ts),
    )
    copyto!(
        residual.bus_reactive_constant_I,
        view(data.bus_reactive_power_constant_current_withdrawals, :, ts),
    )
    copyto!(
        residual.bus_active_constant_Z,
        view(data.bus_active_power_constant_impedance_withdrawals, :, ts),
    )
    copyto!(
        residual.bus_reactive_constant_Z,
        view(data.bus_reactive_power_constant_impedance_withdrawals, :, ts),
    )
    @inbounds for ix in eachindex(residual.P_net)
        residual.P_net[ix] =
            data.bus_active_power_injections[ix, ts] -
            get_bus_active_power_total_withdrawals(data, ix, ts) +
            data.bus_hvdc_net_power[ix, ts]
        residual.Q_net[ix] =
            data.bus_reactive_power_injections[ix, ts] -
            get_bus_reactive_power_total_withdrawals(data, ix, ts)
        residual.P_net_set[ix] = residual.P_net[ix]
    end
    x = calculate_x0(data, ts)
    ctx.residual(x, ts)
    ctx.J(ts)
    try
        numeric_refactor!(ctx.lin_cache, ctx.J.Jv)
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        return false
    end
    return true
end

# ∂F/∂p into `rhs` (zeroed first). Returns `true` iff the family has an analytic polar form here;
# the `false` path is reserved for a future family without one — the caller then uses the FD probe
# (see `_linear_plant_sign`). Row convention: F[2b−1] active, F[2b] reactive balance at bus b
# (see `_update_residual_values!`).
function _dF_dp!(rhs::Vector{Float64}, d::ControlledTap, data, ts::Int)
    fill!(rhs, 0.0)
    f, t = d.from_ix, d.to_ix
    Vf = data.bus_magnitude[f, ts] * cis(data.bus_angles[f, ts])
    Vt = data.bus_magnitude[t, ts] * cis(data.bus_angles[t, ts])
    p, a = d.current, d.alpha
    # ∂Y/∂p of the branch terms (t_c = p·cis(a); see `_branch_terms`): Y_ff = yt/p²,
    # Y_ft = −yt·cis(a)/p, Y_tf = −yt·cis(−a)/p, Y_tt = yt, where yt = yt(p) varies with p only
    # through an impedance correction table (dyt = 0 without one).
    yt, dyt = _series_admittance_and_slope(d, p)
    dYff = dyt / p^2 - 2.0 * yt / p^3
    dYft = (yt / p^2 - dyt / p) * cis(a)
    dYtf = (yt / p^2 - dyt / p) * cis(-a)
    dYtt = dyt
    # ∂S_i/∂p = V_i·conj(Σ_k ∂Y_ik/∂p·V_k); Y_ff,Y_ft (row f) and Y_tf,Y_tt (row t) change.
    dSf = Vf * conj(dYff * Vf + dYft * Vt)
    dSt = Vt * conj(dYtf * Vf + dYtt * Vt)
    @inbounds begin
        rhs[2 * f - 1] = real(dSf)
        rhs[2 * f] = imag(dSf)
        rhs[2 * t - 1] = real(dSt)
        rhs[2 * t] = imag(dSt)
    end
    return true
end

function _dF_dp!(
    rhs::Vector{Float64},
    d::Union{ControlledSwitchedShunt, ControlledFACTS},
    data,
    ts::Int,
)
    fill!(rhs, 0.0)
    b = d.bus_ix
    Vm = data.bus_magnitude[b, ts]
    # Constant-Z reactive withdrawal w enters Q_net as −w·Vm²; apply_parameter! sets ∂w/∂susc = −1,
    # so ∂Q_net/∂susc = +Vm² and ∂F[2b]/∂susc = −∂Q_net/∂susc = −Vm² (reactive row only).
    @inbounds rhs[2 * b] = -Vm^2
    return true
end

# Phase shifter: ∂F/∂α from the angle derivative of its branch block (`_branch_term_slopes`).
function _dF_dp!(rhs::Vector{Float64}, d::ControlledPhaseShifter, data, ts::Int)
    fill!(rhs, 0.0)
    f, t = d.from_ix, d.to_ix
    Vf = data.bus_magnitude[f, ts] * cis(data.bus_angles[f, ts])
    Vt = data.bus_magnitude[t, ts] * cis(data.bus_angles[t, ts])
    dYff, dYft, dYtf, dYtt = _branch_term_slopes(d, d.current)
    dSf = Vf * conj(dYff * Vf + dYft * Vt)
    dSt = Vt * conj(dYtf * Vf + dYtt * Vt)
    @inbounds begin
        rhs[2 * f - 1] = real(dSf)
        rhs[2 * f] = imag(dSf)
        rhs[2 * t - 1] = real(dSt)
        rhs[2 * t] = imag(dSt)
    end
    return true
end

# dx_b/dp entries of the state a branch flow depends on: |V_b| is a state only at a PQ bus and
# θ_b at a PQ or PV bus (x[2b−1], x[2b]; see `update_state!`); elsewhere it is held, so 0.
@inline function _state_slope(data, ts::Int, sol::Vector{Float64}, b::Int)
    bt = data.bus_type[b, ts]
    dv = bt == PSY.ACBusTypes.PQ ? -sol[2 * b - 1] : 0.0
    dθ = bt == PSY.ACBusTypes.REF ? 0.0 : -sol[2 * b]
    return dv, dθ
end

# Linearized dP/dα of a phase shifter's regulated flow (its own, or its parallel group's mean):
# dP/dα = ∂P/∂α + ∇ₓP·dx/dα with dx/dα = −J⁻¹·∂F/∂α, one triangular solve on the sensitivity
# context's factorization; the partials come from each unit's terminal flow.
function _linear_plant_sign(d::ControlledPhaseShifter, data, ts::Int,
    ctx::_SensitivityContext)
    _dF_dp!(ctx.rhs, d, data, ts)
    copyto!(ctx.sol, ctx.rhs)
    solve!(ctx.lin_cache, ctx.sol)
    members = isempty(d.group) ? (d,) : d.group
    g = 0.0
    for m in members
        _, dvf, dvt, dθf, dθt, dα = _metered_flow_partials(m, data, ts)
        xvf, xθf = _state_slope(data, ts, ctx.sol, m.from_ix)
        xvt, xθt = _state_slope(data, ts, ctx.sol, m.to_ix)
        g += dvf * xvf + dvt * xvt + dθf * xθf + dθt * xθt + (m === d ? dα : 0.0)
    end
    return g / length(members), true
end

# Linearized dy/dp for a voltage device via the factored Jacobian. `y = Vm(controlled_bus)`
# = x[2·cbus−1], and dx/dp = −J⁻¹·(∂F/∂p). Returns `(dy/dp, true)`, or `(0.0, false)` when the
# family has no analytic form (caller then uses the FD probe).
function _linear_plant_sign(d, data, ts::Int, ctx::_SensitivityContext)
    _dF_dp!(ctx.rhs, d, data, ts) || return 0.0, false
    cbus = controlled_bus_ix(d)
    # x[2b−1] is Vm ONLY at PQ buses (Q_gen at PV, P at REF — see update_state!). At a
    # PV/REF controlled bus the voltage is pinned by the bus model, so dVm/dp = 0 exactly:
    # return a reliable zero so the caller's gain floor freezes the device, matching the
    # FD probe's behavior.
    if data.bus_type[cbus, ts] != PSY.ACBusTypes.PQ
        return 0.0, true
    end
    copyto!(ctx.sol, ctx.rhs)
    solve!(ctx.lin_cache, ctx.sol)        # sol = J⁻¹·(∂F/∂p)
    return -ctx.sol[2 * cbus - 1], true       # dy/dp = (dx/dp)[Vm(cbus)] = −sol[2·cbus−1]
end

# Shared bisection sub-step walk for the robust continuation applicators
# (`_continuation_to!` and `_restore_one!`). The full move having failed, step from `start`
# toward `target` (each interpolated point clamped to `[lo, hi]`), growing the step on NR
# success and halving it on failure; give up below `MIN_LAMBDA_STEP`. On entry `snap` holds
# the converged state at `start` and the device/data sit at `start`. Leaves the system at the
# last converged iterate and returns `(reached, completed)`: the parameter reached (== `start`
# if nothing budged) and whether the full `[start, target]` interval was traversed.
function _bisection_walk!(d, data, ts::Int, start::Float64, target::Float64,
    lo::Float64, hi::Float64, pf, snap::ControlStateSnapshot; kwargs...)
    done = 0.0                       # fraction of [start, target] applied so far
    step = 0.5
    reached = start
    while done < 1.0
        trial = min(1.0, done + step)
        p = clamp(start + trial * (target - start), lo, hi)
        apply_parameter!(d, data, p, ts)
        if _ctrl_solve!(pf, data, ts; kwargs...)
            done = trial
            reached = p
            _capture_state!(snap, data, ts)
            step = min(step * CONTROL_STEP_GROWTH, MAX_LAMBDA_STEP)
        else
            # Revert to the last converged parameter and state, not the failed trial.
            apply_parameter!(d, data, reached, ts)
            _restore_state!(data, ts, snap)
            step /= 2.0
            step < MIN_LAMBDA_STEP && return reached, false
        end
    end
    return reached, true
end

# Incremental robust applicator: walk the parameter from `start = d.current`
# toward `target` so NR stays converged.  The full move is tried FIRST (one
# inner solve in the common case); only if it fails does the walk fall back to
# bisection sub-stepping (via `_bisection_walk!`).  Returns `(reached, moved)`:
# the parameter actually reached (solver left converged there) and whether ANY
# sub-step was applied — a requested move that could not budge at all must not
# masquerade as a settled device.
function _continuation_to!(
    d, data, ts::Int, target::Float64, pf, snap::ControlStateSnapshot; kwargs...,
)
    start = current_parameter(d)
    abs(target - start) < _param_tol(d) && return start, true
    lo, hi = parameter_limits(d)
    _capture_state!(snap, data, ts)  # last converged state, restored on a failed trial
    clamped = clamp(target, lo, hi)  # no-op here (damped target is pre-clamped)
    # Full move first: the damped target is usually within the warm-started solver's
    # reach, so the common case costs ONE inner solve instead of a multi-sub-step walk.
    apply_parameter!(d, data, clamped, ts)
    _ctrl_solve!(pf, data, ts; kwargs...) && return clamped, true
    apply_parameter!(d, data, start, ts)
    _restore_state!(data, ts, snap)
    # Bisection fallback: the full step failed, so retry from half the interval.
    reached, completed =
        _bisection_walk!(d, data, ts, start, target, lo, hi, pf, snap; kwargs...)
    completed && return reached, true
    if reached != start
        # Re-solve from the restored converged state (partial move applied).
        _ctrl_solve!(pf, data, ts; kwargs...)
    end
    return reached, reached != start
end

# Adaptive under-relaxation. The damped iteration p ← p + ω·(σ(V(p)) − p) has local
# slope m = 1 + ω·(g′−1), g′ = σ′(V)·dV/dp ≤ 0 after sign correction. ω is chosen to
# keep m NON-negative (monotone, 0≤m<1, not merely |m|<1): m ≥ θ ⟹ ω ≤ (1−θ)/(1+|g′|).
# |σ′| ≤ |hi−lo|·S/4 bounds g′ at the CURRENT gain estimate (refreshed each step by a
# secant update, so the bound tracks the operating point).
@inline function _relaxation(d, S::Float64, dVdp::Float64)
    lo, hi = parameter_limits(d)
    gbound = 0.25 * abs(hi - lo) * S * abs(dVdp)
    # ω ≤ 1−θ = 0.5 for any gbound ≥ 0, so no additional cap is needed.
    return (1.0 - CONTROL_CONTRACTION) / (1.0 + gbound)
end

# Freeze a device (PSS/E lock-and-continue): it holds its current parameter, counts as
# settled so it cannot stall the steepness ramp for healthy devices, and is reported.
function _freeze_device!(frozen::Vector{Bool}, idx::Int, d, ts::Int, reason::String)
    frozen[idx] = true
    _release_band!(d)
    @warn "discrete control: freezing device $(d.name) at parameter \
        $(current_parameter(d)) (time step $ts): $reason"
    return
end

# Refresh one ControlledFACTS's voltage-dependent susceptance bound (`_facts_b_limit`) from its
# measured controlled-bus voltage, once per continuation pass, BEFORE this pass's damped
# target/clamp reads `parameter_limits`. A `frozen` device holds `b_lim` at its last refreshed
# value: it is never stepped again, so re-tracking voltage would only chatter the bound with no
# corresponding parameter move.
function _refresh_facts_limit!(d::ControlledFACTS, data, ts::Int, frozen::Bool)
    frozen && return d.b_lim
    d.b_lim = _facts_b_limit(d, measured_value(d, data, ts))
    return d.b_lim
end

# Per-pass refresh of every FACTS device's bound (function-barrier group wrapper, offset-
# indexed into the shared `frozen` bookkeeping — mirrors `_probe_device_signs!`).
function _refresh_facts_limits!(
    devices::Vector{ControlledFACTS}, offset::Int, data, ts::Int, frozen::Vector{Bool},
)
    for (i, d) in enumerate(devices)
        _refresh_facts_limit!(d, data, ts, frozen[offset + i])
    end
    return
end

# Compute the damped, sign-corrected target parameter for one device and advance its
# oscillation state. Returns `(p_new, yc, ok)`: `yc` is the pre-move regulated quantity (for a
# secant refresh); `ok=false` means the device was frozen (oscillation) or held in its deadband —
# do not move it. Shared by the sequential (`_step_device!`) and batched (`_batched_pass!`) paths.
function _damped_target!(
    d, idx::Int, data, ts::Int, S::Float64, frozen::Vector{Bool},
    dVdp::Vector{Float64}, osc::Vector{Int}, prev_sign::Vector{Int}, n_shared::Vector{Int},
)
    yc = measured_value(d, data, ts)
    p_now = current_parameter(d)
    _update_band_state!(d, yc)
    if _in_deadband(d, yc)
        # PSS/E deadband semantics: a device whose regulated quantity is inside its
        # band is held, not driven to the band midpoint.
        prev_sign[idx] = 0
        return p_now, yc, false
    end
    lo, hi = parameter_limits(d)
    tol_d = _param_tol(d)
    dv = dVdp[idx]
    # Devices sharing a controlled bus split the correction: the per-device contraction
    # bound does not see the cross-coupling, and N co-located controllers stepping the
    # full error together have an in-phase gain ≈ N× the measured self-gain.
    ω = _relaxation(d, S, dv) / n_shared[idx]
    p_tgt = _control_target(d, yc, S, dv)
    # Track sign reversals to detect within-stage oscillation. Sub-tolerance target
    # moves carry no direction information (grid/tolerance dither, not instability).
    s = 0
    if abs(p_tgt - p_now) >= tol_d
        s = Int(sign(p_tgt - p_now))
    end
    ps = prev_sign[idx]
    if !iszero(ps) && !iszero(s) && s != ps
        osc[idx] += 1
        if osc[idx] > CONTROL_OSCILLATION_LIMIT
            _freeze_device!(frozen, idx, d, ts,
                "oscillating ($(osc[idx]) direction reversals within a steepness stage)")
            return p_now, yc, false
        end
    end
    prev_sign[idx] = s
    p_new = clamp(p_now + ω * (p_tgt - p_now), lo, hi)
    # Rail snap (continuous devices): a target within tol_d of a bound, with the regulated
    # quantity still off its setpoint, is the device saturating (the PV→PQ analogue). The damped
    # step ω·(bound − p) shrinks toward tol_d as the steepness grows, so the walk would stall
    # short of the rail (~6 % of range for FACTS at S = 5000) and use up the stage's pass budget:
    # apply the bound itself. Only from within CONTROL_RAIL_SNAP_REACH of it: from farther away
    # (a far-off start, an early low-steepness stage) a jump to the rail can overshoot onto
    # another solution branch, and the damped walk is not yet creeping there.
    if _rail_snaps(d) && abs(yc - control_setpoint(d)) > CONTROL_FACTS_SETPOINT_BAND
        reach = CONTROL_RAIL_SNAP_REACH * (hi - lo)
        p_tgt >= hi - tol_d && hi - p_now <= reach && (p_new = hi)
        p_tgt <= lo + tol_d && p_now - lo <= reach && (p_new = lo)
    end
    return _band_clamp(d, p_new, dv), yc, true
end

_rail_snaps(::AbstractControlledDevice) = false
_rail_snaps(::ControlledFACTS) = true
_rail_snaps(d::ControlledSwitchedShunt) = !isnothing(_continuous_target(d))

# Freeze on a detected plant-gain sign reversal (OLTC reverse action) or a collapse below the
# effectiveness floor; otherwise accept the refreshed gain `g`. `dv` is the pre-refresh gain.
function _apply_gain_refresh!(d, idx::Int, data, ts::Int, frozen::Vector{Bool},
    dVdp::Vector{Float64}, dv::Float64, g::Float64)
    lo, hi = parameter_limits(d)
    # A device whose orientation is imposed (PSS/E CONT sign, or a parallel group's alignment)
    # takes only the measured magnitude: its own measured sign may legitimately oppose it.
    f = _forced_sign(d)
    iszero(f) || (g = f * abs(g))
    if !iszero(dv) && !iszero(g) && sign(g) != sign(dv)
        _freeze_device!(frozen, idx, d, ts,
            "plant sensitivity changed sign along the trajectory (reverse action); \
            continuing would be positive feedback")
    elseif abs(g) * (hi - lo) < CONTROL_GAIN_FLOOR
        _freeze_device!(frozen, idx, d, ts,
            "plant sensitivity collapsed below the effectiveness floor \
            (|dy/dp|·range = $(abs(g) * (hi - lo)))")
    else
        dVdp[idx] = g
    end
    return
end

# Secant refresh gate: a measured Δy at or below solver noise carries no sign
# information — skip the refresh entirely (both the reverse-action freeze AND the
# effectiveness-floor freeze would be spurious on a noise sample).
function _maybe_refresh_gain!(d, idx::Int, data, ts::Int, frozen::Vector{Bool},
    dVdp::Vector{Float64}, dv::Float64, g::Float64, Δy::Float64)
    abs(Δy) < CONTROL_MEASUREMENT_FLOOR && return
    _apply_gain_refresh!(d, idx, data, ts, frozen, dVdp, dv, g)
    return
end

# One damped, sign-corrected proportional update of a single device (SEQUENTIAL path: apply +
# solve per device). Returns the magnitude of the parameter change actually applied (for the
# settling test); frozen and in-deadband devices return 0.0. The measured plant gain is refreshed
# by a secant update from the numbers the step just produced (zero extra solves).
function _step_device!(
    d,
    idx::Int,
    data,
    ts::Int,
    S::Float64,
    pf,
    scratch_snap::ControlStateSnapshot,
    frozen::Vector{Bool},
    dVdp::Vector{Float64},
    osc::Vector{Int},
    prev_sign::Vector{Int},
    n_shared::Vector{Int};
    kwargs...,
)::Float64
    frozen[idx] && return 0.0
    p_new, yc, ok =
        _damped_target!(d, idx, data, ts, S, frozen, dVdp, osc, prev_sign, n_shared)
    ok || return 0.0
    p_now = current_parameter(d)
    tol_d = _param_tol(d)
    dv = dVdp[idx]
    reached, moved = _continuation_to!(d, data, ts, p_new, pf, scratch_snap; kwargs...)
    if !moved && abs(p_new - p_now) >= tol_d
        # Inner solver rejects any movement — freeze rather than let a zero change look settled.
        _freeze_device!(frozen, idx, d, ts,
            "the inner solver rejects any parameter movement (requested \
            $(p_new - p_now))")
        return 0.0
    end
    Δp = reached - p_now
    if abs(Δp) >= tol_d
        Δy = measured_value(d, data, ts) - yc
        _maybe_refresh_gain!(d, idx, data, ts, frozen, dVdp, dv, Δy / Δp, Δy)
    end
    return abs(Δp)
end

# One device's plant sign: the linear sensitivity when a polar `ctx` is live and the family has an
# analytic form (tap/shunt/FACTS), else the FD probe (non-polar formulations, singular base).
_probe_one_sign(
    d, data, ts::Int, pf, scratch_snap::ControlStateSnapshot, ::Nothing; kwargs...,
) = _plant_sign(d, data, ts, pf, scratch_snap; kwargs...)
function _probe_one_sign(
    d, data, ts::Int, pf, scratch_snap::ControlStateSnapshot,
    ctx::_SensitivityContext; kwargs...,
)
    dydp, ok = _linear_plant_sign(d, data, ts, ctx)
    ok && return dydp, true
    return _plant_sign(d, data, ts, pf, scratch_snap; kwargs...)
end

# Probe every device in a concrete vector (function barrier — no dynamic dispatch over the
# heterogeneous set), writing dy/dp at `offset + i`. Freeze probes that fail or whose full-range
# effect is below the gain floor (e.g. a PV-pinned bus, sensitivity 0) — stepping them would rail.
function _probe_device_signs!(
    devices, offset::Int, dVdp::Vector{Float64}, frozen::Vector{Bool},
    ctx::Union{Nothing, _SensitivityContext}, data, ts::Int, pf,
    scratch_snap::ControlStateSnapshot; kwargs...,
)
    for (i, d) in enumerate(devices)
        s, reliable = _probe_one_sign(d, data, ts, pf, scratch_snap, ctx; kwargs...)
        lo, hi = parameter_limits(d)
        if reliable && abs(s) * (hi - lo) >= CONTROL_GAIN_FLOOR
            f = _forced_sign(d)
            dVdp[offset + i] = iszero(f) ? s : f * abs(s)
        else
            frozen[offset + i] = true
        end
    end
    return
end

# Parallel taps regulating one bus (the same controlled bus and the same pair of terminal
# buses) are one control: PSS/E moves such units the same way. Measured one at a time their
# dV/dp can disagree in sign where each unit's own effect is weak (a tap regulating its own
# winding-1 bus, or units at unequal ratios with a circulating current between them), and the
# continuation then walks them apart, each "helping" its bus through the circulating current: a
# tap fight. When the members' signs disagree (taken in a common arc orientation: a member
# whose arc is reversed counts with its sign flipped), each member whose orientation is not
# fixed by `tap_direction = :cont_sign` is aligned to the sign of the members' summed
# sensitivity, the effect of moving them together; its measured magnitude still sizes its
# steps (`aligned`, see `_forced_sign`).
#
# Every group, aligned or not, also keeps its units' ratio offsets (GAP-058). The sigmoid law
# drives a device to an absolute parameter for a given voltage, so units at unequal ratios were
# pulled toward one common ratio: the higher one down, the lower one up, or, where the band clamp
# held one at its anchor, only the other moved, and the pair split (at 15 steps apart, one unit 3
# steps, the other none). Each unit's `offset` is its ratio less the mean of the units that share
# its arc orientation, and its sigmoid window is shifted by it (`_control_target`), so the units
# move by the same amount, as PSS/E steps parallel units together.
#
# Parallel phase shifters (the same from and to buses, in the same orientation) are grouped the
# same way, and each member measures the group's mean flow (`group`), so the units regulate one
# quantity and move together.
_parallel_key(d::ControlledTap) = (d.controlled_ix, minmax(d.from_ix, d.to_ix)...)
_parallel_key(d::ControlledPhaseShifter) = (0, d.from_ix, d.to_ix)
_set_group!(::ControlledTap, members) = nothing
_set_group!(d::ControlledPhaseShifter, members) = (d.group = members; nothing)

function _align_parallel_taps!(taps::Vector{<:BranchDevice}, dVdp::Vector{Float64},
    frozen::Vector{Bool}, offset::Int = 0)
    length(taps) < 2 && return
    groups = Dict{NTuple{3, Int}, Vector{Int}}()
    for (i, d) in enumerate(taps)
        frozen[offset + i] && continue
        push!(get!(groups, _parallel_key(d), Int[]), i)
    end
    for members in values(groups)
        length(members) < 2 && continue
        orient(i) = taps[i].from_ix < taps[i].to_ix ? 1 : -1
        for o in (1, -1)
            same = [i for i in members if orient(i) == o]
            isempty(same) && continue
            m = sum(taps[i].current for i in same) / length(same)
            for i in same
                taps[i].offset = taps[i].current - m
            end
        end
        unit_group = taps[members]
        for i in members
            _set_group!(taps[i], unit_group)
        end
        signs = [sign(dVdp[offset + i]) * orient(i) for i in members]
        all(==(signs[1]), signs) && continue
        total = sum(dVdp[offset + i] * orient(i) for i in members)
        iszero(total) && continue
        for i in members
            d = taps[i]
            iszero(d.orientation) || continue
            d.aligned = Int(sign(total)) * orient(i)
            dVdp[offset + i] = d.aligned * abs(dVdp[offset + i])
        end
        names = join((taps[i].name for i in members), ", ")
        @debug "discrete control: parallel units $names measured opposite sensitivity \
            signs; aligned to their joint effect."
    end
    return
end

# One proportional update over a concrete device vector; returns `true` iff every device
# in it settled (parameter change below its scale-aware tolerance). Same function-barrier
# + `offset` indexing.
function _step_device_group!(
    devices, offset::Int, data, ts::Int, S::Float64, pf,
    scratch_snap::ControlStateSnapshot,
    frozen::Vector{Bool}, dVdp::Vector{Float64}, osc::Vector{Int},
    prev_sign::Vector{Int}, n_shared::Vector{Int}; kwargs...,
)::Bool
    settled = true
    for (i, d) in enumerate(devices)
        g = _step_device!(
            d, offset + i, data, ts, S, pf, scratch_snap, frozen, dVdp, osc, prev_sign,
            n_shared; kwargs...)
        g >= _param_tol(d) && (settled = false)
    end
    return settled
end

# ── Batched (Jacobi) pass ─────────────────────────────────────────────────────────────────
# Apply every device's damped target, then ONE joint inner solve (vs ~N sequential — the
# relaxation theory already models a pass as one damped joint update). Gains are refreshed
# analytically (a joint move corrupts the secant gain at co-located buses); a non-converged joint
# solve rolls the whole pass back and the caller falls to the sequential path.

# Apply every non-frozen device's damped target WITHOUT solving; record pre-move parameters in
# `p_prev`/`did_move` (indexed globally) for rollback and the post-solve gain refresh. Returns
# whether any device moved. Function-barrier per concrete device vector (no dynamic dispatch).
function _apply_targets_group!(
    devices, offset::Int, data, ts::Int, S::Float64, frozen::Vector{Bool},
    dVdp::Vector{Float64}, osc::Vector{Int}, prev_sign::Vector{Int}, n_shared::Vector{Int},
    p_prev::Vector{Float64}, did_move::Vector{Bool},
)
    # Every target is taken before any is applied: a phase shifter's measured flow depends on
    # its own angle and its parallel units' (`measured_value`), which an earlier member's apply
    # would change on the unsolved state. (A voltage device's measurement is its bus voltage,
    # which an apply does not touch, so for those the two orders agree.)
    any_moved = false
    targets = Vector{Float64}(undef, length(devices))
    for (i, d) in enumerate(devices)
        idx = offset + i
        targets[i] = NaN
        frozen[idx] && continue
        p_now = current_parameter(d)
        p_new, _, ok =
            _damped_target!(d, idx, data, ts, S, frozen, dVdp, osc, prev_sign, n_shared)
        (ok && abs(p_new - p_now) >= _param_tol(d)) || continue
        targets[i] = p_new
    end
    for (i, d) in enumerate(devices)
        isnan(targets[i]) && continue
        idx = offset + i
        p_prev[idx] = current_parameter(d)
        apply_parameter!(d, data, targets[i], ts)
        did_move[idx] = true
        any_moved = true
    end
    return any_moved
end

# Undo an applied batched pass: restore each moved device's parameter (delta-based apply reverses
# the Y-bus/withdrawal edit). The caller also `_restore_state!`s V/θ/bustype/injections.
function _rollback_targets_group!(
    devices, offset::Int, data, ts::Int, p_prev::Vector{Float64}, did_move::Vector{Bool},
)
    for (i, d) in enumerate(devices)
        idx = offset + i
        did_move[idx] || continue
        apply_parameter!(d, data, p_prev[idx], ts)
    end
    return
end

# Analytic (coupling-free) gain refresh for the moved devices of one group, using a `ctx` built at
# the post-solve state. Same sign-reversal / gain-floor freeze policy as the secant refresh.
function _refresh_gains_group!(
    devices, offset::Int, data, ts::Int, frozen::Vector{Bool}, dVdp::Vector{Float64},
    did_move::Vector{Bool}, ctx::_SensitivityContext,
)
    for (i, d) in enumerate(devices)
        idx = offset + i
        did_move[idx] || continue
        g, ok = _linear_plant_sign(d, data, ts, ctx)
        ok || continue
        _apply_gain_refresh!(d, idx, data, ts, frozen, dVdp, dVdp[idx], g)
    end
    return
end

# One batched pass over the voltage-device groups (taps, shunts, FACTS). Returns
# `(settled, converged)`: `converged=false` means the joint solve failed (or, rarely, the
# post-solve sensitivity refresh hit a singular Jacobian) and the pass was fully rolled back —
# the caller must run the sequential path for this pass. `ctx` is the ONE
# `_SensitivityContext` persisted for the whole continuation (built once in
# `_control_continuation!`); a converged joint solve refreshes it in place instead of rebuilding.
function _batched_pass!(
    set::ControlledDeviceSet, n_taps::Int, n_shunts::Int, data, ts::Int, S::Float64, pf,
    snap::ControlStateSnapshot, frozen::Vector{Bool}, dVdp::Vector{Float64},
    osc::Vector{Int}, prev_sign::Vector{Int}, n_shared::Vector{Int},
    p_prev::Vector{Float64}, did_move::Vector{Bool},
    ctx::_SensitivityContext; kwargs...,
)
    fill!(did_move, false)
    _capture_state!(snap, data, ts)
    # A failed+rolled-back batched attempt must leave NO trace, so the sequential fallback is the
    # authoritative pass: snapshot the per-device bookkeeping `_damped_target!` mutates
    # (oscillation counter, direction, freezes) and restore it on rollback.
    osc0, prev0, frozen0 = copy(osc), copy(prev_sign), copy(frozen)
    side0 = Int[d.side for d in set.taps]
    anchor0 = Float64[d.anchor for d in set.taps]
    shunt_side0 = Int[d.side for d in set.shunts]
    shunt_anchor0 = Float64[d.anchor for d in set.shunts]
    pst_side0 = Int[d.side for d in set.phase_shifters]
    pst_anchor0 = Float64[d.anchor for d in set.phase_shifters]
    off_facts = n_taps + n_shunts
    off_pst = off_facts + length(set.facts)
    moved = _apply_targets_group!(set.taps, 0, data, ts, S, frozen, dVdp, osc, prev_sign,
        n_shared, p_prev, did_move)
    moved |= _apply_targets_group!(set.shunts, n_taps, data, ts, S, frozen, dVdp, osc,
        prev_sign, n_shared, p_prev, did_move)
    # FACTS branch: refresh each device's |V|-dependent susceptance bound from the voltage at
    # the start of this pass (no solve has run yet within this batched pass) before it moves.
    _refresh_facts_limits!(set.facts, off_facts, data, ts, frozen)
    moved |= _apply_targets_group!(set.facts, off_facts, data, ts, S, frozen, dVdp, osc,
        prev_sign, n_shared, p_prev, did_move)
    moved |= _apply_targets_group!(set.phase_shifters, off_pst, data, ts, S, frozen, dVdp,
        osc, prev_sign, n_shared, p_prev, did_move)
    moved || return true, true            # nothing wanted to move ⇒ settled, no solve needed
    if _ctrl_solve!(pf, data, ts; kwargs...) && _refresh_sensitivity_context!(ctx, data, ts; kwargs...)
        _refresh_gains_group!(set.taps, 0, data, ts, frozen, dVdp, did_move, ctx)
        _refresh_gains_group!(set.shunts, n_taps, data, ts, frozen, dVdp, did_move, ctx)
        _refresh_gains_group!(set.facts, off_facts, data, ts, frozen, dVdp, did_move, ctx)
        _refresh_gains_group!(set.phase_shifters, off_pst, data, ts, frozen, dVdp, did_move,
            ctx)
        return false, true                # moved + converged ⇒ not settled
    end
    _rollback_targets_group!(set.taps, 0, data, ts, p_prev, did_move)
    _rollback_targets_group!(set.shunts, n_taps, data, ts, p_prev, did_move)
    _rollback_targets_group!(set.facts, off_facts, data, ts, p_prev, did_move)
    _rollback_targets_group!(set.phase_shifters, off_pst, data, ts, p_prev, did_move)
    _restore_state!(data, ts, snap)
    copyto!(osc, osc0)
    copyto!(prev_sign, prev0)
    copyto!(frozen, frozen0)
    for (i, d) in enumerate(set.taps)
        d.side = side0[i]
        d.anchor = anchor0[i]
    end
    for (i, d) in enumerate(set.shunts)
        d.side = shunt_side0[i]
        d.anchor = shunt_anchor0[i]
    end
    for (i, d) in enumerate(set.phase_shifters)
        d.side = pst_side0[i]
        d.anchor = pst_anchor0[i]
    end
    return false, false          # joint solve or sensitivity refresh failed ⇒ run sequential path
end

function _count_controlled_buses!(counts::Dict{Int, Int}, devices)
    for d in devices
        cix = controlled_bus_ix(d)
        counts[cix] = get(counts, cix, 0) + 1
    end
    return
end

function _fill_shared!(
    n_shared::Vector{Int}, offset::Int, counts::Dict{Int, Int}, devices,
)
    for (i, d) in enumerate(devices)
        n_shared[offset + i] = counts[controlled_bus_ix(d)]
    end
    return
end

# Sequential update of the three voltage-device groups; returns whether all settled.
function _step_voltage_groups!(
    set::ControlledDeviceSet, n_taps::Int, n_shunts::Int, data, ts::Int, S::Float64, pf,
    scratch_snap::ControlStateSnapshot, frozen::Vector{Bool}, dVdp::Vector{Float64},
    osc::Vector{Int}, prev_sign::Vector{Int}, n_shared::Vector{Int}; kwargs...,
)
    settled = _step_device_group!(set.taps, 0, data, ts, S, pf, scratch_snap, frozen, dVdp,
        osc, prev_sign, n_shared; kwargs...)
    settled &= _step_device_group!(set.shunts, n_taps, data, ts, S, pf, scratch_snap,
        frozen, dVdp, osc, prev_sign, n_shared; kwargs...)
    # FACTS branch: refresh each device's |V|-dependent susceptance bound from the CURRENT
    # measured voltage (post taps/shunts moves earlier in this same pass) before it is stepped.
    _refresh_facts_limits!(set.facts, n_taps + n_shunts, data, ts, frozen)
    settled &= _step_device_group!(set.facts, n_taps + n_shunts, data, ts, S, pf,
        scratch_snap, frozen, dVdp, osc, prev_sign, n_shared; kwargs...)
    settled &= _step_device_group!(set.phase_shifters, n_taps + n_shunts + length(set.facts),
        data, ts, S, pf, scratch_snap, frozen, dVdp, osc, prev_sign, n_shared; kwargs...)
    return settled
end

# One continuation pass. Voltage devices go through the batched (one-solve) path when it is
# enabled and its joint solve converges, else the sequential path (which fully preserves the
# backtracking/freeze behavior). Returns whether the whole pass settled. `ctx` is the
# persisted sensitivity context (non-`nothing` iff `use_batched`); passed through unchanged
# so `_batched_pass!` can refresh it in place.
function _control_pass!(
    set::ControlledDeviceSet, n_taps::Int, n_shunts::Int, use_batched::Bool,
    data, ts::Int, S::Float64, pf, scratch_snap::ControlStateSnapshot,
    frozen::Vector{Bool}, dVdp::Vector{Float64},
    osc::Vector{Int}, prev_sign::Vector{Int}, n_shared::Vector{Int},
    p_prev::Vector{Float64}, did_move::Vector{Bool},
    ctx::Union{Nothing, _SensitivityContext}; kwargs...,
)
    if use_batched
        s, converged =
            _batched_pass!(set, n_taps, n_shunts, data, ts, S, pf, scratch_snap, frozen,
                dVdp, osc, prev_sign, n_shared, p_prev, did_move, ctx; kwargs...)
        converged && return s
        set.batched_fallbacks[] += 1
    end
    return _step_voltage_groups!(set, n_taps, n_shunts, data, ts, S, pf, scratch_snap,
        frozen, dVdp, osc, prev_sign, n_shared; kwargs...)
end

function _control_continuation!(
    pf,
    data,
    ts::Int;
    kwargs...,
)::Bool
    set = data.controlled_devices::ControlledDeviceSet
    set.inner_solves[] = 0
    set.symbolic_factors[] = 0
    set.numeric_refactors[] = 0
    set.batched_fallbacks[] = 0
    set.context_rebuilds[] = 0
    converged = _ctrl_solve!(pf, data, ts; kwargs...)
    converged || return false
    for d in set.taps
        d.side = 0
        d.anchor = d.current
        d.aligned = 0
        d.offset = 0.0
    end
    for d in set.shunts
        d.side = 0
        d.anchor = d.current
    end
    for d in set.phase_shifters
        d.side = 0
        d.anchor = d.current
        d.aligned = 0
        d.offset = 0.0
        d.group = ControlledPhaseShifter[]
    end

    n_taps = length(set.taps)
    n_shunts = length(set.shunts)
    n_facts = length(set.facts)
    n_psts = length(set.phase_shifters)
    off_pst = n_taps + n_shunts + n_facts
    n_dev = off_pst + n_psts
    # Per-device state, indexed in parallel with [taps; shunts; facts]. dVdp: sign
    # sets the negative-feedback orientation, magnitude drives ω; frozen devices are held, not stepped.
    dVdp = zeros(n_dev)
    frozen = fill(false, n_dev)
    osc = zeros(Int, n_dev)
    prev_sign = zeros(Int, n_dev)
    # Voltage devices sharing a controlled bus split the correction (ω / n_shared).
    n_shared = ones(Int, n_dev)
    counts = Dict{Int, Int}()
    _count_controlled_buses!(counts, set.taps)
    _count_controlled_buses!(counts, set.shunts)
    _count_controlled_buses!(counts, set.facts)
    _fill_shared!(n_shared, 0, counts, set.taps)
    _fill_shared!(n_shared, n_taps, counts, set.shunts)
    _fill_shared!(n_shared, n_taps + n_shunts, counts, set.facts)
    # Phase shifters regulate branch flows, not bus voltages: parallel units (one group, see
    # `_align_parallel_taps!`) split the correction among themselves.
    pst_counts = Dict{NTuple{3, Int}, Int}()
    for d in set.phase_shifters
        k = _parallel_key(d)
        pst_counts[k] = get(pst_counts, k, 0) + 1
    end
    for (i, d) in enumerate(set.phase_shifters)
        n_shared[off_pst + i] = pst_counts[_parallel_key(d)]
    end

    # One buffer shared by the probe phase, the pass loop, and the post-loop per-device
    # restore: those phases run in sequence and are never simultaneously live. The exception
    # is `snap_and_restore!`'s `pre`, which allocates its own.
    scratch_snap = _snapshot_state(data, ts)

    # Build the linearized-sensitivity context ONCE (one numeric factorization reusing the base NR
    # solve's symbolic factor); all device probes below are then triangular solves against it. `nothing`
    # for non-polar formulations or a singular base ⇒ each probe falls back to the FD solve.
    ctx = _sensitivity_context(pf, data, ts; kwargs...)
    _probe_device_signs!(
        set.taps,
        0,
        dVdp,
        frozen,
        ctx,
        data,
        ts,
        pf,
        scratch_snap;
        kwargs...,
    )
    _align_parallel_taps!(set.taps, dVdp, frozen)
    _probe_device_signs!(
        set.shunts, n_taps, dVdp, frozen, ctx, data, ts, pf, scratch_snap; kwargs...)
    _probe_device_signs!(
        set.facts, n_taps + n_shunts, dVdp, frozen, ctx, data, ts, pf, scratch_snap;
        kwargs...)
    # Group parallel phase shifters first, so each probe measures the group's mean flow; then
    # align any opposite signs.
    _align_parallel_taps!(set.phase_shifters, dVdp, frozen, off_pst)
    _probe_device_signs!(
        set.phase_shifters, off_pst, dVdp, frozen, ctx, data, ts, pf, scratch_snap;
        kwargs...)
    _align_parallel_taps!(set.phase_shifters, dVdp, frozen, off_pst)
    if any(frozen)
        frozen_names = join(
            vcat(
                [set.taps[i].name for i in 1:n_taps if frozen[i]],
                [set.shunts[j].name
                    for j in eachindex(set.shunts) if frozen[n_taps + j]],
                [
                    set.facts[k].name
                    for k in eachindex(set.facts) if frozen[n_taps + n_shunts + k]
                ],
                [set.phase_shifters[k].name for k in 1:n_psts if frozen[off_pst + k]],
            ),
            ", ",
        )
        @warn "discrete control: $(count(frozen)) device(s) had an unreliable or \
            insensitive plant probe and were frozen at their current parameter \
            (time step $ts): $frozen_names"
    end

    # Per-stage pass budget + per-stage oscillation reset (a ramp legitimately reverses direction
    # once). Intermediate stages solve at CONTROL_STAGE_TOL; full tol only at the final stage and
    # snap/restore, and never looser than a user-supplied tol.
    user_tol = Float64(get(kwargs, :tol, DEFAULT_NR_TOL))
    # Batch the voltage-device passes when the polar linear path is live (a non-`nothing` probe
    # `ctx`): one joint solve per pass with analytic per-pass gain refresh, falling back to the
    # sequential path on a failed joint solve. Non-polar formulations step sequentially.
    use_batched = !isnothing(ctx)
    p_prev = zeros(n_dev)
    did_move = fill(false, n_dev)
    S = INITIAL_CONTROL_STEEPNESS
    regulation_complete = false
    while true
        stage_tol = user_tol
        if S < MAX_CONTROL_STEEPNESS
            stage_tol = max(user_tol, CONTROL_STAGE_TOL)
        end
        settled = false
        for _ in 1:MAX_CONTROL_PASSES_PER_STAGE
            _control_time_up!(set) && break
            if !use_batched
                # No context yet (the probe-phase build hit a singular Jacobian): try again at
                # this pass's converged state, so one bad base point does not leave the whole
                # continuation on the per-device path. `nothing` again for a non-polar
                # formulation, which has no linear path.
                ctx = _sensitivity_context(pf, data, ts; kwargs...)
                use_batched = !isnothing(ctx)
            end
            settled = _control_pass!(
                set, n_taps, n_shunts, use_batched, data, ts, S, pf,
                scratch_snap,
                frozen, dVdp, osc, prev_sign, n_shared, p_prev, did_move, ctx;
                kwargs..., tol = stage_tol)
            settled && break
        end
        if S >= MAX_CONTROL_STEEPNESS || _control_stopped(set)
            regulation_complete = settled
            break
        end
        S = min(S * CONTROL_STEEPNESS_GROWTH, MAX_CONTROL_STEEPNESS)
        fill!(osc, 0)
        fill!(prev_sign, 0)
    end
    if !regulation_complete
        @warn "discrete control: the final steepness stage did not settle within \
            MAX_CONTROL_PASSES_PER_STAGE=$(MAX_CONTROL_PASSES_PER_STAGE) passes \
            (S=$S); regulation may be loose at time step $ts"
    end
    # Regulating continuous devices (shunts, FACTS) end on their target exactly, or at a limit:
    # before the band settle, whose tests of the banded devices' voltages then read the network
    # with the continuous devices where they end (T161), and again after it, for what its steps
    # moved.
    finish!() = _continuous_finish!(pf, data, set, ts, scratch_snap, frozen, dVdp, n_shared,
        n_taps, ctx; kwargs...)
    ok = snap_and_restore!(pf, data, set, ts, scratch_snap, dVdp, frozen, finish!; kwargs...)
    ok && finish!()
    # Classify each FACTS endpoint against its final voltage-dependent bound and setpoint
    # (`saturated` reported by `get_controlled_device_results` and written back to PSY). Uses
    # the final converged voltages from `snap_and_restore!`.
    for d in set.facts
        classify_facts_saturation!(d, measured_value(d, data, ts))
    end
    # Reported branch flows are computed from the arc-admittance matrices right after
    # this time step in `solve_power_flow!`; bring the moved branch devices' rows in
    # line with their final parameters so flows match the solved network.
    _sync_arc_admittances!(data, set)
    return ok
end

# ── Exact finish for continuous devices ──────────────────────────────────────────────────
# The sigmoid law's fixed point at the final steepness sits off the target by ~ln(·)/S, and a
# device that should be pinned at a limit stops short of it while the sigmoid is not fully
# saturated. Over a zero-width band (VSWLO = VSWHI: a setpoint, as WECC models PSLF-style SVDs)
# that residue is the whole answer: measured on a WECC replay, 0.2–3.6 % of range (up to 7.6e-4
# p.u. of voltage) off a free device's setpoint, and 0.9–1.8 % of range short of the limit of
# one PSS/E pins there; FACTS that PSS/E holds at |Q| = SHMX·V ended at 0.94 of their bound,
# neither at setpoint nor classified saturated. So after the snap every regulating continuous
# device (a continuous switched shunt with a zero-width band or latched on an edge, and every
# FACTS) takes undamped Newton steps on its susceptance, all devices together with one joint
# solve per round: the joint step on the devices' sensitivities at the round's state
# (`_finish_joint_step!`, T161), or without a sensitivity context b ← clamp(b − (y − y*)/(dy/db))
# on each device's own secant-refreshed dy/db, until
# |y − y*| ≤ CONTROL_FINISH_VTOL, the step clamps at a limit (the device then sits exactly
# there; a FACTS's |V|-dependent bound is refreshed every round), or the step's own mismatch
# |Δb|·V² is below the finish's solve tolerance. Those solves converge to
# CONTROL_FINISH_SOLVE_TOL (T152): at the caller's tolerance a step smaller than it was taken
# without a Newton iteration, so the finish read the voltage from before its step and stopped
# wherever its last round left it, off its target by up to dV/dQ times that tolerance. A
# banded shunt never retreats past its anchor (`_band_clamp`), and a latched one on GAP-056's
# hold band ends half the band inside its edge (`finish_margin`). A failed round rolls back to
# the last converged state with a warning.

# The quantity a continuous device is held to at the end, or `nothing` when it is not
# regulating (a discrete shunt, or one held inside a nonzero band). A latched banded device ends
# `finish_margin` inside the edge it latched on (GAP-056's hold band: half the band; T152).
function _continuous_target(d::ControlledSwitchedShunt)
    d.continuous || return nothing
    d.vset_lo == d.vset_hi && return d.vset
    d.band && d.side != 0 && return control_setpoint(d) - d.side * d.finish_margin
    return nothing
end
_continuous_target(d::ControlledFACTS) = d.vset

_finish_limits!(d::ControlledSwitchedShunt, data, ts::Int) = parameter_limits(d)
function _finish_limits!(d::ControlledFACTS, data, ts::Int)
    d.b_lim = _facts_b_limit(d, measured_value(d, data, ts))
    return parameter_limits(d)
end

# One Newton round's moves for one group; returns whether any device moved.
function _finish_step_group!(devices, offset::Int, data, ts::Int, active, dVdp, n_shared,
    prev, yprev, moved, ftol::Float64)
    any_moved = false
    for (j, d) in enumerate(devices)
        idx = offset + j
        active[idx] || continue
        y = measured_value(d, data, ts)
        lo, hi = _finish_limits!(d, data, ts)
        e = y - _continuous_target(d)
        g = dVdp[idx]
        if abs(e) <= CONTROL_FINISH_VTOL
            lo <= d.current <= hi && continue      # on target, within its (refreshed) limits
            p = clamp(d.current, lo, hi)
        else
            p = clamp(d.current - e / (g * n_shared[idx]), lo, hi)
        end
        p = _band_clamp(d, p, g)
        if abs(p - d.current) <= 1e-12 * (hi - lo)
            active[idx] = false   # pinned at a limit (or its anchor): it stays there
            continue
        end
        # a step inside the range below what the solve resolves: finished (a step onto a limit is
        # still taken, so a railed device ends exactly on it)
        if lo < p < hi && abs(p - d.current) * data.bus_magnitude[d.bus_ix, ts]^2 <= ftol
            active[idx] = false
            continue
        end
        prev[idx] = d.current
        yprev[idx] = y
        apply_parameter!(d, data, p, ts)
        moved[idx] = true
        any_moved = true
    end
    return any_moved
end

function _finish_secant_group!(devices, offset::Int, data, ts::Int, dVdp, prev, yprev, moved)
    for (j, d) in enumerate(devices)
        idx = offset + j
        moved[idx] || continue
        Δy = measured_value(d, data, ts) - yprev[idx]
        g = Δy / (d.current - prev[idx])
        # Keep the previous gain on a noise-level or reversed secant sample.
        if abs(Δy) > 1e-10 && sign(g) == sign(dVdp[idx])
            dVdp[idx] = g
        end
    end
    return
end

function _finish_undo_group!(devices, offset::Int, data, ts::Int, prev, moved)
    for (j, d) in enumerate(devices)
        moved[offset + j] && apply_parameter!(d, data, prev[offset + j], ts)
    end
    return
end

# The finish's joint Newton step (T161). Each device's own secant gain is measured from a voltage
# its neighbours moved in the same round, so coupled devices stepped together on those gains
# swing between under- and overshoot; instead the round solves the devices' joint linearisation
# at its converged state: with M[i, j] = dy_i/db_j from the sensitivity context (one triangular
# solve per device), the steps Δb solve y + M·Δb = y*. Devices regulating one bus have equal rows
# and share the correction equally in voltage (`u_h`, each member b_j += u_h/(m·M[j, j])), with
# the members' mean target. A device whose step leaves its range (or passes its anchor, see
# `_band_clamp`) is held at that bound and the rest solved again, at most once per device; one
# already there, with nowhere to go, is pinned for the rest of the finish, as in the secant step.
# A step inside the range below what the solve resolves (|Δb|·V² ≤ `ftol`) is not taken this
# round. A device whose own effect is below the gain floor (its bus held by a generator) is left
# out of the round. Returns whether any device moved, or `nothing` when the reduced system is
# singular (the caller takes the secant step instead).
function _finish_joint_step!(data, set::ControlledDeviceSet, ts::Int, n_taps::Int,
    active::Vector{Bool}, prev::Vector{Float64}, moved::Vector{Bool}, ftol::Float64,
    ctx::_SensitivityContext)
    off_facts = n_taps + length(set.shunts)
    idxs = Int[]
    devs = AbstractShuntControl[]
    for (j, d) in enumerate(set.shunts)
        active[n_taps + j] && (push!(idxs, n_taps + j); push!(devs, d))
    end
    for (j, d) in enumerate(set.facts)
        active[off_facts + j] && (push!(idxs, off_facts + j); push!(devs, d))
    end
    k = length(devs)
    k == 0 && return false
    e = Vector{Float64}(undef, k)
    lo = Vector{Float64}(undef, k)
    hi = Vector{Float64}(undef, k)
    cbus = Vector{Int}(undef, k)
    for (i, d) in enumerate(devs)
        lo[i], hi[i] = _finish_limits!(d, data, ts)
        e[i] = measured_value(d, data, ts) - _continuous_target(d)
        cbus[i] = controlled_bus_ix(d)
    end
    # M[i, j] = dy_i/db_j: y_i is |V| at a PQ bus, x[2b−1] (see `_linear_plant_sign`).
    M = zeros(k, k)
    for (j, d) in enumerate(devs)
        _dF_dp!(ctx.rhs, d, data, ts)
        copyto!(ctx.sol, ctx.rhs)
        solve!(ctx.lin_cache, ctx.sol)
        for i in 1:k
            data.bus_type[cbus[i], ts] == PSY.ACBusTypes.PQ &&
                (M[i, j] = -ctx.sol[2 * cbus[i] - 1])
        end
    end
    # In the round: a device with an effect of its own. Held: the step a device is fixed at.
    free = [abs(M[i, i]) * (hi[i] - lo[i]) >= CONTROL_GAIN_FLOOR for i in 1:k]
    # Every device in the round on its target and inside its (refreshed) range: finished.
    all(i -> !free[i] || (abs(e[i]) <= CONTROL_FINISH_VTOL && lo[i] <= devs[i].current <= hi[i]),
        1:k) && return false
    held = fill(NaN, k)
    Δb = zeros(k)
    target = [d.current for d in devs]
    for _ in 0:k
        fr = findall(free)
        isempty(fr) && break
        # One unknown per regulated bus among the free devices.
        buses = unique(cbus[fr])
        nb = length(buses)
        col = Dict(b => h for (h, b) in enumerate(buses))
        members = [fr[cbus[fr] .== b] for b in buses]
        A = zeros(nb, nb)
        r = zeros(nb)
        for (g, b) in enumerate(buses)
            ig = members[g][1]                  # equal rows: any member's
            r[g] = -sum(e[members[g]]) / length(members[g])
            for j in 1:k
                isnan(held[j]) || (r[g] -= M[ig, j] * held[j])
            end
            for j in fr
                A[g, col[cbus[j]]] += M[ig, j] / (length(members[col[cbus[j]]]) * M[j, j])
            end
        end
        F = LinearAlgebra.lu(A; check = false)
        LinearAlgebra.issuccess(F) || return nothing
        u = F \ r
        clamped = false
        for j in fr
            d = devs[j]
            h = col[cbus[j]]
            p = d.current + u[h] / (length(members[h]) * M[j, j])
            q = _band_clamp(d, clamp(p, lo[j], hi[j]), M[j, j])
            target[j] = q
            Δb[j] = q - d.current
            if q != p
                held[j] = Δb[j]
                free[j] = false
                clamped = true
            end
        end
        clamped || break
    end
    any_moved = false
    for (j, d) in enumerate(devs)
        idx = idxs[j]
        isnan(held[j]) && !free[j] && continue          # no effect of its own: not this round
        if !isnan(held[j]) && abs(Δb[j]) <= 1e-12 * (hi[j] - lo[j])
            active[idx] = false     # pinned at a limit (or its anchor): it stays there
            continue
        end
        p = target[j]
        # a step inside the range below what the solve resolves: not taken (a step onto a limit
        # is, so a railed device ends exactly on it)
        lo[j] < p < hi[j] && abs(Δb[j]) * data.bus_magnitude[d.bus_ix, ts]^2 <= ftol && continue
        prev[idx] = d.current
        apply_parameter!(d, data, p, ts)
        moved[idx] = true
        any_moved = true
    end
    return any_moved
end

function _continuous_finish!(
    pf, data, set::ControlledDeviceSet, ts::Int, snap::ControlStateSnapshot,
    frozen::Vector{Bool}, dVdp::Vector{Float64}, n_shared::Vector{Int}, n_taps::Int,
    ctx::Union{Nothing, _SensitivityContext} = nothing;
    kwargs...,
)
    n_shunts = length(set.shunts)
    off_facts = n_taps + n_shunts
    n = length(dVdp)
    active = fill(false, n)
    for (j, d) in enumerate(set.shunts)
        idx = n_taps + j
        active[idx] = !frozen[idx] && !isnothing(_continuous_target(d)) && !iszero(dVdp[idx])
    end
    for (j, _) in enumerate(set.facts)
        idx = off_facts + j
        active[idx] = !frozen[idx] && !iszero(dVdp[idx])
    end
    any(active) || return true
    prev = zeros(n)
    yprev = zeros(n)
    moved = fill(false, n)
    # Every solve of the finish converges to CONTROL_FINISH_SOLVE_TOL (or the caller's tighter
    # `tol`), so each round reads the voltage its steps produced (T152).
    ftol = min(Float64(get(kwargs, :tol, DEFAULT_NR_TOL)), CONTROL_FINISH_SOLVE_TOL)
    for _ in 1:CONTROL_FINISH_MAX_ROUNDS
        fill!(moved, false)
        _capture_state!(snap, data, ts)
        # The joint step on the sensitivity context at this round's state; the per-device secant
        # step without one (a non-polar formulation, or a singular Jacobian here).
        joint = !isnothing(ctx) && _refresh_sensitivity_context!(ctx, data, ts; kwargs...)
        any_moved = false
        if joint
            any_moved = _finish_joint_step!(data, set, ts, n_taps, active, prev, moved, ftol,
                ctx)
        end
        if isnothing(any_moved) || !joint
            any_moved = _finish_step_group!(set.shunts, n_taps, data, ts, active, dVdp,
                n_shared, prev, yprev, moved, ftol)
            any_moved |= _finish_step_group!(set.facts, off_facts, data, ts, active, dVdp,
                n_shared, prev, yprev, moved, ftol)
            joint = false
        end
        any_moved || return true
        if !_ctrl_solve!(pf, data, ts; kwargs..., tol = ftol)
            _finish_undo_group!(set.shunts, n_taps, data, ts, prev, moved)
            _finish_undo_group!(set.facts, off_facts, data, ts, prev, moved)
            _restore_state!(data, ts, snap)
            @warn "discrete control: the exact finish of a continuous device did not \
                converge; leaving it at its last converged setting (time step $ts)."
            return false
        end
        if !joint
            _finish_secant_group!(set.shunts, n_taps, data, ts, dVdp, prev, yprev, moved)
            _finish_secant_group!(set.facts, off_facts, data, ts, dVdp, prev, yprev, moved)
        end
    end
    return true
end

# The area-interchange ΔP column at time step `ts` (empty without enrolled areas). Every residual
# evaluation writes it, and `ControlStateSnapshot` does not hold it, so a failed solve can leave it
# changed; `_restore_one!` compares it before reusing a failed solve's outcome.
_restore_area_dp(data, ts::Int) =
    iszero(n_controlled_areas(data)) ? Float64[] :
    data.area_interchange.delta_p[:, ts]

# Incremental λ-restore of one device from its snapped value toward the
# continuous value (used only if snapping made the system infeasible).  First
# probe is a SMALL step, matching `_continuation_to!`.
#
# `failed` carries, across one restore walk, the start of the walk's last failed solve when
# `data` is still exactly there: the same voltages, bus types, injections and remote-regulation
# status (the failed solve's state was restored from the snapshot taken just before it), no
# parameter moved since, and the same area ΔP (held in `failed[]`). A device with no parameter
# to walk would solve again from that start, which is the same computation and fails the same
# way, leaving `data` as it found it; so it returns `false` without solving. On a large case
# whose snapped network does not solve for a reason no one device explains, the walk otherwise
# runs one capped solve per enrolled device (thousands). `nothing` (the default): no reuse.
function _restore_one!(
    d, data, ts::Int, continuous::Float64, pf, snap::ControlStateSnapshot,
    failed::Base.RefValue{Union{Nothing, Vector{Float64}}} =
        Ref{Union{Nothing, Vector{Float64}}}(nothing);
    kwargs...,
)::Bool
    snapped = current_parameter(d)
    if abs(continuous - snapped) < _param_tol(d)
        dp = _restore_area_dp(data, ts)
        failed[] == dp && return false
        # No parameter to walk, but the solve can still fail (another device's snap
        # made the network infeasible): protect it like every other solve path so a
        # diverged iterate never leaks into the next device's "last converged" snapshot.
        _capture_state!(snap, data, ts)
        ok = _ctrl_solve!(pf, data, ts; kwargs...)
        ok || _restore_state!(data, ts, snap)
        failed[] = ok ? nothing : dp
        return ok
    end
    # A parameter moves below (even a reverted move is a Y-bus delta update): no reuse after it.
    failed[] = nothing
    lo, hi = parameter_limits(d)
    _capture_state!(snap, data, ts)  # last converged state, restored on a failed trial
    # Full move first, matching `_continuation_to!`.
    apply_parameter!(d, data, clamp(continuous, lo, hi), ts)
    _ctrl_solve!(pf, data, ts; kwargs...) && return true
    apply_parameter!(d, data, snapped, ts)
    _restore_state!(data, ts, snap)
    _, completed =
        _bisection_walk!(d, data, ts, snapped, continuous, lo, hi, pf, snap; kwargs...)
    completed || return false
    return _ctrl_solve!(pf, data, ts; kwargs...)
end

# Snap a concrete device vector onto its discrete grid (continuous devices clamp),
# returning the pre-snap continuous values index-aligned with `devices` for the restore
# path. (Index alignment, not name keying: device names are only unique per concrete
# type, and a cross-family collision must not cross the stashed values.)
_snap_with_gain(d::BranchDevice, p::Float64, g) = snap_to_discrete(d, p, g)
_snap_with_gain(d, p::Float64, _) = snap_to_discrete(d, p)

# `dVdp` (indexed [taps; shunts; facts; phase shifters], `goffset` this group's offset) orients
# the snap of a latched tap still at its anchor (see `snap_to_discrete`).
function _snap_device_group!(devices, data, ts::Int, dVdp = nothing, goffset::Int = 0)
    cont = Vector{Float64}(undef, length(devices))
    for (i, d) in enumerate(devices)
        cont[i] = d.current
        if abs(d.current - d.initial) < _param_tol(d)
            # Never moved (deadband-held or already regulated at enrollment): hold the
            # original setting. PSS/E keeps an in-band shunt at BINIT and an untouched
            # tap at its parsed ratio even when that value is off the discrete grid. The
            # setting is restored exactly: a band latch released within `_param_tol` of its
            # anchor leaves a sub-tolerance drift (up to ~2e-5 in ratio) that is not a move.
            d.current == d.initial || apply_parameter!(d, data, d.initial, ts)
            continue
        end
        g = _settle_gain(dVdp, goffset + i)
        apply_parameter!(d, data, _snap_with_gain(d, d.current, g), ts)
    end
    return cont
end

# λ-restore a concrete device vector toward its stashed continuous value; returns `true`
# iff all restored to a converged state. `failed`: see `_restore_one!`. Past the control time
# limit the walk stops: the result is `false` whatever the remaining devices would do.
function _restore_device_group!(
    devices, data, ts::Int, cont::Vector{Float64}, pf,
    scratch_snap::ControlStateSnapshot,
    failed::Base.RefValue{Union{Nothing, Vector{Float64}}} =
        Ref{Union{Nothing, Vector{Float64}}}(nothing);
    kwargs...,
)::Bool
    ok = true
    set = data.controlled_devices
    for (i, d) in enumerate(devices)
        !isnothing(set) && _control_time_up!(set) && return false
        ok &= _restore_one!(d, data, ts, cont[i], pf, scratch_snap, failed; kwargs...)
    end
    return ok
end

# PSS/E's "first step inside the band" for banded taps and discrete switched shunts, after the
# snap's converged solve. The continuation leaves a latched device with its voltage on the edge
# only to within the sigmoid law's fixed-point offset (≲1e-3 p.u. at full steepness; a tap step
# is ~0.006 p.u.), and devices couple, so the snapped position is checked on the solved network:
# 1. forward: a device still outside its band on the latched side steps one more level in the
#    direction it moved, until inside or at a limit;
# 2. back-off: a device one level back toward its anchor that would still be inside steps back,
#    so it stops at the first position inside rather than one beyond;
# 3. forward again, for any voltage the back-off moved out through coupling;
# and 2–3 again (at most `CONTROL_BAND_SETTLE_SWEEPS` times) while a sweep moved anything, since
# a forward step can lift a neighbour that could not back off before.
# Each round applies its steps together (taps, then shunts, indexed [taps; shunts]) and solves
# once. A failed solve rolls the round back and ends the walk at the last converged state, with
# a warning.
function _band_settle!(
    pf, data, set::ControlledDeviceSet, ts::Int, snap::ControlStateSnapshot,
    dVdp::Union{Nothing, Vector{Float64}} = nothing,
    frozen::Union{Nothing, Vector{Bool}} = nothing; kwargs...,
)
    # A held device (side 0) that the snap's solve left outside its band latches there, at its
    # position, so the forward walk steps it (GAP-068): PSS/E moves any banded device whose
    # voltage is outside at the end. Counted before the early return below.
    _latch_pushed_out!(set, data, ts, frozen)
    (any(_band_latched, set.taps) || any(_band_latched, set.shunts) ||
     any(_band_latched, set.phase_shifters)) || return
    n = length(set.taps) + length(set.shunts) + length(set.phase_shifters)
    prev = zeros(n)
    moved = fill(false, n)
    rounds = max(
        maximum(d -> length(d.levels), set.taps; init = 0),
        maximum(d -> length(d.levels), set.shunts; init = 0),
        maximum(d -> length(d.levels), set.phase_shifters; init = 0),
    )
    _band_forward!(pf, data, set, ts, snap, prev, moved, rounds, dVdp; kwargs...) || return
    positions() = vcat(Float64[d.current for d in set.taps],
        Float64[d.current for d in set.shunts],
        Float64[d.current for d in set.phase_shifters])
    for _ in 1:CONTROL_BAND_SETTLE_SWEEPS
        before = positions()
        _band_back_off!(pf, data, set, ts, snap, prev, moved, rounds, dVdp; kwargs...) ||
            return
        _band_forward!(pf, data, set, ts, snap, prev, moved, rounds, dVdp; kwargs...) ||
            return
        positions() == before && return
    end
    return
end

_band_latched(d::ControlledTap) = d.band && d.side != 0
_band_latched(d::ControlledPhaseShifter) = d.band && d.side != 0 && !isempty(d.levels)
_band_latched(d::ControlledSwitchedShunt) = d.band && d.side != 0 && !d.continuous
_band_direction(d::BandedDevice) = Int(sign(d.current - d.anchor))

# The direction that moves a latched device's regulated quantity back toward its band, as
# `_can_correct` reads it: a discrete shunt raises its voltage by stepping up; a tap or phase
# shifter steps −side·sign(g), g the continuation's (oriented) gain. Without a gain, the
# direction it moved from its anchor.
_correcting_direction(d::ControlledSwitchedShunt, ::Union{Nothing, Float64}) = -d.side
function _correcting_direction(d::BranchDevice, g::Union{Nothing, Float64})
    (isnothing(g) || iszero(g)) && return _band_direction(d)
    return -d.side * Int(sign(g))
end

# Latch every held, unfrozen banded device of a group whose regulated quantity is outside its
# band: side ±1, anchor at its position. `frozen` is indexed as `dVdp`.
function _latch_pushed_out_group!(devices, goffset::Int, data, ts::Int, frozen)
    for (i, d) in enumerate(devices)
        (_band_capable(d) && d.side == 0) || continue
        !isnothing(frozen) && frozen[goffset + i] && continue
        y = measured_value(d, data, ts)
        lo, hi = band_edges(d)
        if y < lo - CONTROL_TAP_BAND_TOL
            d.side, d.anchor = -1, d.current
        elseif y > hi + CONTROL_TAP_BAND_TOL
            d.side, d.anchor = 1, d.current
        end
    end
    return
end
function _latch_pushed_out!(set::ControlledDeviceSet, data, ts::Int, frozen)
    _latch_pushed_out_group!(set.taps, 0, data, ts, frozen)
    _latch_pushed_out_group!(set.shunts, length(set.taps), data, ts, frozen)
    _latch_pushed_out_group!(set.phase_shifters, _gain_pst_offset(set), data, ts, frozen)
    return
end

# Move a banded device to level `lv` (a shunt also records that level's blocks).
function _band_apply!(d::BandedDevice, data, lv::Float64, ts::Int)
    apply_parameter!(d, data, lv, ts)
    _sync_blocks!(d)
    return
end

function _band_undo_group!(devices, offset::Int, data, ts::Int, prev, moved)
    for (i, d) in enumerate(devices)
        moved[offset + i] && _band_apply!(d, data, prev[offset + i], ts)
    end
    return
end

# The band settle indexes its own scratch [taps; shunts; phase shifters]; the continuation's gains
# `dVdp` are indexed [taps; shunts; facts; phase shifters].
_settle_pst_offset(set) = length(set.taps) + length(set.shunts)
_gain_pst_offset(set) = length(set.taps) + length(set.shunts) + length(set.facts)

function _band_round_failed!(data, set, ts, snap, prev, moved)
    _band_undo_group!(set.taps, 0, data, ts, prev, moved)
    _band_undo_group!(set.shunts, length(set.taps), data, ts, prev, moved)
    _band_undo_group!(set.phase_shifters, _settle_pst_offset(set), data, ts, prev, moved)
    _restore_state!(data, ts, snap)
    @warn "discrete control: a banded device's step toward the first position inside its \
        voltage band did not converge; leaving the devices at their last converged \
        positions (time step $ts)."
    return false
end

# One forward step of every latched device of a group still outside its band, in the direction
# that corrects it (`_correcting_direction`; GAP-068: stepping away from the anchor walked a tap
# snapped behind its anchor away from its band, to its limit). No step when no level lies that
# way: a railed device stays, as PSS/E leaves it. `goffset` indexes `dVdp`.
function _band_forward_group!(devices, offset::Int, data, ts::Int, prev, moved,
    dVdp = nothing, goffset::Int = offset)
    # Measured before any step is applied (see `_apply_targets_group!`).
    outside = [_band_latched(d) && _outside_latched_side(d, measured_value(d, data, ts))
               for d in devices]
    for (i, d) in enumerate(devices)
        outside[i] || continue
        dir = _correcting_direction(d, _settle_gain(dVdp, goffset + i))
        iszero(dir) && continue
        nxt = _next_level(d, dir)
        isnothing(nxt) && continue
        prev[offset + i] = d.current
        _band_apply!(d, data, nxt, ts)
        moved[offset + i] = true
    end
    return
end

function _band_forward!(pf, data, set, ts, snap, prev, moved, rounds, dVdp = nothing;
    kwargs...)
    for _ in 1:rounds
        fill!(moved, false)
        _capture_state!(snap, data, ts)
        _band_forward_group!(set.taps, 0, data, ts, prev, moved, dVdp, 0)
        _band_forward_group!(set.shunts, length(set.taps), data, ts, prev, moved, dVdp,
            length(set.taps))
        _band_forward_group!(set.phase_shifters, _settle_pst_offset(set), data, ts, prev,
            moved, dVdp, _gain_pst_offset(set))
        any(moved) || return true
        _ctrl_solve!(pf, data, ts; kwargs...) ||
            return _band_round_failed!(data, set, ts, snap, prev, moved)
    end
    return true
end

# One step back toward the anchor for every still-active device of a group; a device with no
# level left before its anchor drops out.
function _band_back_group!(devices, offset::Int, data, ts::Int, prev, moved, active)
    for (i, d) in enumerate(devices)
        idx = offset + i
        active[idx] || continue
        dir = _band_direction(d)
        back = _next_level(d, -dir)
        tol = _param_tol(d)
        # Never back past the anchor: the device latched there with its voltage outside. A level
        # within `_param_tol` of the anchor is the anchor (a stored ratio a few 1e-6 off its level
        # by raw rounding, GAP-068), and the step lands on the anchor itself.
        if iszero(dir) || isnothing(back) || dir * (back - d.anchor) < -tol
            active[idx] = false
            continue
        end
        abs(back - d.anchor) < tol && (back = d.anchor)
        prev[idx] = d.current
        _band_apply!(d, data, back, ts)
        moved[idx] = true
    end
    return
end

# A device whose voltage left the band on the step back is returned to where it was and backed
# off no further in the joint rounds. When it stepped back alone it was at its first position
# inside; when others stepped back with it, their steps may have pushed it out, so it is marked
# for a solo retry (`retry`). Returns whether any device reverted.
function _band_revert_group!(devices, offset::Int, data, ts::Int, prev, moved, active,
    retry, solo::Bool)
    reverted = false
    # Measured before any device is returned (see `_apply_targets_group!`).
    outside = [moved[offset + i] && _outside_latched_side(d, measured_value(d, data, ts))
               for (i, d) in enumerate(devices)]
    for (i, d) in enumerate(devices)
        idx = offset + i
        outside[i] || continue
        _band_apply!(d, data, prev[idx], ts)
        active[idx] = false
        retry[idx] = !solo
        reverted = true
    end
    return reverted
end

# Whether a latched device has a level left in the direction that moves its voltage back toward
# its band. A discrete shunt raises its voltage by stepping up (dV/db > 0), so one below its band
# needs a level above; a tap's direction is −side·sign(dV/dtap), from the continuation's gain
# `g` (without one, `nothing` or 0, it counts as able, the rule before). A device railed with no
# such level stays outside its band whatever the others do; PSS/E accepts it there.
_can_correct(d::ControlledSwitchedShunt, ::Union{Nothing, Float64}) =
    !isnothing(_next_level(d, -d.side))
function _can_correct(d::BranchDevice, g::Union{Nothing, Float64})
    (isnothing(g) || iszero(g)) && return true
    return !isnothing(_next_level(d, -d.side * Int(sign(g))))
end
_settle_gain(dVdp::Nothing, ::Int) = nothing
_settle_gain(dVdp::Vector{Float64}, idx::Int) = dVdp[idx]

# Whether a latched device is outside its band on its side and could step back toward it.
_correctable_outside(d, data, ts::Int, g) =
    _band_latched(d) && _outside_latched_side(d, measured_value(d, data, ts)) &&
    _can_correct(d, g)

# Whether any latched banded device (tap or discrete shunt) that can still correct is outside its
# band on its side (GAP-030: a railed reactor 1.4e-4 below its band, with no level left upward,
# used to veto every other device's back-off). `dVdp` is indexed [taps; shunts; …].
function _any_latched_outside(set::ControlledDeviceSet, data, ts::Int, dVdp = nothing)
    for (i, d) in enumerate(set.taps)
        _correctable_outside(d, data, ts, _settle_gain(dVdp, i)) && return true
    end
    n_taps = length(set.taps)
    for (i, d) in enumerate(set.shunts)
        _correctable_outside(d, data, ts, _settle_gain(dVdp, n_taps + i)) && return true
    end
    off = _gain_pst_offset(set)
    for (i, d) in enumerate(set.phase_shifters)
        _correctable_outside(d, data, ts, _settle_gain(dVdp, off + i)) && return true
    end
    return false
end

# A unit of a parallel phase-shifter group regulates the group's mean flow and steps only with
# the group (the joint rounds), never alone: alone it would split the units by a step.
_solo_steps(::AbstractControlledDevice) = true
_solo_steps(d::ControlledPhaseShifter) = isempty(d.group)

# Solo back-off of one group's devices marked for a retry: each steps back alone, with one
# solve, and keeps the step while every latched device stays inside its band; otherwise it
# returns (the state is restored, no extra solve) and is done. Returns whether any device
# stepped back.
function _band_solo_group!(pf, data, set, devices, offset::Int, ts::Int, snap, retry,
    dVdp = nothing; kwargs...)
    stepped = false
    for (i, d) in enumerate(devices)
        idx = offset + i
        retry[idx] || continue
        if !_solo_steps(d)
            retry[idx] = false
            continue
        end
        dir = _band_direction(d)
        back = _next_level(d, -dir)
        tol = _param_tol(d)
        if iszero(dir) || isnothing(back) || dir * (back - d.anchor) < -tol
            retry[idx] = false
            continue
        end
        abs(back - d.anchor) < tol && (back = d.anchor)
        p0 = d.current
        _capture_state!(snap, data, ts)
        _band_apply!(d, data, back, ts)
        if _ctrl_solve!(pf, data, ts; kwargs...) &&
           !_any_latched_outside(set, data, ts, dVdp)
            stepped = true
            continue
        end
        _band_apply!(d, data, p0, ts)
        _restore_state!(data, ts, snap)
        retry[idx] = false
    end
    return stepped
end

function _unmoved_pushed_out(set::ControlledDeviceSet, data, ts::Int, moved, dVdp = nothing)
    for (i, d) in enumerate(set.taps)
        moved[i] && continue
        _correctable_outside(d, data, ts, _settle_gain(dVdp, i)) && return true
    end
    n_taps = length(set.taps)
    for (i, d) in enumerate(set.shunts)
        moved[n_taps + i] && continue
        _correctable_outside(d, data, ts, _settle_gain(dVdp, n_taps + i)) && return true
    end
    off, goff = _settle_pst_offset(set), _gain_pst_offset(set)
    for (i, d) in enumerate(set.phase_shifters)
        moved[off + i] && continue
        _correctable_outside(d, data, ts, _settle_gain(dVdp, goff + i)) && return true
    end
    return false
end

function _band_back_off!(pf, data, set, ts, snap, prev, moved, rounds, dVdp = nothing;
    kwargs...)
    n_taps = length(set.taps)
    off_p = _settle_pst_offset(set)
    active = vcat(
        [_band_latched(d) && !iszero(_band_direction(d)) for d in set.taps],
        [_band_latched(d) && !iszero(_band_direction(d)) for d in set.shunts],
        [_band_latched(d) && !iszero(_band_direction(d)) for d in set.phase_shifters],
    )
    retry = fill(false, length(active))
    for _ in 1:rounds
        fill!(moved, false)
        _capture_state!(snap, data, ts)
        _band_back_group!(set.taps, 0, data, ts, prev, moved, active)
        _band_back_group!(set.shunts, n_taps, data, ts, prev, moved, active)
        _band_back_group!(set.phase_shifters, off_p, data, ts, prev, moved, active)
        any(moved) || break
        _ctrl_solve!(pf, data, ts; kwargs...) ||
            return _band_round_failed!(data, set, ts, snap, prev, moved)
        if _unmoved_pushed_out(set, data, ts, moved, dVdp)
            # The joint step pushed a device that did not move out of its band: undo the round
            # (the captured state is exact) and leave its devices to the solo retries.
            _band_undo_group!(set.taps, 0, data, ts, prev, moved)
            _band_undo_group!(set.shunts, n_taps, data, ts, prev, moved)
            _band_undo_group!(set.phase_shifters, off_p, data, ts, prev, moved)
            _restore_state!(data, ts, snap)
            for i in eachindex(moved)
                moved[i] || continue
                active[i] = false
                retry[i] = true
            end
            continue
        end
        solo = count(moved) == 1
        reverted = _band_revert_group!(set.taps, 0, data, ts, prev, moved, active, retry,
            solo)
        reverted |= _band_revert_group!(set.shunts, n_taps, data, ts, prev, moved, active,
            retry, solo)
        reverted |= _band_revert_group!(set.phase_shifters, off_p, data, ts, prev, moved,
            active, retry, solo)
        if reverted && !_ctrl_solve!(pf, data, ts; kwargs...)
            return _band_round_failed!(data, set, ts, snap, prev, moved)
        end
    end
    # Coupled devices that stepped back together and pushed one another out: back each off
    # alone, so every device ends at its first position inside given where the others are.
    for _ in 1:rounds
        any(retry) || break
        stepped = _band_solo_group!(pf, data, set, set.taps, 0, ts, snap, retry, dVdp;
            kwargs...)
        stepped |= _band_solo_group!(pf, data, set, set.shunts, n_taps, ts, snap, retry,
            dVdp; kwargs...)
        stepped |= _band_solo_group!(pf, data, set, set.phase_shifters, off_p, ts, snap,
            retry, dVdp; kwargs...)
        stepped || break
    end
    return true
end

function snap_and_restore!(
    pf,
    data,
    set::ControlledDeviceSet,
    ts::Int,
    scratch_snap::ControlStateSnapshot,
    dVdp::Union{Nothing, Vector{Float64}} = nothing,
    frozen::Union{Nothing, Vector{Bool}} = nothing,
    before_settle = nothing;
    kwargs...,
)::Bool
    # Last continuous converged state: the restore path must start from here, not from
    # whatever diverged iterate a failed post-snap solve leaves in `data`. Needs its OWN
    # buffer because it stays live across the `_restore_one!` calls below, which reuse
    # `scratch_snap`.
    pre = _snapshot_state(data, ts)
    n_taps, n_shunts = length(set.taps), length(set.shunts)
    cont_taps = _snap_device_group!(set.taps, data, ts, dVdp, 0)
    cont_shunts = _snap_device_group!(set.shunts, data, ts, dVdp, n_taps)
    cont_facts = _snap_device_group!(set.facts, data, ts, dVdp, n_taps + n_shunts)
    cont_psts = _snap_device_group!(set.phase_shifters, data, ts, dVdp, _gain_pst_offset(set))
    if _ctrl_solve!(pf, data, ts; kwargs...)
        # `before_settle()`: the continuation's exact finish of its continuous devices (T161).
        isnothing(before_settle) || before_settle()
        _band_settle!(pf, data, set, ts, scratch_snap, dVdp, frozen; kwargs...)
        return true
    end
    _restore_state!(data, ts, pre)
    # One record of the walk's last failed start, across the four families (`_restore_one!`).
    failed = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    ok = _restore_device_group!(
        set.taps, data, ts, cont_taps, pf, scratch_snap, failed; kwargs...)
    ok &= _restore_device_group!(
        set.shunts, data, ts, cont_shunts, pf, scratch_snap, failed; kwargs...)
    ok &= _restore_device_group!(
        set.facts, data, ts, cont_facts, pf, scratch_snap, failed; kwargs...)
    ok &= _restore_device_group!(
        set.phase_shifters, data, ts, cont_psts, pf, scratch_snap, failed; kwargs...)
    if !ok
        names = join(
            vcat(
                [d.name for d in set.taps],
                [d.name for d in set.shunts],
                [d.name for d in set.facts],
                [d.name for d in set.phase_shifters],
            ),
            ", ",
        )
        @error "discrete control could not restore feasibility after \
            snapping; devices: $names (time step $ts)"
        data.converged[ts] = false
        return false
    end
    return true
end

"""
    get_controlled_device_results(data) -> DataFrames.DataFrame

Solved discrete-control device settings: one row per enrolled device per time step, with
its family, name, time step, control band, enrollment-time (`initial`) and solved
(`final`) parameter for that step. Every family (including taps) reports its own
per-time-step state. For a single-time-step solve (`time_steps == 1`), the solved
settings are also written back to
the `PSY.System` by [`solve_and_store_power_flow!`](@ref) under active controls, and
applied to PSS/E exports by [`update_exporter!`](@ref) — see
[`write_device_settings!`](@ref). For `time_steps > 1`, a PSY component cannot hold a
per-time-step schedule, so this DataFrame is the only place the full per-step results
are available. Returns an empty frame when the data was built without discrete control.
"""
function get_controlled_device_results(data)
    family = String[]
    name = String[]
    time_step = Int[]
    lower = Float64[]
    upper = Float64[]
    initial = Float64[]
    final = Float64[]
    delivered_q_mvar = Union{Float64, Missing}[]
    saturated = Bool[]
    set = get_controlled_devices(data)
    if !isnothing(set)
        for ts in 1:get_time_steps(data)
            for (fam, devices) in (
                ("TapTransformer", set.taps),
                ("SwitchedAdmittance", set.shunts),
                ("FACTSControlDevice", set.facts),
                ("PhaseShiftingTransformer", set.phase_shifters),
            )
                for (i, d) in enumerate(devices)
                    lo, hi = stored_parameter_limits(set, d, i, ts)
                    push!(family, fam)
                    push!(name, d.name)
                    push!(time_step, ts)
                    push!(lower, lo)
                    push!(upper, hi)
                    push!(initial, d.initial)
                    push!(final, stored_current(set, d, i, ts))
                    push!(delivered_q_mvar, stored_delivered_q_mvar(set, d, i, data, ts))
                    push!(saturated, stored_saturated(set, d, i, ts))
                end
            end
        end
    end
    return DataFrames.DataFrame(
        "family" => family,
        "name" => name,
        "time_step" => time_step,
        "lower_limit" => lower,
        "upper_limit" => upper,
        "initial" => initial,
        "final" => final,
        "delivered_q_mvar" => delivered_q_mvar,
        "saturated" => saturated,
    )
end
