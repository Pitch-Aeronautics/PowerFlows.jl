"""Cache for non-linear methods.

# Fields
- `x::Vector{Float64}`: the current state vector.
- `r::Vector{Float64}`: the current residual.
- `Δx_nr::Vector{Float64}`: the step under the Newton-Raphson method.
The remainder of the fields are only used in the `TrustRegionACPowerFlow`:
- `r_predict::Vector{Float64}`: the predicted residual at `x+Δx_proposed`,
    under a linear approximation: i.e `J_x⋅(x+Δx_proposed)`.
- `Δx_proposed::Vector{Float64}`: the suggested step `Δx`, selected among `Δx_nr`,
    `Δx_cauchy`, and the dogleg interpolation between the two. The first is chosen when
    `x+Δx_nr` is inside the trust region, the second when both `x+Δx_cauchy`
    and `x+Δx_nr` are outside the trust region, and the third when `x+Δx_cauchy`
    is inside and `x+Δx_nr` outside. The dogleg step selects the point where the line
    from `x+Δx_cauchy` to `x+Δx_nr` crosses the boundary of the trust region.
- `Δx_cauchy::Vector{Float64}`: the step to the Cauchy point if the Cauchy point
    lies within the trust region, otherwise a step in that direction."""
struct StateVectorCache
    x::Vector{Float64}
    r::Vector{Float64} # residual
    r_predict::Vector{Float64} # predicted residual
    Δx_proposed::Vector{Float64} # proposed Δx: Cauchy, NR, or dogleg step.
    Δx_cauchy::Vector{Float64} # Cauchy step
    Δx_nr::Vector{Float64} # Newton-Raphson step
    d::Vector{Float64}
    # Persistent regularized singular-Jacobian fallback `-(JᵀJ + λI)`, reused across repeated
    # fallbacks: `fallback_matrix` keeps a fixed pattern so values are refreshed in place and
    # `fallback_cache`'s symbolic factorization is reused; both are rebuilt only on a pattern shift.
    fallback_cache::Base.RefValue{
        Union{Nothing, PNM.KLULinSolveCache{Float64, J_INDEX_TYPE}},
    }
    fallback_matrix::Base.RefValue{Union{Nothing, SparseMatrixCSC{Float64, J_INDEX_TYPE}}}
end

function StateVectorCache(x0::Vector{Float64}, f0::Vector{Float64})
    x = copy(x0)
    r = copy(f0)
    r_predict = copy(x0)
    Δx_proposed = copy(x0)
    Δx_cauchy = copy(x0)
    Δx_nr = copy(x0)
    return StateVectorCache(
        x, r, r_predict, Δx_proposed, Δx_cauchy, Δx_nr, ones(size(x0)),
        Base.RefValue{Union{Nothing, PNM.KLULinSolveCache{Float64, J_INDEX_TYPE}}}(nothing),
        Base.RefValue{Union{Nothing, SparseMatrixCSC{Float64, J_INDEX_TYPE}}}(nothing),
    )
end

"""Solve for the Newton-Raphson step, given the factorization object for `J.Jv`
(if non-singular) or its stand-in (if singular)."""
function _solve_Δx_nr!(stateVector::StateVectorCache, cache::PFLinearSolverCache)
    copyto!(stateVector.Δx_nr, stateVector.r)
    solve!(cache, stateVector.Δx_nr)
    return
end

"""Compute the relative residual `‖A·Δx_nr − r‖₁ / ‖r‖₁` of the linear solve and, if it
exceeds `refinement_threshold`, run iterative refinement and recompute it. Returns the
(post-refinement) relative residual; the caller uses it as a backend-agnostic singularity
signal (see [`_set_Δx_nr!`](@ref))."""
function _do_refinement!(stateVector::StateVectorCache,
    A::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    cache::PFLinearSolverCache,
    refinement_threshold::Float64,
    refinement_eps::Float64,
)
    # use stateVector.r_predict as temporary buffer.
    δ_temp = stateVector.r_predict
    r_norm = norm(stateVector.r, 1)
    # A zero residual is an exact (already-converged) solve, not a singular Jacobian. Return a
    # zero relative residual rather than dividing 0/0 into a NaN, which the caller's
    # `!isfinite(residual)` guard would otherwise misread as a singularity.
    iszero(r_norm) && return 0.0
    mul!(δ_temp, A, stateVector.Δx_nr)
    δ_temp .-= stateVector.r
    delta = norm(δ_temp, 1) / r_norm
    if delta > refinement_threshold
        stateVector.Δx_nr .= solve_w_refinement(cache,
            A,
            stateVector.r,
            refinement_eps)
        mul!(δ_temp, A, stateVector.Δx_nr)
        δ_temp .-= stateVector.r
        delta = norm(δ_temp, 1) / r_norm
    end
    return delta
end

"""Sets the Newton-Raphson step. Usually, this is just `J.Jv \\ stateVector.r`, but
`J.Jv` might be singular."""
function _set_Δx_nr!(stateVector::StateVectorCache,
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    linSolveCache::PFLinearSolverCache,
    solver::ACPowerFlowSolverType,
    refinement_threshold::Float64,
    refinement_eps::Float64)
    use_fallback = false
    _count_numeric_refactor!(J.data)
    try
        numeric_refactor!(linSolveCache, J.Jv)
    catch e
        # KLU signals a singular factorization by throwing a `SingularException`;
        # AppleAccelerate and MKLPardiso do not (the residual guard below catches their
        # silent garbage solves). Only a `SingularException` routes to the regularized
        # fallback. Any other exception (dimension mismatch, allocation failure, an MKL
        # error) is a genuine solver failure, not a singular Jacobian — rethrow it rather
        # than masking it behind the "Jacobian is singular" warning.
        e isa LinearAlgebra.SingularException || rethrow()
        use_fallback = true
    end

    if !use_fallback
        _solve_Δx_nr!(stateVector, linSolveCache)
        # Backend-agnostic singular-Jacobian guard. KLU throws on a singular matrix (caught
        # above), but AppleAccelerate and MKLPardiso silently return a finite garbage
        # solution. `_do_refinement!` returns the relative residual ‖J·Δx − r‖/‖r‖ (after
        # attempting iterative refinement, which rescues merely ill-conditioned solves). If
        # the linear solve still cannot be driven below `refinement_threshold`, the Jacobian
        # is (numerically) singular regardless of backend.
        residual = _do_refinement!(
            stateVector,
            J.Jv,
            linSolveCache,
            refinement_threshold,
            refinement_eps,
        )
        use_fallback = !isfinite(residual) || residual > refinement_threshold
    end

    if use_fallback
        @warn("$solver hit a point where the Jacobian is singular.")
        # KLU is used because the fallback must reliably solve the regularized system. Refresh
        # values in place while the pattern holds (reusing the factorization); rebuild if it shifts.
        M_prev = stateVector.fallback_matrix[]
        cache_prev = stateVector.fallback_cache[]
        if !isnothing(M_prev) && !isnothing(cache_prev) &&
           _refresh_singular_J_fallback!(M_prev, J.Jv, stateVector.x)
            M = M_prev
            cache = cache_prev
            numeric_refactor!(cache, M)
        else
            M = _build_singular_J_fallback(J.Jv, stateVector.x)
            cache = make_linear_solver_cache(PNM.KLUSolver(), M)
            full_factor!(cache, M)
            stateVector.fallback_matrix[] = M
            stateVector.fallback_cache[] = cache
        end
        _solve_Δx_nr!(stateVector, cache)
        _do_refinement!(stateVector, M, cache, refinement_threshold, refinement_eps)
    end
    LinearAlgebra.rmul!(stateVector.Δx_nr, -1.0)
    return
end

"""Returns a freshly-allocated stand-in matrix `-(JᵀJ + λI)` for a singular `J`. The result
defines the sparsity pattern that [`_refresh_singular_J_fallback!`](@ref) reuses in place."""
function _build_singular_J_fallback(Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    x::Vector{Float64})
    fjac2 = Jv' * Jv
    lambda = NR_SINGULAR_SCALING * sqrt(length(x) * eps()) * norm(fjac2, 1)
    return -(fjac2 + lambda * LinearAlgebra.I)
end

"""Refresh `M = -(JᵀJ + λI)` in place (λ as in [`_build_singular_J_fallback`](@ref)). Returns
`false` without touching `M` when the `JᵀJ` pattern no longer matches `M`'s, so the caller rebuilds."""
function _refresh_singular_J_fallback!(M::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    x::Vector{Float64})
    fjac2 = Jv' * Jv
    (fjac2.colptr == M.colptr && fjac2.rowval == M.rowval) || return false
    lambda = NR_SINGULAR_SCALING * sqrt(length(x) * eps()) * norm(fjac2, 1)
    Mnz = M.nzval
    Fnz = fjac2.nzval
    @inbounds for col in 1:size(M, 2)
        for p in M.colptr[col]:(M.colptr[col + 1] - 1)
            if M.rowval[p] == col
                Mnz[p] = -(Fnz[p] + lambda)
            else
                Mnz[p] = -Fnz[p]
            end
        end
    end
    return true
end

"""Sets `Δx_proposed` equal to the `Δx` by which we should update `x`. Decides
between the Cauchy step `Δx_cauchy`, Newton-Raphson step `Δx_nr`, and the dogleg
interpolation between the two, based on which fall within the trust region."""
function _dogleg!(Δx_proposed::Vector{Float64},
    Δx_cauchy::Vector{Float64},
    Δx_nr::Vector{Float64},
    r::Vector{Float64},
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    d::Vector{Float64},
    delta::Float64,
    Jg::Vector{Float64},
)
    nr_norm = wnorm(d, Δx_nr)
    @debug "Trust region: ||Δx_nr|| = $(siground(nr_norm)), δ = $(siground(delta))"

    if nr_norm <= delta
        copyto!(Δx_proposed, Δx_nr) # update Δx_proposed: newton-raphson case.
        @debug "Newton-Raphson step selected (inside trust region)"
    else
        # using Δx_proposed as a temporary buffer: alias to g for readability
        g = Δx_proposed
        LinearAlgebra.mul!(g, Jv', r)
        g .= g ./ d .^ 2
        LinearAlgebra.mul!(Jg, Jv, g)
        Δx_cauchy .= -wnorm(d, g)^2 / sum(abs2, Jg) .* g # Cauchy point

        cauchy_norm = wnorm(d, Δx_cauchy)
        @debug "Cauchy point: ||Δx_cauchy|| = $(siground(cauchy_norm))"

        if cauchy_norm >= delta
            # Δx_cauchy outside region => take step of length delta in direction of -g.
            LinearAlgebra.rmul!(g, -delta / wnorm(d, g))
            @debug "Cauchy step selected (truncated to trust region boundary)"
            # not needed because g is already an alias for Δx_proposed.
            # copyto!(Δx_proposed, g) # update Δx_proposed: cauchy point case
        else
            # Δx_cauchy inside region => next point is the spot where the line from
            # Δx_cauchy to Δx_nr crosses the boundary of the trust region.
            # this is the "dogleg" part.

            # using Δx_nr as temporary buffer: alias to Δx_diff for readability.
            Δx_nr .-= Δx_cauchy
            Δx_diff = Δx_nr

            b = wdot(d, Δx_cauchy, d, Δx_diff)
            a = wnorm(d, Δx_diff)^2
            tau = (-b + sqrt(b^2 - 4a * (wnorm(d, Δx_cauchy)^2 - delta^2))) / (2a)
            Δx_cauchy .+= tau .* Δx_diff
            copyto!(Δx_proposed, Δx_cauchy) # update Δx_proposed: dogleg case.
            @debug "Dogleg step selected (τ = $(siground(tau)))"
        end
    end
    return
end

"""Accept a trust region step: update cached residual and autoscale vector `d`.
The caller is responsible for recomputing the Jacobian via `J(time_step)` before
calling this, so that `Jv` reflects the new state."""
function _accept_trust_region_step!(
    stateVector::StateVectorCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    autoscale::Bool,
)
    stateVector.r .= residual.Rv
    if autoscale
        for i in 1:length(stateVector.x)
            stateVector.d[i] = max(0.1 * stateVector.d[i], norm(view(Jv, :, i)))
        end
    end
    return
end

"""Attempt Iwamoto damping on a rejected trust region step.

Uses the already-evaluated trial-point residual to compute an optimal damped step.
Returns `true` if the damped step was accepted, `false` if reverted."""
function _iwamoto_fallback!(
    time_step::Int,
    stateVector::StateVectorCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    old_residual::Vector{Float64},
    old_residual_norm::Float64,
    autoscale::Bool,
)::Bool
    g0 = old_residual_norm
    # Quadratic model F(x+μΔx) = f₀ + μ·(J·Δx) + μ²·a along Δx_proposed. r_predict
    # (= f₀ + J·Δx) from the ρ test gives a = F(x+Δx) − r_predict for free (no extra matvec).
    c_fb, c_bb, c_fa, c_ba, c_aa =
        _iwamoto_quadratic_dots(old_residual, stateVector.r_predict, residual.Rv)
    μ = _iwamoto_multiplier(2.0 * c_fb, c_bb + 2.0 * c_fa, 2.0 * c_ba, c_aa)
    # Revert full step, apply damped step in a single fused pass.
    @. stateVector.x += (μ - 1.0) * stateVector.Δx_proposed
    residual(stateVector.x, time_step)
    g_damped = dot(residual.Rv, residual.Rv)
    if g_damped < g0
        @debug "Iwamoto fallback accepted: μ = $(siground(μ)), " *
               "g_damped/g₀ = $(siground(g_damped / g0))"
        J(time_step)
        _accept_trust_region_step!(stateVector, residual, J.Jv, autoscale)
        return true
    else
        # Damped step also failed — full revert.
        @. stateVector.x -= μ * stateVector.Δx_proposed
        copyto!(residual.Rv, old_residual)
        @debug "Iwamoto fallback rejected: μ = $(siground(μ)), " *
               "g_damped/g₀ = $(siground(g_damped / g0)); reverting"
        return false
    end
end

"""Does a single iteration of the `TrustRegionNRMethod`:
updates the `x` and `r` fields of the `stateVector` and computes
the value of the Jacobian at the new `x`, if needed. Unlike
`_simple_step`, this has a return value, the updated value of `delta``."""
function _trust_region_step(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::PFLinearSolverCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    delta::Float64,
    delta_max::Float64,
    eta::Float64,
    autoscale::Bool,
    iwamoto_fallback::Bool,
)
    old_delta = delta
    _set_Δx_nr!(
        stateVector,
        J,
        linSolveCache,
        TrustRegionACPowerFlow(),
        DEFAULT_REFINEMENT_THRESHOLD,
        DEFAULT_REFINEMENT_EPS,
    )
    _dogleg!(
        stateVector.Δx_proposed,
        stateVector.Δx_cauchy,
        stateVector.Δx_nr,
        stateVector.r,
        J.Jv,
        stateVector.d,
        delta,
        stateVector.r_predict,  # scratch for J·g; overwritten with the true r_predict below
    )
    # find proposed next point.
    stateVector.x .+= stateVector.Δx_proposed

    # use cache.Δx_nr as temporary buffer to store old residual
    # to avoid recomputing if we don't change x.
    oldResidual = stateVector.Δx_nr
    copyto!(oldResidual, residual.Rv)
    old_residual_norm = sum(abs2, stateVector.r)
    residual(stateVector.x, time_step)
    new_residual_norm = sum(abs2, residual.Rv)

    # Ratio of actual to predicted reduction
    LinearAlgebra.mul!(stateVector.r_predict, J.Jv, stateVector.Δx_proposed)
    stateVector.r_predict .+= stateVector.r
    predicted_reduction = old_residual_norm - sum(abs2, stateVector.r_predict)
    # The dogleg model reduction is non-negative by construction; a non-positive value
    # here is floating-point cancellation near convergence (‖r‖²≈0). Force a rejected-step
    # ρ to shrink the trust region — standard recovery, matching the LM solver's guard.
    rho = if predicted_reduction > 0.0
        (old_residual_norm - new_residual_norm) / predicted_reduction
    else
        @debug "Non-positive predicted reduction $(siground(predicted_reduction)); \
            rejecting step, shrinking trust region"
        -Inf
    end

    @debug "Trust region step: ρ = $(siground(rho)), η = $(siground(eta)), ||Δx|| = $(siground(norm(stateVector.Δx_proposed)))"

    step_accepted = false
    if rho > eta
        # Successful iteration
        @debug "Step accepted: sum of squares $(siground(dot(residual.Rv, residual.Rv))), L ∞ norm $(siground(norm(residual.Rv, Inf))), Δ = $(siground(delta)), ||Δx|| = $(siground(norm(stateVector.Δx_proposed)))"
        J(time_step)
        _accept_trust_region_step!(stateVector, residual, J.Jv, autoscale)
        step_accepted = true
    else
        # Unsuccessful step — try Iwamoto damping before reverting.
        if iwamoto_fallback
            iwamoto_accepted = _iwamoto_fallback!(
                time_step, stateVector, residual, J,
                oldResidual, old_residual_norm, autoscale)
            if iwamoto_accepted
                # Iwamoto accepted a damped step — shrink trust region since the
                # full proposed step was rejected by rho. Do not use rho-based
                # expansion logic because rho corresponds to the rejected full step.
                delta = min(delta / 2, delta_max)
                @debug "Trust region decreased (Iwamoto fallback accepted): δ $(siground(old_delta)) → $(siground(delta))"
                return delta
            end
        else
            stateVector.x .-= stateVector.Δx_proposed
            copyto!(residual.Rv, oldResidual)
            @debug "Step rejected: ρ = $(siground(rho)) ≤ η = $(siground(eta))"
        end
    end

    # Update size of trust region based on rho (only reached when the full step
    # was accepted via rho, or Iwamoto is disabled, or Iwamoto didn't help).
    if rho < HALVE_TRUST_REGION # rho < 0.1: insufficient improvement
        delta = delta / 2
        @debug "Trust region decreased: δ $(siground(old_delta)) → $(siground(delta)) (ρ < $(HALVE_TRUST_REGION))"
    elseif step_accepted && rho >= DOUBLE_TRUST_REGION # rho >= 0.9: good improvement
        delta = 2 * wnorm(stateVector.d, stateVector.Δx_proposed)
        @debug "Trust region increased (good): δ $(siground(old_delta)) → $(siground(delta)) (ρ ≥ $(DOUBLE_TRUST_REGION))"
    elseif step_accepted && rho >= MAX_DOUBLE_TRUST_REGION # rho >= 0.5: so-so improvement
        delta = max(delta, 2 * wnorm(stateVector.d, stateVector.Δx_proposed))
        @debug "Trust region increased (moderate): δ $(siground(old_delta)) → $(siground(delta)) (ρ ≥ $(MAX_DOUBLE_TRUST_REGION))"
    else
        @debug "Trust region unchanged: δ = $(siground(delta))"
    end
    delta = min(delta, delta_max)
    return delta
end

"""Inner products for the quadratic model `F(x+μΔx) = f₀ + μ·b + μ²·a` with
`b = J·Δx`, `a = F(x+Δx) − f₀ − b`. From `f₀`, `rpred = f₀ + b`, and `rv = F(x+Δx)`
returns `(f₀·b, b·b, f₀·a, b·a, a·a)`."""
@inline function _iwamoto_quadratic_dots(
    f0::Vector{Float64}, rpred::Vector{Float64}, rv::Vector{Float64},
)::NTuple{5, Float64}
    c_fb = 0.0
    c_bb = 0.0
    c_fa = 0.0
    c_ba = 0.0
    c_aa = 0.0
    @inbounds @simd for i in eachindex(f0, rpred, rv)
        b = rpred[i] - f0[i]
        a = rv[i] - rpred[i]
        c_fb += f0[i] * b
        c_bb += b * b
        c_fa += f0[i] * a
        c_ba += b * a
        c_aa += a * a
    end
    return c_fb, c_bb, c_fa, c_ba, c_aa
end

"""Iwamoto objective minus its μ-independent constant:
`g̃(μ) = q₁μ + q₂μ² + q₃μ³ + q₄μ⁴`. Dropping the constant preserves the minimizer."""
@inline function _iwamoto_objective(
    μ::Float64, q1::Float64, q2::Float64, q3::Float64, q4::Float64,
)::Float64
    return μ * (q1 + μ * (q2 + μ * (q3 + μ * q4)))
end

"""If μ ∈ [IWAMOTO_MU_MIN, IWAMOTO_MU_MAX] and g̃(μ) < best_g, return the
improved (μ, g̃(μ)); otherwise return (best_μ, best_g) unchanged."""
@inline function _try_iwamoto_candidate(
    μ::Float64,
    best_μ::Float64,
    best_g::Float64,
    q1::Float64,
    q2::Float64,
    q3::Float64,
    q4::Float64,
)::Tuple{Float64, Float64}
    if IWAMOTO_MU_MIN <= μ <= IWAMOTO_MU_MAX
        gval = _iwamoto_objective(μ, q1, q2, q3, q4)
        if gval < best_g
            return μ, gval
        end
    end
    return best_μ, best_g
end

"""Optimal Iwamoto multiplier μ ∈ [IWAMOTO_MU_MIN, IWAMOTO_MU_MAX] minimizing
`g̃(μ) = q₁μ + q₂μ² + q₃μ³ + q₄μ⁴` (coefficients from [`_iwamoto_quadratic_dots`](@ref)).
Stationary points solve the cubic `g̃'(μ) = 4q₄μ³ + 3q₃μ² + 2q₂μ + q₁ = 0`, found
analytically (depressed-cubic Cardano/trig form). Exact for the dogleg step;
reduces to classical Iwamoto & Tamura (1981) when `b = −f₀` (Newton step)."""
function _iwamoto_multiplier(q1::Float64, q2::Float64, q3::Float64, q4::Float64)::Float64
    # Initialize best candidate from domain boundaries.
    best_μ = IWAMOTO_MU_MIN
    best_g = _iwamoto_objective(IWAMOTO_MU_MIN, q1, q2, q3, q4)
    best_μ, best_g =
        _try_iwamoto_candidate(IWAMOTO_MU_MAX, best_μ, best_g, q1, q2, q3, q4)

    # Cubic coefficients: c₃μ³ + c₂μ² + c₁μ + c₀ = 0
    c3 = 4.0 * q4
    c2 = 3.0 * q3
    c1 = 2.0 * q2
    c0 = q1

    if abs(c3) < IWAMOTO_DEGENERACY_TOL
        # Degenerate: solve quadratic c₂μ² + c₁μ + c₀ = 0
        if abs(c2) > IWAMOTO_DEGENERACY_TOL
            disc = c1 * c1 - 4.0 * c2 * c0
            if disc >= 0.0
                sq = sqrt(disc)
                for μ in ((-c1 + sq) / (2.0 * c2), (-c1 - sq) / (2.0 * c2))
                    best_μ, best_g =
                        _try_iwamoto_candidate(μ, best_μ, best_g, q1, q2, q3, q4)
                end
            end
        elseif abs(c1) > IWAMOTO_DEGENERACY_TOL
            best_μ, best_g =
                _try_iwamoto_candidate(-c0 / c1, best_μ, best_g, q1, q2, q3, q4)
        end
        return best_μ
    end

    # Full cubic — depress to t³ + At + B = 0 via μ = t - p/3
    p = c2 / c3
    q = c1 / c3
    c0_n = c0 / c3
    p3 = p / 3.0
    A = q - p * p3
    B = c0_n - q * p3 + 2.0 * p3^3
    Δ = -4.0 * A^3 - 27.0 * B^2

    if Δ > 0.0
        # Three distinct real roots — trigonometric form (A < 0 guaranteed when Δ > 0).
        s = sqrt(-A / 3.0)
        m = 2.0 * s
        arg = clamp(-B / (2.0 * s * s * s), -1.0, 1.0)
        φ3 = acos(arg) / 3.0
        for k in 0:2
            best_μ, best_g = _try_iwamoto_candidate(
                m * cos(φ3 - 2.0 * π * k / 3.0) - p3,
                best_μ, best_g, q1, q2, q3, q4)
        end
    elseif Δ < 0.0
        # One real root — Cardano's formula.
        sqD = sqrt(max(-Δ / 108.0, 0.0))
        best_μ, best_g = _try_iwamoto_candidate(
            cbrt(-B / 2.0 + sqD) + cbrt(-B / 2.0 - sqD) - p3,
            best_μ, best_g, q1, q2, q3, q4)
    else
        # Δ ≈ 0 — repeated roots.
        if abs(A) < IWAMOTO_DEGENERACY_TOL
            # Triple root at t = 0.
            best_μ, best_g = _try_iwamoto_candidate(-p3, best_μ, best_g, q1, q2, q3, q4)
        else
            # Simple root t₁ = 3B/A and double root t₂ = -3B/(2A).
            for t in (3.0 * B / A, -3.0 * B / (2.0 * A))
                best_μ, best_g = _try_iwamoto_candidate(
                    t - p3, best_μ, best_g, q1, q2, q3, q4)
            end
        end
    end

    return best_μ
end

"""Classical Iwamoto & Tamura (1981) multiplier for the Newton step (`b = −f₀`),
with `g₀ = ‖f₀‖²`, `g₁ = f₀ᵀf₁`, `g₂ = ‖f₁‖²`, `f₁ = F(x+Δx)`."""
@inline function _iwamoto_multiplier(g0::Float64, g1::Float64, g2::Float64)::Float64
    return _iwamoto_multiplier(-2.0 * g0, g0 + 2.0 * g1, -2.0 * g1, g2)
end

"""Does a single iteration of `NewtonRaphsonACPowerFlow`. Updates the `r` and `x`
 fields of the `stateVector`, and computes the Jacobian at the new `x`."""
function _simple_step(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::PFLinearSolverCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
)
    copyto!(stateVector.r, residual.Rv)
    _set_Δx_nr!(
        stateVector,
        J,
        linSolveCache,
        NewtonRaphsonACPowerFlow(),
        refinement_threshold,
        refinement_eps,
    )
    # update x
    stateVector.x .+= stateVector.Δx_nr
    # update data's fields (the bus angles/voltages) to match x, and update the residual.
    # do this BEFORE updating the Jacobian. The Jacobian computation uses data's fields, not x.
    residual(stateVector.x, time_step)
    # update jacobian.
    J(time_step)
    return
end

"""Does a single iteration of Newton-Raphson with Iwamoto step control.
Computes the Newton step, takes a full trial step, and checks whether the
residual norm decreased. If not, computes an optimal damping multiplier `μ`
and applies a damped step instead. When the damped step also fails to reduce
the residual, the step is reverted to avoid divergence.

Returns `true` if the step made progress (residual decreased), `false` if
the step was reverted. Consecutive reverts signal stagnation and the caller
should terminate early."""
function _iwamoto_step(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::PFLinearSolverCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
)::Bool
    # Save pre-step residual f into stateVector.r
    copyto!(stateVector.r, residual.Rv)
    # Compute Newton step Δx_nr = -J⁻¹f
    _set_Δx_nr!(
        stateVector,
        J,
        linSolveCache,
        NewtonRaphsonACPowerFlow(),
        refinement_threshold,
        refinement_eps,
    )
    # Take full trial step: x += Δx_nr
    stateVector.x .+= stateVector.Δx_nr
    # Evaluate trial residual b = F(x + Δx)
    residual(stateVector.x, time_step)

    # Compute gram scalars for Iwamoto criterion
    g0 = dot(stateVector.r, stateVector.r)
    g1 = dot(stateVector.r, residual.Rv)
    g2 = dot(residual.Rv, residual.Rv)

    if g2 < g0
        # Full step reduced residual — accept it (μ = 1).
        @debug "Iwamoto: full step accepted (g₂/g₀ = $(g2/g0))"
        J(time_step)
        return true
    end

    # Full step did not reduce residual — compute optimal μ.
    μ = _iwamoto_multiplier(g0, g1, g2)
    @debug "Iwamoto: damped step μ = $μ (g₂/g₀ = $(g2/g0))"
    # Undo full step and apply damped step.
    stateVector.x .-= stateVector.Δx_nr
    stateVector.x .+= μ .* stateVector.Δx_nr
    # Re-evaluate residual at damped point.
    residual(stateVector.x, time_step)
    # Check whether the damped step actually improved the residual.
    g_damped = dot(residual.Rv, residual.Rv)
    if g_damped >= g0
        # Damped step did not improve — revert to pre-step state.
        @debug "Iwamoto: damped step did not reduce residual " *
               "(g_damped/g₀ = $(g_damped/g0), μ = $μ); reverting"
        stateVector.x .-= μ .* stateVector.Δx_nr
        residual(stateVector.x, time_step)
        return false
    end
    # Damped step improved — accept it.
    J(time_step)
    return true
end

"""Settings of the backtracking line search on the Newton step
(`solver_settings = Dict(:line_search => :backtracking)`, see [`NewtonRaphsonACPowerFlow`](@ref)):
the Armijo constant `c`, the factor `shrink` applied to α after a rejected trial, the step
`min_step` at or below which the step is taken without the test, and which residual rows the
2-norm counts (`norm`, see [`_line_search_rows`](@ref))."""
struct BacktrackingLineSearch
    c::Float64
    shrink::Float64
    min_step::Float64
    norm::Symbol
end

"""Per-solve tally of the backtracking line search, for the solver log: Newton steps taken,
steps shortened (α < 1), rejected trials (step cuts) in all, steps taken at the minimum step
without the Armijo test, and the smallest α taken."""
mutable struct LineSearchTally
    steps::Int
    shortened::Int
    cuts::Int
    at_min_step::Int
    min_α::Float64
end
LineSearchTally() = LineSearchTally(0, 0, 0, 0, 1.0)

"""Check the line search settings and return a [`BacktrackingLineSearch`](@ref), or `nothing`
for `line_search = :none`. Throws an `ArgumentError` for an unknown method, a parameter out of
range, or `iwamoto = true` beside a line search (two step controls)."""
function _line_search_settings(
    line_search,
    c::Real,
    shrink::Real,
    min_step::Real,
    iwamoto::Bool,
    norm::Symbol = DEFAULT_LINE_SEARCH_NORM,
)::Union{Nothing, BacktrackingLineSearch}
    line_search in LINE_SEARCH_METHODS || throw(
        ArgumentError(
            "line_search = $(repr(line_search)) is not one of $(LINE_SEARCH_METHODS)",
        ),
    )
    line_search === :none && return nothing
    iwamoto && throw(
        ArgumentError(
            "line_search = :backtracking and iwamoto = true are two step controls; " *
            "set one of them",
        ),
    )
    0.0 < c < 1.0 ||
        throw(ArgumentError("line_search_c = $c must lie in (0, 1)"))
    0.0 < shrink < 1.0 ||
        throw(ArgumentError("line_search_shrink = $shrink must lie in (0, 1)"))
    0.0 < min_step <= 1.0 ||
        throw(ArgumentError("line_search_min_step = $min_step must lie in (0, 1]"))
    norm in LINE_SEARCH_NORMS || throw(
        ArgumentError("line_search_norm = $(repr(norm)) is not one of $(LINE_SEARCH_NORMS)"),
    )
    return BacktrackingLineSearch(Float64(c), Float64(shrink), Float64(min_step), norm)
end

"""The residual rows the line search's 2-norm counts, or `nothing` for every row.

`norm = :balance` on the polar formulation counts the network balance equations a reduced
Newton solves: the P row of every PV and PQ bus and the Q row of every PQ bus, plus every row
past the buses' (LCC, VSC, area interchange). It leaves out the rows whose mismatch the state's
own free injection absorbs, the REF buses' P and Q rows and the PV buses' Q rows, so the
measure does not depend on how far a step moved the slack and generator Q states (it equals the
‖F‖₂ of a solver whose state is |V| and θ only). A PV bus regulated by remote generators
(GAP-015) keeps its Q row: its Q slot is the group's total, shared over member buses whose Q
rows are network equations. Bus types are read at `time_step` when the solve starts (they do not
change inside one Newton solve). `norm = :full`, and the rectangular and mixed formulations,
count every row."""
_line_search_rows(::Union{ACRectangularCIResidual, ACMixedCPBResidual}, ::Symbol, ::Int) =
    nothing
function _line_search_rows(residual::ACPowerFlowResidual, norm::Symbol, time_step::Int)
    norm === :full && return nothing
    data = residual.data
    n = size(data.bus_type, 1)
    regulated = data.remote_voltage_control.regulated_group
    rows = Int[]
    sizehint!(rows, length(residual.Rv))
    @inbounds for ix in 1:n
        bt = data.bus_type[ix, time_step]
        if bt == PSY.ACBusTypes.PQ
            push!(rows, 2 * ix - 1, 2 * ix)
        elseif bt == PSY.ACBusTypes.PV
            push!(rows, 2 * ix - 1)
            (!isempty(regulated) && regulated[ix] != 0) && push!(rows, 2 * ix)
        end
    end
    append!(rows, (2 * n + 1):length(residual.Rv))
    return rows
end

"""The line search's measure: `‖F‖₂` over `rows` (`nothing`: every row)."""
_line_search_norm(F::Vector{Float64}, ::Nothing) = norm(F, 2)
function _line_search_norm(F::Vector{Float64}, rows::Vector{Int})
    s = 0.0
    @inbounds for r in rows
        s += F[r]^2
    end
    return sqrt(s)
end

"""Does a single iteration of Newton-Raphson with a backtracking line search on the residual's
2-norm (over `rows`, see [`_line_search_rows`](@ref)). The full Newton step is tried first (α = 1); while the trial residual is not finite or
fails the Armijo test `‖F(x + αΔx)‖₂ < (1 − c·α)‖F(x)‖₂`, α is multiplied by `shrink`. Once
α ≤ `min_step` the step `x + αΔx` is taken without the test, so the iteration always moves
(the step is never reverted). Updates `x`, the residual and the Jacobian; returns α."""
function _backtracking_step(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::PFLinearSolverCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    ls::BacktrackingLineSearch,
    rows::Union{Nothing, Vector{Int}},
    tally::LineSearchTally,
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
)::Float64
    copyto!(stateVector.r, residual.Rv)
    f0 = _line_search_norm(stateVector.r, rows)
    _set_Δx_nr!(
        stateVector,
        J,
        linSolveCache,
        NewtonRaphsonACPowerFlow(),
        refinement_threshold,
        refinement_eps,
    )
    # Δx_proposed is the trust region's buffer, unused by Newton: it keeps the pre-step x so
    # each trial is x₀ + αΔx exactly, not a running sum.
    x_old = stateVector.Δx_proposed
    copyto!(x_old, stateVector.x)
    α = 1.0
    cuts = 0
    forced = false
    while true
        @. stateVector.x = x_old + α * stateVector.Δx_nr
        residual(stateVector.x, time_step)
        if α <= ls.min_step
            forced = true
            break
        end
        f_trial = _line_search_norm(residual.Rv, rows)
        if isfinite(f_trial) && f_trial < (1.0 - ls.c * α) * f0
            break
        end
        α *= ls.shrink
        cuts += 1
    end
    tally.steps += 1
    tally.cuts += cuts
    α < 1.0 && (tally.shortened += 1)
    forced && (tally.at_min_step += 1)
    tally.min_α = min(tally.min_α, α)
    @debug "Line search: α = $(siground(α)) after $cuts cut(s)" *
           (forced ? " (minimum step, taken untested)" : "") *
           ", ‖F‖₂ $(siground(f0)) → $(siground(_line_search_norm(residual.Rv, rows)))"
    J(time_step)
    return α
end

"""Log the line search's tally for one Newton solve."""
function _report_line_search(tally::LineSearchTally, ls::BacktrackingLineSearch)
    @info "Backtracking line search: $(tally.shortened) of $(tally.steps) Newton steps " *
          "shortened ($(tally.cuts) step cuts), $(tally.at_min_step) at the minimum step " *
          "$(ls.min_step); smallest α $(siground(tally.min_α))."
    return
end

# Formulation-dispatched voltage-magnitude validation, driven entirely by the
# per-formulation index list precomputed once on the residual. Polar indexes
# the state as `[|V|, θ, …]` (`x[2i-1]` = |V|, PQ only); rectangular CI and
# mixed CPB states are `(e, f, …)` per-bus blocks validating `e²+f² ∈
# [min², max²]` over PQ/PV.
function _validate_state_magnitudes(
    r::ACPowerFlowResidual,
    x::Vector{Float64},
    range::MinMax,
    i::Int64,
)
    validate_voltage_magnitudes(x, r.validate_indices, range, i)
    return
end

function _validate_state_magnitudes(
    r::ACRectangularCIResidual,
    x::Vector{Float64},
    range::MinMax,
    i::Int64,
)
    _validate_squared_voltage_magnitudes(x, r.validate_offsets, range, i)
    return
end

function _validate_state_magnitudes(
    r::ACMixedCPBResidual,
    x::Vector{Float64},
    range::MinMax,
    i::Int64,
)
    _validate_squared_voltage_magnitudes(x, r.validate_offsets, range, i)
    return
end

"""Runs the full `NewtonRaphsonACPowerFlow`.
# Keyword arguments:
- `maxIterations::Int`: maximum iterations. Default: $DEFAULT_NR_MAX_ITER.
- `tol::Float64`: tolerance. The iterative search ends when `norm(abs.(residual)) < tol`.
    Default: $DEFAULT_NR_TOL.
- `refinement_threshold::Float64`: If the solution to `J_x Δx = r` satisfies
    `norm(J_x Δx - r, 1)/norm(r, 1) > refinement_threshold`, do iterative refinement to
    improve the accuracy. Default: $DEFAULT_REFINEMENT_THRESHOLD.
- `refinement_eps::Float64`: run iterative refinement on `J_x Δx = r` until
    `norm(Δx_{i}-Δx_{i+1}, 1)/norm(r,1) < refinement_eps`. Default:
    $DEFAULT_REFINEMENT_EPS
- `iwamoto::Bool`: Iwamoto step control (see [`NewtonRaphsonACPowerFlow`](@ref)). Default: `false`.
- `line_search::Symbol`: `:none` (the full Newton step) or `:backtracking` (see
    `_backtracking_step`). Default: `$(repr(DEFAULT_LINE_SEARCH))`.
- `line_search_c`, `line_search_shrink`, `line_search_min_step`: the line search's Armijo
    constant, step factor and minimum step. Defaults: $DEFAULT_LINE_SEARCH_C,
    $DEFAULT_LINE_SEARCH_SHRINK, $DEFAULT_LINE_SEARCH_MIN_STEP.
- `line_search_norm::Symbol`: the rows the line search's 2-norm counts, `:balance` or `:full`
    (see `_line_search_rows`). Default: `$(repr(DEFAULT_LINE_SEARCH_NORM))`. """
function _run_power_flow_method(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::PFLinearSolverCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    ::Type{NewtonRaphsonACPowerFlow};
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    tol::Float64 = DEFAULT_NR_TOL,
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    iwamoto::Bool = false,
    line_search::Symbol = DEFAULT_LINE_SEARCH,
    line_search_c::Real = DEFAULT_LINE_SEARCH_C,
    line_search_shrink::Real = DEFAULT_LINE_SEARCH_SHRINK,
    line_search_min_step::Real = DEFAULT_LINE_SEARCH_MIN_STEP,
    line_search_norm::Symbol = DEFAULT_LINE_SEARCH_NORM,
    stop_at_fold::Bool = false,
    _ignored...,  # absorb unknown keys from caller without error
)
    validate_vms = validate_voltage_magnitudes
    ls = _line_search_settings(line_search, line_search_c, line_search_shrink,
        line_search_min_step, iwamoto, line_search_norm)
    rows = isnothing(ls) ? nothing : _line_search_rows(residual, ls.norm, time_step)
    tally = LineSearchTally()
    α = 1.0
    i, converged = 1, false
    consecutive_reverts = 0
    monitor, diag_state = setup_solver_diagnostics(J, stop_at_fold)
    while i < maxIterations && !converged
        if !isnothing(ls)
            α = _backtracking_step(
                time_step,
                stateVector,
                linSolveCache,
                residual,
                J,
                ls,
                rows,
                tally,
                refinement_threshold,
                refinement_eps,
            )
        elseif iwamoto
            made_progress = _iwamoto_step(
                time_step,
                stateVector,
                linSolveCache,
                residual,
                J,
                refinement_threshold,
                refinement_eps,
            )
            if made_progress
                consecutive_reverts = 0
            else
                consecutive_reverts += 1
                if consecutive_reverts >= IWAMOTO_MAX_REVERTS
                    @debug "Iwamoto: $consecutive_reverts consecutive reverted steps; terminating early"
                    break
                end
            end
        else
            _simple_step(
                time_step,
                stateVector,
                linSolveCache,
                residual,
                J,
                refinement_threshold,
                refinement_eps,
            )
        end
        validate_vms && _validate_state_magnitudes(
            residual,
            stateVector.x,
            vm_validation_range,
            i,
        )
        if !isnothing(diag_state)
            # After `_simple_step`, J.Jv and residual.Rv are at the same iterate, so
            # one refactor feeds both the log line and the bail-out.
            label = isnothing(ls) ? "NR iter $i" : "NR iter $i (α = $(siground(α)))"
            if run_solver_diagnostics!(
                diag_state, label, residual, J, time_step,
                linSolveCache, monitor, stop_at_fold)
                isnothing(ls) || _report_line_search(tally, ls)
                return false, i
            end
        end
        converged = norm(residual.Rv, Inf) < tol
        # A non-finite residual cannot recover (the line search's untested minimum step can
        # land on one): stop, as the line search's reference does.
        if !converged && !isnothing(ls) && !all(isfinite, residual.Rv)
            @debug "Line search: the residual is not finite after step $i; stopping"
            break
        end
        if !converged
            i += 1
        end
    end
    isnothing(ls) || _report_line_search(tally, ls)
    return converged, i
end

"""Runs the full `TrustRegionNRMethod`.
# Keyword arguments:
- `maxIterations::Int`: maximum iterations. Default: $DEFAULT_NR_MAX_ITER.
- `tol::Float64`: tolerance. The iterative search ends when `maximum(abs.(residual)) < tol`.
    Default: $DEFAULT_NR_TOL.
- `factor::Float64`: the trust region starts out with radius `factor*norm(x_0, 1)`,
    where `x_0` is our initial guess, taken from `data`. Default: $DEFAULT_TRUST_REGION_FACTOR.
- `eta::Float64`: improvement threshold. If the observed improvement in our residual
    exceeds `eta` times the predicted improvement, we accept the new `x_i`.
    Default: $DEFAULT_TRUST_REGION_ETA.
- `iwamoto_fallback::Bool`: when a trust region step is rejected, attempt Iwamoto
    damping to salvage the step before reverting. Default: $DEFAULT_IWAMOTO_FALLBACK."""
function _run_power_flow_method(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::PFLinearSolverCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    ::Type{TrustRegionACPowerFlow};
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    tol::Float64 = DEFAULT_NR_TOL,
    factor::Float64 = DEFAULT_TRUST_REGION_FACTOR,
    eta::Float64 = DEFAULT_TRUST_REGION_ETA,
    autoscale::Bool = DEFAULT_AUTOSCALE,
    iwamoto_fallback::Bool = DEFAULT_IWAMOTO_FALLBACK,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    stop_at_fold::Bool = false,
    _ignored...,  # absorb unknown keys from caller without error
)
    validate_vms = validate_voltage_magnitudes

    if eta > 1.0 || eta < 0.0
        @warn("η = $eta is outside [0, 1]") # eta is set to 2.0 in one test.
    end

    if autoscale
        for i in 1:length(stateVector.x)
            stateVector.d[i] = norm(view(J.Jv, :, i))
            if iszero(stateVector.d[i])
                stateVector.d[i] = 1.0
            end
        end
    end

    delta = norm(stateVector.x) > 0 ? factor * norm(stateVector.x) : factor
    delta_max = DEFAULT_TRUST_REGION_DELTA_MAX_FACTOR * delta
    i, converged = 0, false
    residualSize = dot(residual.Rv, residual.Rv)
    linf = norm(residual.Rv, Inf)
    @debug "initially: sum of squares $(siground(residualSize)), L ∞ norm $(siground(linf)), Δ $(siground(delta))"

    monitor, diag_state = setup_solver_diagnostics(J, stop_at_fold)
    while i < maxIterations && !converged
        delta = _trust_region_step(
            time_step,
            stateVector,
            linSolveCache,
            residual,
            J,
            delta,
            delta_max,
            eta,
            autoscale,
            iwamoto_fallback,
        )
        validate_vms && _validate_state_magnitudes(
            residual,
            stateVector.x,
            vm_validation_range,
            i,
        )
        if !isnothing(diag_state)
            # After `_trust_region_step` (incl. reject and iwamoto-fallback), J.Jv and
            # residual.Rv are at the same iterate, so one refactor feeds both.
            run_solver_diagnostics!(
                diag_state, "TR iter $i", residual, J, time_step,
                linSolveCache, monitor, stop_at_fold) &&
                return false, i
        end
        converged = norm(residual.Rv, Inf) < tol
        if !converged
            i += 1
        end
    end
    return converged, i
end

"""Log final residual, report convergence, compute optional post-processing factors,
and return `true`/`false`. Shared by all AC power flow drivers."""
# `Jv === nothing`: the fast-decoupled :decoupled driver skipped the formulation Jacobian (neither
# loss nor voltage-stability factors were requested), so there is nothing to compute — just report
# convergence. Dispatch keeps this path free of the factor machinery entirely.
function _finalize_power_flow(
    converged::Bool,
    i::Int,
    solver_name::String,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    data::ACPowerFlowData,
    ::Nothing,
    time_step::Int64,
)
    return _report_power_flow_convergence(converged, i, solver_name, residual)
end

function _finalize_power_flow(
    converged::Bool,
    i::Int,
    solver_name::String,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    data::ACPowerFlowData,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    time_step::Int64,
)
    if converged
        _warn_small_lcc_angles(data, time_step)
        if get_calculate_loss_factors(data)
            _calculate_loss_factors(data, Jv, time_step)
        end
        if get_calculate_voltage_stability_factors(data)
            _calculate_voltage_stability_factors(data, Jv, time_step)
        end
    end
    return _report_power_flow_convergence(converged, i, solver_name, residual)
end

"""Log the final residual size and convergence/non-convergence, returning `converged`. Shared by
both `_finalize_power_flow` methods (Jacobian and Jacobian-free)."""
function _report_power_flow_convergence(
    converged::Bool,
    i::Int,
    solver_name::String,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
)
    @info("Final residual size: $(norm(residual.Rv, 2)) L2, $(norm(residual.Rv, Inf)) L∞.")
    if converged
        @info("The $solver_name solver converged after $i iterations.")
        return true
    end
    @error("The $solver_name solver failed to converge after $i iterations.")
    return false
end

"""Warn if any LCC's converged thyristor angle lies outside the physical
operating window `(LCC_SMALL_ANGLE_THRESHOLD, π/2 − LCC_SMALL_ANGLE_THRESHOLD)`
(≈ 5° to 85° by default). Real PSS/E LCCs operate well inside this range
(rectifier α_r ≈ 10-20°, inverter γ_i ≈ 14-18°). Either extreme sits near
an `arccos` clamp boundary where `Q_s`'s second derivatives are singular
(`1/sin³ϕ`): low α puts the rectifier-side `u_r → +1` or the inverter-side
`u_i → −1`; high α (approaching π/2) puts them on the other boundary, and
beyond π/2 the converter's rectifying/inverting role would reverse.
Hessian-based solvers (LM, RobustHomotopy) degrade in either regime, and
even direct Newton hitting one of these bounds is a sign the input data
is non-physical."""
function _warn_small_lcc_angles(data::ACPowerFlowData, time_step::Int)
    n_lcc = size(data.lcc.p_set, 1)
    iszero(n_lcc) && return
    lo = LCC_SMALL_ANGLE_THRESHOLD
    hi = π / 2 - LCC_SMALL_ANGLE_THRESHOLD
    for i in 1:n_lcc
        α_r = data.lcc.rectifier.thyristor_angle[i, time_step]
        α_i = data.lcc.inverter.thyristor_angle[i, time_step]
        out_of_range = α_r < lo || α_r > hi || α_i < lo || α_i > hi
        if out_of_range
            (fb, tb) = data.lcc.arcs[i]
            @warn(
                "LCC $i (arc $(fb) → $(tb)): converged thyristor angles " *
                "α_r = $(rad2deg(α_r))°, α_i = $(rad2deg(α_i))° — one or " *
                "both outside the physical-realism window " *
                "($(rad2deg(lo))°, $(rad2deg(hi))°). Typical PSS/E LCCs " *
                "operate at α_r ≈ 10-20° and γ_i ≈ 14-18°. Values near " *
                "0° or π/2 sit at the LCC arccos clamp boundary " *
                "(singular Q_s Hessian). Check the configured " *
                "rectifier_delay_angle_limits and " *
                "inverter_extinction_angle_limits.", maxlog = PF_MAX_LOG,
            )
        end
    end
    return
end

"""Formulation-specific post-Newton step. Polar needs nothing; the rectangular
CI formulation distributes the converged subnetwork slack into the bus
injection arrays."""
_finalize_formulation!(::ACPolarPowerFlow, data, x, residual, time_step) = nothing

function _finalize_formulation!(
    ::ACRectangularPowerFlow,
    data::ACPowerFlowData,
    x::Vector{Float64},
    residual::ACRectangularCIResidual,
    time_step::Int64,
)
    rect_finalize_bus_injections!(
        data, x, residual.bus_state_offset, residual.P_net_set,
        residual.bus_slack_participation_factors, residual.subnetworks,
        residual.independent_ref, time_step,
    )
    return
end

function _finalize_formulation!(
    ::ACMixedPowerFlow,
    data::ACPowerFlowData,
    x::Vector{Float64},
    residual::ACMixedCPBResidual,
    time_step::Int64,
)
    mixed_finalize_bus_injections!(
        data, x, residual.bus_state_offset,
        residual.bus_slack_participation_factors, residual.subnetworks,
        residual.independent_ref,
        residual.e_state, residual.f_state,
        time_step,
    )
    return
end

# Persistent NR/TR cache reusing the symbolic factorization across solves on one `data` (Q-limit
# loop, continuation, PCM steps). The sparsity pattern is invariant across NR iterations and
# PV↔PQ flips (as `ACJacobianStructureCache` relies on), so symbolic analysis runs once and
# `numeric_refactor!` refreshes values. Keyed on network-matrix identity + slack nonzero pattern
# + backend type; an in-place tap/PAR Y-bus edit keeps identity, so the cache correctly survives it.
#
# The numeric refactor (`klu_refactor`) reuses the pivot sequence of the last pivoting
# factorization. A bus-type change (a dropped island's rows turned REF, a new swing, a PV bus
# turned PQ by an outage) changes which Jacobian entries carry the large values, so a pivot
# sequence chosen under the old types can be numerically useless and NR stalls (it also did
# across the time steps of one multiperiod solve when they differ in types). `bus_types` is the
# bus-type column of the time step that sequence was chosen for: when the column being solved
# differs, the numeric factor is dropped (symbolic kept), so the next `numeric_refactor!`
# repivots (`_refresh_pivots!`). A PV→PQ flip in the Q-limit loop therefore costs one pivoting
# factorization, not a symbolic analysis.
struct PolarNRCache{M, B, C} <: SolverCache
    matrix::M
    nzind::Vector{Int}
    backend::B
    linSolveCache::C
    bus_types::Vector{PSY.ACBusTypes}
end

# Backend compared by TYPE (mirrors `_reuse_dc_cache`): a change of backend rebuilds.
_reuse_polar_nr_cache(::Nothing, matrix, nzind, backend) = nothing
_reuse_polar_nr_cache(e::PolarNRCache, matrix, nzind, backend) =
    if e.matrix === matrix && e.nzind == nzind && typeof(e.backend) === typeof(backend)
        e.linSolveCache
    else
        nothing
    end

# Make the reused cache repivot when the bus types differ from those its pivots were chosen
# under. Returns `false` when the backend cannot drop its numeric factor alone (rebuild).
function _refresh_pivots!(e::PolarNRCache, bus_type::AbstractVector{PSY.ACBusTypes})
    e.bus_types == bus_type && return true
    drop_numeric_factor!(e.linSolveCache) || return false
    copyto!(e.bus_types, bus_type)
    return true
end

# Return the persisted symbolic-factored cache when the structure+backend match; otherwise build
# it (make + symbolic_factor!), store it, and count the symbolic factorization. `nzind` is the
# slack-participation nonzero pattern — the same key component `_get_or_build_jacobian_structure`
# uses — so this cache and the structure cache always agree on a rebuild.
function _get_or_build_polar_nr_cache!(
    data::ACPowerFlowData,
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    backend,
    slack_factors::SparseVector{Float64, Int},
    time_step::Int,
)
    nzind = SparseArrays.nonzeroinds(slack_factors)
    bus_type = view(data.bus_type, :, time_step)
    entry = data.polar_nr_cache[]
    reused = _reuse_polar_nr_cache(entry, data.power_network_matrix, nzind, backend)
    if !isnothing(reused) && length(entry.bus_types) == length(bus_type) &&
       _refresh_pivots!(entry, bus_type)
        return reused
    end
    linSolveCache = make_linear_solver_cache(backend, Jv)
    symbolic_factor!(linSolveCache, Jv)
    _count_symbolic_factor!(data)
    data.polar_nr_cache[] = PolarNRCache(
        data.power_network_matrix, copy(nzind), backend, linSolveCache, collect(bus_type))
    return linSolveCache
end

# Polar: the Jacobian sparsity is invariant across NR iterations and PV↔PQ flips, so reuse the
# persisted symbolic factorization (see `PolarNRCache`).
_nr_linear_solver_cache!(
    data::ACPowerFlowData, J::ACPowerFlowJacobian, backend,
    slack_factors::SparseVector{Float64, Int}, time_step::Int) =
    _get_or_build_polar_nr_cache!(data, J.Jv, backend, slack_factors, time_step)

# Rectangular / mixed CB: a PV↔PQ flip changes the per-bus variable count / block pattern, so the
# sparsity structure is NOT flip-invariant across the Q-limit loop — build + symbolically factor
# fresh each call (the pre-P1 behavior for these formulations).
function _nr_linear_solver_cache!(
    data::ACPowerFlowData,
    J::Union{ACRectangularCIJacobian, ACMixedCPBJacobian},
    backend,
    ::SparseVector{Float64, Int},
    ::Int,
)
    linSolveCache = make_linear_solver_cache(backend, J.Jv)
    symbolic_factor!(linSolveCache, J.Jv)
    _count_symbolic_factor!(data)
    return linSolveCache
end

# Polar: defer the Jacobian entirely — a 0-iteration warm start must not pay for a
# Jacobian evaluation + sparse-structure copy. The caller builds J only when the
# convergence check fails.
function _nr_initialize_with_jacobian_deferred(
    pf::ACPolarPowerFlow, data::ACPowerFlowData, time_step::Int64; kwargs...,
)
    residual, x0 = _initialize_residual_x0(pf, data, time_step; kwargs...)
    return residual, nothing, x0
end

# Rectangular/mixed: J is structure-only (no value evaluation), cheap enough to build eagerly.
# These formulations do not call J(time_step) in their setup, so the cost is just the
# sparse-structure allocation (~1-2 MB), not the full evaluation.
function _nr_initialize_with_jacobian_deferred(
    pf::ACRectangularPowerFlow{T},
    data::ACPowerFlowData,
    time_step::Int64;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    _ignored...,
) where {T <: ACPowerFlowSolverType}
    residual = ACRectangularCIResidual(data, time_step)
    if isnothing(x0)
        x0_computed = improve_x0(pf, data, residual, time_step)
    else
        x0_computed = copy(x0)
        @warn "Using caller-provided x0; skipping improve_x0."
        residual(x0_computed, time_step)
    end
    _log_initial_residual(residual)
    J = ACRectangularCIJacobian(residual, time_step)
    return residual, J, x0_computed
end

function _nr_initialize_with_jacobian_deferred(
    pf::ACMixedPowerFlow{T},
    data::ACPowerFlowData,
    time_step::Int64;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    _ignored...,
) where {T <: ACPowerFlowSolverType}
    residual = ACMixedCPBResidual(data, time_step)
    if isnothing(x0)
        x0_computed = improve_x0(pf, data, residual, time_step)
    else
        x0_computed = copy(x0)
        @warn "Using caller-provided x0; skipping improve_x0."
        residual(x0_computed, time_step)
    end
    _log_initial_residual(residual)
    J = ACMixedCPBJacobian(residual, time_step)
    return residual, J, x0_computed
end

# Build the Jacobian when the deferred path (polar) needs it after a failed convergence check.
# Rectangular/mixed already have J from setup.
function _nr_build_jacobian(
    ::ACPolarPowerFlow, residual::ACPowerFlowResidual, ::Nothing, time_step::Int64,
)
    J = ACPowerFlowJacobian(residual, time_step)
    J(time_step)
    return J
end
_nr_build_jacobian(::AbstractACPowerFlow, residual, J, time_step::Int64) = J

function _newton_power_flow(
    pf::AbstractACPowerFlow{T},
    data::ACPowerFlowData,
    time_step::Int64;
    # shared kwargs
    tol::Float64 = DEFAULT_NR_TOL,
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    # NR-specific
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
    iwamoto::Bool = false,
    line_search::Symbol = DEFAULT_LINE_SEARCH,
    line_search_c::Real = DEFAULT_LINE_SEARCH_C,
    line_search_shrink::Real = DEFAULT_LINE_SEARCH_SHRINK,
    line_search_min_step::Real = DEFAULT_LINE_SEARCH_MIN_STEP,
    line_search_norm::Symbol = DEFAULT_LINE_SEARCH_NORM,
    # TR-specific
    factor::Float64 = DEFAULT_TRUST_REGION_FACTOR,
    eta::Float64 = DEFAULT_TRUST_REGION_ETA,
    autoscale::Bool = DEFAULT_AUTOSCALE,
    iwamoto_fallback::Bool = DEFAULT_IWAMOTO_FALLBACK,
    # NR and TR: fold / voltage-collapse bail-out (any backend; κ̂ is KLU-only)
    stop_at_fold::Bool = false,
    # initialize_power_flow_variables
    x0::Union{Vector{Float64}, Nothing} = nothing,
    # linear solver backend, resolved by `PNM.resolve_linear_solver`. Canonical names:
    # "KLU" | "AppleAccelerateLU" | "MKLPardiso" (PNM is the source of truth for any
    # aliases); `nothing` uses PNM's platform default.
    linear_solver::Union{Nothing, AbstractString} = nothing,
    _ignored...,
) where {T <: Union{TrustRegionACPowerFlow, NewtonRaphsonACPowerFlow}}
    if T === TrustRegionACPowerFlow && line_search !== :none
        throw(ArgumentError("line_search applies to NewtonRaphsonACPowerFlow only"))
    end

    # setup: common code
    init_kwargs = if isnothing(x0)
        (; validate_voltage_magnitudes, vm_validation_range)
    else
        (; validate_voltage_magnitudes, vm_validation_range, x0)
    end
    # Build residual + x0 WITHOUT the Jacobian: a 0-iteration warm start (already converged)
    # must not pay for a Jacobian evaluation + sparse-structure copy. The Jacobian is built
    # lazily only when the convergence check fails.
    residual, J_or_nothing, x0_init = _nr_initialize_with_jacobian_deferred(
        pf, data, time_step; init_kwargs...)
    converged = norm(residual.Rv, Inf) < tol

    i = 0
    x_final = x0_init
    if !converged
        J = _nr_build_jacobian(pf, residual, J_or_nothing, time_step)
        backend = resolve_linear_solver_backend(linear_solver)
        # Polar reuses the persisted symbolic factor; rect/mixed build fresh (structure not
        # flip-invariant). See `_nr_linear_solver_cache!` / `PolarNRCache`.
        linSolveCache = _nr_linear_solver_cache!(
            data, J, backend, residual.bus_slack_participation_factors, time_step)
        stateVector = StateVectorCache(x0_init, residual.Rv)
        converged, i = _run_power_flow_method(
            time_step,
            stateVector,
            linSolveCache,
            residual,
            J,
            T;
            tol,
            maxIterations,
            validate_voltage_magnitudes,
            vm_validation_range,
            refinement_threshold,
            refinement_eps,
            iwamoto,
            line_search,
            line_search_c,
            line_search_shrink,
            line_search_min_step,
            line_search_norm,
            factor,
            eta,
            autoscale,
            iwamoto_fallback,
            stop_at_fold,
        )
        x_final = stateVector.x
        _finalize_formulation!(pf, data, x_final, residual, time_step)
        return _finalize_power_flow(
            converged, i, string(T), residual, data, J.Jv, time_step)
    end
    _finalize_formulation!(pf, data, x_final, residual, time_step)
    # 0-iteration warm start: skip the Jacobian-dependent post-processing UNLESS the caller
    # opted into loss / voltage-stability factors — those need J even at 0 iterations, or a
    # first solve that lands within tol would leave them at their zero-initialized values.
    if get_calculate_loss_factors(data) || get_calculate_voltage_stability_factors(data)
        J = _nr_build_jacobian(pf, residual, J_or_nothing, time_step)
        return _finalize_power_flow(
            converged, i, string(T), residual, data, J.Jv, time_step)
    end
    return _finalize_power_flow(
        converged, i, string(T), residual, data, nothing, time_step)
end
