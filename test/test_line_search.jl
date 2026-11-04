# Backtracking (Armijo) line search on the Newton step (`solver_settings[:line_search] =
# :backtracking`, T86): the full step first, the step halved while ‖F(x+αΔx)‖₂ ≥ (1 − c·α)‖F(x)‖₂,
# taken untested once α ≤ the minimum step. Synthetic systems and PSB's c_sys14 only.

_ls_settings(; kw...) = Dict{Symbol, Any}(:line_search => :backtracking, kw...)

@testset "line search: settings are checked" begin
    @test PF._line_search_settings(:none, 1e-4, 0.5, 1e-3, false) === nothing
    ls = PF._line_search_settings(:backtracking, 1e-4, 0.5, 1e-3, false)
    @test ls == PF.BacktrackingLineSearch(1e-4, 0.5, 1e-3, :balance)
    @test PF.DEFAULT_LINE_SEARCH === :none
    # pyref's recipe (flatstart.newton_linesearch): Armijo c = 1e-4, halving, minimum 1e-3
    @test (PF.DEFAULT_LINE_SEARCH_C, PF.DEFAULT_LINE_SEARCH_SHRINK,
        PF.DEFAULT_LINE_SEARCH_MIN_STEP, PF.DEFAULT_LINE_SEARCH_NORM) ==
          (1e-4, 0.5, 1e-3, :balance)
    # an unknown method, parameters out of range, two step controls
    @test_throws ArgumentError ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:line_search => :armijo))
    for (k, v) in ((:line_search_c, 0.0), (:line_search_c, 1.0),
        (:line_search_shrink, 1.0), (:line_search_shrink, 0.0),
        (:line_search_min_step, 0.0), (:line_search_min_step, 1.5),
        (:line_search_norm, :max))
        @test_throws ArgumentError ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            solver_settings = _ls_settings(; k => v))
    end
    @test_throws ArgumentError ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = _ls_settings(; iwamoto = true))
    # Newton only, in every formulation's constructor
    for S in (TrustRegionACPowerFlow, LevenbergMarquardtACPowerFlow)
        @test_throws ArgumentError ACPowerFlow{S}(; solver_settings = _ls_settings())
        @test_throws ArgumentError ACRectangularPowerFlow{S}(; solver_settings = _ls_settings())
        @test_throws ArgumentError ACMixedPowerFlow{S}(; solver_settings = _ls_settings())
    end
    @test ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = _ls_settings()) isa ACRectangularPowerFlow
    # passed as a call kwarg, the same checks hold at solve time
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PowerFlowData(ACPowerFlow{TrustRegionACPowerFlow}(), sys)
    @test_throws ArgumentError solve_power_flow!(data; line_search = :backtracking)
    data = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
    @test_throws ArgumentError solve_power_flow!(data; line_search = :backtracking,
        iwamoto = true)
end

@testset "line search: the balance norm leaves out the rows a free injection absorbs" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PowerFlowData(ACPowerFlow(), sys)
    residual = PF.ACPowerFlowResidual(data, 1)
    bt = data.bus_type[:, 1]
    rows = PF._line_search_rows(residual, :balance, 1)
    expected = Int[]
    for (ix, t) in enumerate(bt)
        t == PSY.ACBusTypes.PQ && push!(expected, 2ix - 1, 2ix)
        t == PSY.ACBusTypes.PV && push!(expected, 2ix - 1)
    end
    @test rows == expected
    @test length(rows) == 2 * count(==(PSY.ACBusTypes.PQ), bt) +
                         count(==(PSY.ACBusTypes.PV), bt)
    @test PF._line_search_rows(residual, :full, 1) === nothing
    F = collect(1.0:length(residual.Rv))
    @test PF._line_search_norm(F, rows) ≈ norm(F[rows])
    @test PF._line_search_norm(F, nothing) == norm(F)
end

"""Solve `sys` with `settings` from its stored state (no enhanced flat start) and return
(converged, data, the log text)."""
function _ls_solve(sys; settings = Dict{Symbol, Any}(), kwargs...)
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        enhanced_flat_start = false, solver_settings = settings, kwargs...)
    data = PowerFlowData(pf, sys)
    io = IOBuffer()
    ok = with_logger(SimpleLogger(io, Logging.Info)) do
        solve_power_flow!(data)
    end
    return ok, data, String(take!(io))
end

@testset "line search: default behaviour unchanged, full steps on IEEE 14 equal plain Newton" begin
    mk() = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    ok_nr, nr, log_nr = _ls_solve(mk())
    @test ok_nr
    @test !occursin("line search", log_nr)
    ok_ls, ls, log_ls = _ls_solve(mk(); settings = _ls_settings())
    @test ok_ls
    # every step is the full Newton step, so the iterates are plain Newton's to the bit
    @test occursin(r"Backtracking line search: 0 of \d+ Newton steps shortened", log_ls)
    @test ls.bus_magnitude == nr.bus_magnitude
    @test ls.bus_angles == nr.bus_angles
    # min_step = 1: every step is taken untested, i.e. plain Newton
    ok_1, one, log_1 = _ls_solve(mk(); settings = _ls_settings(; line_search_min_step = 1.0))
    @test ok_1
    m = match(r"(\d+) of (\d+) Newton steps shortened \(0 step cuts\), (\d+) at the minimum",
        log_1)
    @test !isnothing(m) && m[1] == "0" && m[2] == m[3]
    @test one.bus_magnitude == nr.bus_magnitude
    # the rectangular formulation's Newton takes it too (full norm)
    pf = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = _ls_settings())
    @test solve_power_flow!(PowerFlowData(pf, mk()))
end

"""The no-load start (pyref `flatstart.noload_start`, T69): PV and REF buses held at their
stored |V| (the PV angles from a first solve held at the swing alone), every other bus from
`Y[u,u] V_u = −Y[u,h] V_h`, so every tap ratio is satisfied at once. Dense: small systems only.
Returns (|V|, θ) in `data`'s bus order."""
function _ls_noload_start(data)
    Y = Matrix(data.power_network_matrix.data)
    bt = data.bus_type[:, 1]
    vm = copy(data.bus_magnitude[:, 1])
    va = copy(data.bus_angles[:, 1])
    ref = findall(==(PSY.ACBusTypes.REF), bt)
    held = findall(t -> t == PSY.ACBusTypes.REF || t == PSY.ACBusTypes.PV, bt)
    function held_solve(h)
        u = setdiff(eachindex(bt), h)
        V = zeros(ComplexF64, length(bt))
        V[h] .= vm[h] .* cis.(va[h])
        V[u] .= Y[u, u] \ (-(Y[u, h] * V[h]))
        return V
    end
    pv = setdiff(held, ref)
    va[pv] .= angle.(held_solve(ref)[pv])
    V = held_solve(held)
    u = setdiff(eachindex(bt), held)
    vm[u] .= abs.(V[u])
    va[u] .= angle.(V[u])
    return vm, va
end

"""Solve `mk()` from the start `(vm, va)` with `settings`; returns (converged, data, log)."""
function _ls_solve_from(mk, vm, va; settings = Dict{Symbol, Any}(), kwargs...)
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        enhanced_flat_start = false, solver_settings = settings, kwargs...)
    data = PowerFlowData(pf, mk())
    data.bus_magnitude[:, 1] .= vm
    data.bus_angles[:, 1] .= va
    io = IOBuffer()
    ok = with_logger(SimpleLogger(io, Logging.Info)) do
        solve_power_flow!(data)
    end
    return ok, data, String(take!(io))
end

"""The same solution as `warm`: |V| equal and θ equal modulo 2π."""
function _ls_same_solution(data, warm; atol = 1e-8)
    dv = maximum(abs.(data.bus_magnitude[:, 1] .- warm.bus_magnitude[:, 1]))
    dθ = maximum(abs.(rem2pi.(data.bus_angles[:, 1] .- warm.bus_angles[:, 1], RoundNearest)))
    return dv < atol && dθ < atol
end

"""The line search's tally line: (shortened, steps, cuts, at the minimum step), or nothing."""
function _ls_tally(log)
    m = match(r"Backtracking line search: (\d+) of (\d+) Newton steps shortened \((\d+) step cuts\), (\d+) at the minimum step",
        log)
    return isnothing(m) ? nothing : parse.(Int, (m[1], m[2], m[3], m[4]))
end

@testset "line search: IEEE 14 from a rotated no-load start, plain Newton fails" begin
    mk() = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data0 = PowerFlowData(ACPowerFlow(; enhanced_flat_start = false), mk())
    vm, va = _ls_noload_start(data0)
    ok_w, warm, _ = _ls_solve_from(mk, vm, va)
    @test ok_w
    # every bus but the swing turned 100° against it: the swing's branches start past 90°
    nonref = [t != PSY.ACBusTypes.REF for t in data0.bus_type[:, 1]]
    va_bad = copy(va)
    va_bad[nonref] .+= deg2rad(100)
    ok_nr, _, _ = _ls_solve_from(mk, vm, va_bad)
    @test !ok_nr
    ok_ls, ls, log = _ls_solve_from(mk, vm, va_bad; settings = _ls_settings())
    @test ok_ls
    @test _ls_same_solution(ls, warm)
    t = _ls_tally(log)
    @test !isnothing(t) && t[1] >= 1 && t[3] >= 1
end

"""A synthetic generator step-up on three buses (T69): REF(1) feeds a star point (2, PQ, 50 MW
load) through four parallel windings at a 1.05 ratio and 3e-4 pu each; a PV generator (3,
100 MW, |V| 1.02) sits behind the star on a 0.3 pu winding (ratio 0.9565). With `ltc`, a 69 kV load bus (4) hangs off the REF bus through a
voltage-controlled tap (regulating bus 4)."""
function _ls_t69_system(; ltc::Bool = false)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    st = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    g = _add_simple_bus!(sys, 3, ACBusTypes.PV, 13.8, 1.02, 0.0)
    winding(name, from, to, x, tap; objective = PSY.TransformerControlObjective.FIXED) =
        add_component!(sys, TapTransformer(; name = name, available = true,
            active_power_flow = 0.0, reactive_power_flow = 0.0,
            arc = Arc(; from = from, to = to), r = 0.0, x = x, primary_shunt = 0.0im,
            tap = tap, rating = 10.0, base_power = 100.0, control_objective = objective))
    for u in 1:4
        winding("w1_$u", b1, st, 3e-4, 1.05)
    end
    winding("w2", g, st, 0.3, 0.9565)
    _add_simple_load!(sys, st, 50.0, 50.0 / 3)
    gen = _add_simple_thermal_standard!(sys, g, 1.0, 0.0)
    set_active_power_limits!(gen, (min = 0.0, max = 10.0))
    set_reactive_power_limits!(gen, (min = -10.0, max = 10.0))
    set_rating!(gen, 10.0)
    if ltc
        b4 = _add_simple_bus!(sys, 4, ACBusTypes.PQ, 69, 1.0, 0.0)
        add_component!(sys, TapTransformer(; name = "ltc", available = true,
            active_power_flow = 0.0, reactive_power_flow = 0.0,
            arc = Arc(; from = b1, to = b4), r = 0.005, x = 0.08, primary_shunt = 0.0im,
            tap = 1.0, rating = 10.0, base_power = 100.0,
            control_objective = PSY.TransformerControlObjective.VOLTAGE))
        _add_simple_load!(sys, b4, 40.0, 20.0)
    end
    return sys
end

"""The no-load start of `_ls_t69_system` with the generator's angle at −90°: its winding to the
star then carries no ∂P/∂θ, and the first Newton step turns the generator by thousands of
radians."""
function _ls_t69_bad_start(; ltc = false)
    data = PowerFlowData(ACPowerFlow(; enhanced_flat_start = false), _ls_t69_system(; ltc))
    vm, va = _ls_noload_start(data)
    va[PF.get_bus_lookup(data)[3]] = deg2rad(-90)
    return vm, va, _ls_noload_start(data)...
end

@testset "line search: T69's 1.05 ratio across 3e-4 pu from a bad start, Newton and Iwamoto fail" begin
    mk() = _ls_t69_system()
    vm, va, vm0, va0 = _ls_t69_bad_start()
    ok_w, warm, _ = _ls_solve_from(mk, vm0, va0)   # the warm solution, from the no-load start
    @test ok_w
    ok_nr, _, _ = _ls_solve_from(mk, vm, va)
    @test !ok_nr
    ok_iw, _, _ = _ls_solve_from(mk, vm, va; settings = Dict{Symbol, Any}(:iwamoto => true))
    @test !ok_iw
    for norm_rows in (:balance, :full)
        ok_ls, ls, log = _ls_solve_from(mk, vm, va;
            settings = _ls_settings(; line_search_norm = norm_rows))
        @test ok_ls
        @test _ls_same_solution(ls, warm)
        # Iwamoto's damped step fails three times at the start; the line search cuts the first
        # step down to the minimum step and takes it untested (pyref's rule), which moves the
        # state off the singular point, and the next full steps converge
        t = _ls_tally(log)
        @test !isnothing(t) && t[1] >= 1 && t[4] >= 1
        @test occursin("converged", log)
    end
end

@testset "line search: discrete control's batched path from T69's bad start" begin
    mk() = _ls_t69_system(; ltc = true)
    vm, va, vm0, va0 = _ls_t69_bad_start(; ltc = true)
    ok_w, warm, _ = _ls_solve_from(mk, vm0, va0; control_discrete_devices = true)
    @test ok_w
    tap_w = only(PF.get_controlled_devices(warm).taps).current
    @test tap_w != 1.0
    ok_nr, _, _ = _ls_solve_from(mk, vm, va; control_discrete_devices = true)
    @test !ok_nr
    ok_iw, _, _ = _ls_solve_from(mk, vm, va; control_discrete_devices = true,
        settings = Dict{Symbol, Any}(:iwamoto => true))
    @test !ok_iw
    ok_ls, ls, log = _ls_solve_from(mk, vm, va; control_discrete_devices = true,
        settings = _ls_settings())
    @test ok_ls
    @test _ls_same_solution(ls, warm)
    @test only(PF.get_controlled_devices(ls).taps).current == tap_w
    @test PF.get_control_batched_fallback_count(ls) == 0
    @test !isnothing(_ls_tally(log))
    # controls locked (the first use, L4's locked stage): the same start converges too
    ok_locked, locked, _ = _ls_solve_from(mk, vm, va; settings = _ls_settings())
    @test ok_locked
    ok_wl, warm_locked, _ = _ls_solve_from(mk, vm0, va0)
    @test ok_wl && _ls_same_solution(locked, warm_locked)
end
