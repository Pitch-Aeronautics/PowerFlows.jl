"""
    solve_and_store_power_flow!(pf::AbstractACPowerFlow{<:ACPowerFlowSolverType}, system::PSY.System; kwargs...)

Solves the power flow in the system and writes the solution into the relevant structs.
Updates active and reactive power setpoints for generators and active and reactive
power flows for branches (calculated in the From - To direction and in the To - From direction).

Configuration options like `time_steps`, `time_step_names`, `network_reductions`, and
`correct_bustypes` should be set on the `ACPowerFlow` object.

The bus types can be changed from PV to PQ if the reactive power limits are violated.

# Arguments
- [`pf::AbstractACPowerFlow{<:ACPowerFlowSolverType}`](@ref AbstractACPowerFlow): the power flow struct,
    which contains configuration options.
- `system::PSY.System`: The power system model, a [`PowerSystems.System`](@extref) struct.
- `kwargs...`: Additional keyword arguments passed to the solver.

When the solve ran with `control_discrete_devices`, the solved tap ratios / shunt admittances /
phase-shifter angles are written back into the system (see [`write_device_settings!`](@ref)): the
stored branch flows are only self-consistent with the mutated device settings, so the input
system's controlled devices are updated to the solved values. With no controls active this is a
no-op. This write-back only happens for single-period solves; for multiperiod solves
(`time_steps > 1`) it is skipped with a warning, since a PSY component holds one scalar per
setting and cannot round-trip a per-time-step schedule — use
[`get_controlled_device_results`](@ref) for the per-time-step schedule instead.

## Keyword Arguments
- `tol`: Infinite norm of residuals under which convergence is declared. Default is `1e-9`.
- `maxIterations`: Maximum number of Newton-Raphson iterations. Default is `30`.

# Returns
- `converged::Bool`: Indicates whether the power flow solution converged.
- The power flow results are written into the system struct.

# Examples

```julia
solve_and_store_power_flow!(pf, sys)

# With correct_bustypes enabled
pf = ACPowerFlow(; correct_bustypes = true)
solve_and_store_power_flow!(pf, sys)

# Passing solver keyword arguments
solve_and_store_power_flow!(pf, sys; maxIterations=100)
```
"""
function solve_and_store_power_flow!(
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    system::PSY.System;
    kwargs...,
)
    # converged must be defined in the outer scope to be visible for return
    converged = false
    with_units_base(system, PSY.UnitSystem.SYSTEM_BASE) do
        data = PowerFlowData(pf, system)

        converged = solve_power_flow!(data; kwargs...)

        if converged
            # Write moved device settings back BEFORE write_power_flow_solution! recomputes flows —
            # its consistency assertion compares against the stored (moved) flows. Self-guards to a
            # no-op when no discrete controls ran.
            write_device_settings!(system, data)
            write_power_flow_solution!(
                system,
                pf,
                data,
                get(kwargs, :maxIterations, DEFAULT_NR_MAX_ITER),
            )
            @info("PowerFlow solve converged, the results have been stored in the system")
        else
            @error("The power flow solver returned convergence = $converged")
        end
    end

    return converged
end

"""
    write_device_settings!(system, data)

Write the solved discrete-control device settings back into the `PSY.System`:
tap ratios (`set_tap!`), switched-shunt admittances (`set_Y!`/`set_initial_status!`,
convention-aware — see below), and phase-shifter angles (`set_α!`). FACTS devices
carry no stored setting field in PSY and are skipped. A device no longer present
in `system` is skipped with a `@warn` (its solved setting is not written back).
Mutates the user's system — called by [`solve_and_store_power_flow!`](@ref) after a
converged solve; a no-op when no discrete controls ran.

Switched shunts write back per their sourcing convention (see
[Metadata sourcing](@ref discrete-control-metadata) for how `psse_convention` is
determined): PSS/E-parsed (BINIT) components take the solved total straight into
`Y`; PSY API-built components keep `Y` at the fixed base and write the last-snap
`block_n` into `initial_status`, since overwriting `Y` would double-count the
status on re-enrollment. A never-snapped API-built device (continuous, or held in
its deadband the whole solve) whose `block_n` cannot reconstruct `d.current` falls
back to the BINIT write (solved total into `Y`, `initial_status` zeroed).

No-op for `time_steps > 1`: a PSY component holds a single scalar setting, but a
multiperiod solve produces one setting per time step, so there is no single value to
write back without silently discarding all but the last-processed step. Per-time-step
results remain available via [`get_controlled_device_results`](@ref).
"""
function write_device_settings!(system::PSY.System, data)
    set = get_controlled_devices(data)
    isnothing(set) && return
    if get_time_steps(data) > 1
        @warn "write_device_settings!: skipped — a PSY component holds a single scalar and \
            cannot round-trip a per-time-step schedule. Use get_controlled_device_results \
            for per-time-step device settings." maxlog = 1
        return
    end
    for d in set.taps
        tx = PSY.get_component(PSY.TapTransformer, system, d.name)
        if isnothing(tx)
            @warn "write_device_settings!: TapTransformer \"$(d.name)\" not found in the \
                system; its solved tap ratio $(d.current) was NOT written back."
            continue
        end
        PSY.set_tap!(tx, d.current)
    end
    for d in set.phase_shifters
        pst = PSY.get_component(PSY.PhaseShiftingTransformer, system, d.name)
        if isnothing(pst)
            @warn "write_device_settings!: PhaseShiftingTransformer \"$(d.name)\" not found \
                in the system; its solved angle $(d.current) was NOT written back."
            continue
        end
        PSY.set_α!(pst, d.current)
    end
    for d in set.shunts
        sa = PSY.get_component(PSY.SwitchedAdmittance, system, d.name)
        if isnothing(sa)
            @warn "write_device_settings!: SwitchedAdmittance \"$(d.name)\" not found in \
                the system; its solved susceptance $(d.current) was NOT written back."
            continue
        end
        if d.psse_convention
            PSY.set_Y!(sa, Complex(d.g0, d.current))
        else
            realizable = d.b0 + sum(d.block_n .* d.block_dB; init = 0.0)
            if abs(realizable - d.current) <= BOUNDS_TOLERANCE
                PSY.set_Y!(sa, Complex(d.g0, d.b0))
                PSY.set_initial_status!(sa, copy(d.block_n))
            else
                PSY.set_Y!(sa, Complex(d.g0, d.current))
                PSY.set_initial_status!(sa, zeros(Int, length(d.block_n)))
            end
        end
    end
    for d in set.facts
        fd = PSY.get_component(PSY.FACTSControlDevice, system, d.name)
        if isnothing(fd)
            @warn "write_device_settings!: FACTSControlDevice \"$(d.name)\" not found in \
                the system; its solved reactive output was NOT written back."
            continue
        end
        # Delivered reactive power Q = b·|V_local|² (MVA) at the device's own bus.
        PSY.set_reactive_power_required!(
            fd, delivered_q_mvar(d, data.bus_magnitude[d.bus_ix, 1]))
    end
    return
end

"""
Similar to [solve\\_and\\_store\\_power\\_flow!](@ref) but does not update the system struct with results.
Returns the results in a dictionary of dataframes.

## Examples

```julia
res = solve_power_flow(pf, sys)
res = solve_power_flow(pf, sys, FlowReporting.BRANCH_FLOWS)
```
"""
function solve_power_flow(
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    system::PSY.System;
    kwargs...,
)
    return solve_power_flow(pf, system, FlowReporting.ARC_FLOWS; kwargs...)
end

function solve_power_flow(
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    system::PSY.System,
    flow_reporting::FlowReporting;
    kwargs...,
)
    # df_results must be defined in the outer scope first to be visible for return
    df_results = Dict{String, DataFrames.DataFrame}()
    converged = false
    time_step = 1
    with_units_base(system, PSY.UnitSystem.SYSTEM_BASE) do
        data = PowerFlowData(pf, system)

        converged = solve_power_flow!(data; kwargs...)

        if converged
            @info("PowerFlow solve converged, the results are exported in DataFrames")
            df_results = write_results(pf, system, data, time_step, flow_reporting)
        else
            df_results = missing
            @error("The power flow solver returned convergence = $(converged)")
        end
    end

    return df_results
end

"""
    solve_power_flow!(data::ACPowerFlowData; kwargs...)

Solve the multiperiod AC power flow problem for the given power flow data.

The bus types can be changed from PV to PQ if the reactive power limits are violated.
The power flow solver settings are taken from the `ACPowerFlow` object stored in `data`.

# Arguments
- [`data::ACPowerFlowData`](@ref ACPowerFlowData): The power flow data containing the grid information and initial conditions.
- `kwargs...`: Additional keyword arguments. If these overlap with those in the 
    `solver_settings` of the `ACPowerFlow` object, the values in `kwargs` take precedence.

# Keyword Arguments
- `time_steps`: Specifies the time steps to solve. Defaults to sorting and collecting the keys of `get_time_step_map(data)`.

# Description
This function solves the AC power flow problem for each time step specified in `data`.
It preallocates memory for the results and iterates over the sorted time steps.
    For each time step, it calls the `_ac_power_flow` function to solve the power flow equations and updates the `data` object with the results.
    If the power flow converges, it updates the active and reactive power injections, as well as the voltage magnitudes and angles for different bus types (REF, PV, PQ), and calculates that time step's branch power flows.
    If the power flow does not converge, it sets the corresponding entries in `data` to `NaN`.

# Notes
- If the grid topology changes (e.g., tap positions of transformers or in-service status of branches), the admittance matrices `Yft` and `Ytf` must be updated before that time step's branch flows are computed.

# Examples
```julia
solve_power_flow!(data)
```
"""
# The solve chain below specializes on its keyword arguments' NamedTuple type, whose field order
# here is a `Dict`'s iteration order (`solver_settings` merged with the call's kwargs), which follows
# hashes and insertion history. Sorted by key, each key set has one type, however the `Dict` was built.
_sorted_kwargs(d::AbstractDict{Symbol}) = (; (k => d[k] for k in sort!(collect(keys(d))))...)

function solve_power_flow!(
    data::ACPowerFlowData;
    kwargs...,
)
    pf = get_pf(data)
    # Merge solver_settings from pf with any explicitly passed kwargs (explicit kwargs take precedence)
    merged_kwargs = _sorted_kwargs(merge(get_solver_kwargs(pf), kwargs))
    sorted_time_steps =
        get(merged_kwargs, :time_steps, sort(collect(keys(get_time_step_map(data)))))
    # This can be done from PSI by directly writing to `data`'s fields; we just don't
    # do it here in PF alone.
    if length(sorted_time_steps) > 1
        @warn(
            "Multi-period AC power flow: each time step is solved independently " *
            "using the same network data. Time-varying generator setpoints or " *
            "limits are not updated between time steps.",
            maxlog = 1,
        )
    end
    # preallocate results
    ts_converged = fill(false, length(sorted_time_steps))

    Yft = data.power_network_matrix.arc_admittance_from_to
    Ytf = data.power_network_matrix.arc_admittance_to_from
    @assert PNM.get_bus_lookup(Yft) == get_bus_lookup(data)
    arcs = PNM.get_arc_axis(Yft)
    @assert arcs == PNM.get_arc_axis(Ytf)
    @assert length(PNM.get_bus_axis(Yft)) == length(data.bus_angles[:, 1])
    bus_lookup = get_bus_lookup(data)
    fb_ix = [bus_lookup[bus_no] for bus_no in first.(arcs)]  # from bus indices
    tb_ix = [bus_lookup[bus_no] for bus_no in last.(arcs)]   # to bus indices
    @assert length(fb_ix) == length(arcs)

    cd = get_controlled_devices(data)
    validate_device_store_width(cd, get_time_steps(data))
    for (ts_pos, time_step) in enumerate(sorted_time_steps)
        load_device_state!(cd, data, time_step)
        converged = _ac_power_flow_with_area_relax!(data, pf, time_step; merged_kwargs...)
        save_device_state!(cd, data, time_step)
        ts_converged[ts_pos] = converged
        converged && _warn_vsc_limit_violations(data, time_step)

        if OVERWRITE_NON_CONVERGED && !converged
            # set values to NaN for not converged time steps
            data.bus_active_power_injections[:, time_step] .= NaN
            data.bus_active_power_withdrawals[:, time_step] .= NaN
            data.bus_active_power_constant_current_withdrawals[:, time_step] .= NaN
            data.bus_active_power_constant_impedance_withdrawals[:, time_step] .= NaN
            data.bus_reactive_power_injections[:, time_step] .= NaN
            data.bus_reactive_power_withdrawals[:, time_step] .= NaN
            data.bus_reactive_power_constant_current_withdrawals[:, time_step] .= NaN
            data.bus_reactive_power_constant_impedance_withdrawals[:, time_step] .= NaN
            data.bus_magnitude[:, time_step] .= NaN
            data.bus_angles[:, time_step] .= NaN
        elseif get_lcc_count(data) > 0 && converged
            # calculate branch flows for LCCs: their self-admittances may change.
            V =
                data.bus_magnitude[:, time_step] .*
                exp.(1im .* data.bus_angles[:, time_step])
            for (i, (bus_indices, self_admittances)) in
                enumerate(zip(data.lcc.bus_indices, data.lcc.branch_admittances))
                (rectifier_ix, inverter_ix) = bus_indices
                (rectifier_y, inverter_y) = self_admittances
                S_inverter = V[inverter_ix] * conj(inverter_y * V[inverter_ix])
                S_rectifier = V[rectifier_ix] * conj(rectifier_y * V[rectifier_ix])
                data.lcc.arc_active_power_flow_from_to[i, time_step] =
                    real(S_rectifier)
                data.lcc.arc_reactive_power_flow_from_to[i, time_step] =
                    imag(S_rectifier)
                data.lcc.arc_active_power_flow_to_from[i, time_step] =
                    real(S_inverter)
                data.lcc.arc_reactive_power_flow_to_from[i, time_step] =
                    imag(S_inverter)
            end
        end

        # Per-step branch flows (not batched after the loop) so a future per-step Yft/Ytf
        # (e.g. varying tap positions) is used correctly.
        # NOTE PNM's structs use ComplexF32, while the system objects store Float64's.
        #      so if you set the system bus angles/voltages to match these fields, then repeat
        #      this math using the system voltages, you'll see differences in the flows, ~1e-4.
        step_V =
            data.bus_magnitude[:, time_step] .* exp.(1im .* data.bus_angles[:, time_step])
        Sft = step_V[fb_ix] .* conj.(Yft.data * step_V)
        Stf = step_V[tb_ix] .* conj.(Ytf.data * step_V)
        data.arc_active_power_flow_from_to[:, time_step] .= real.(Sft)
        data.arc_reactive_power_flow_from_to[:, time_step] .= imag.(Sft)
        data.arc_active_power_flow_to_from[:, time_step] .= real.(Stf)
        data.arc_reactive_power_flow_to_from[:, time_step] .= imag.(Stf)

        _compute_arc_angle_differences_from_indices!(data, fb_ix, tb_ix, time_step)
    end

    data.converged[sorted_time_steps] .= ts_converged

    return all(ts_converged)
end

function _solve_with_q_limits!(
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    data::ACPowerFlowData,
    time_step::Int64;
    kwargs...,
)
    check_reactive_power_limits = pf.check_reactive_power_limits
    converged = false
    empty!(data.q_limit_switches.returned_this_loop)

    for _ in 1:MAX_REACTIVE_POWER_ITERATIONS
        converged = _newton_power_flow(pf, data, time_step; kwargs...)
        if !converged || !check_reactive_power_limits ||
           _check_q_limit_bounds!(data, time_step)
            return converged
        end
    end

    # Iteration cap reached: the last `_check_q_limit_bounds!` switched one or more buses (PV to
    # PQ with their Q clamped, or PQ back to PV at their setpoint) without a follow-up solve, so
    # `data`'s voltages no longer match its bus types. Pin that final PV/PQ assignment and solve
    # once more so the returned state is self-consistent, then return THAT solve's actual
    # convergence (not a forced `true`) — the classic "fix-as-PQ after N iterations" resolution,
    # rather than discarding a solution that does converge.
    @warn(
        "reactive power limits still oscillating after $MAX_REACTIVE_POWER_ITERATIONS \
        iterations; pinning the final PV/PQ assignment and solving once more"
    )
    return _newton_power_flow(pf, data, time_step; kwargs...)
end

function _ac_power_flow(
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    time_step::Int64;
    kwargs...,
)
    _check_remote_voltage_solver(pf, data)
    _reset_q_limit_flip_flops!(data, time_step)
    cd = data.controlled_devices
    if isnothing(cd) || isempty(cd)
        return _solve_with_q_limits!(pf, data, time_step; kwargs...)
    end
    return _control_continuation!(pf, data, time_step; kwargs...)
end

"""
    _ac_power_flow_with_area_relax!(data, pf, time_step; kwargs...) -> Bool

Wraps `_ac_power_flow` with greedy-relax handling for embedded area net-interchange
control: on non-convergence with areas still enrolled, de-enroll the worst-`|r_a|` area
and re-solve, warm-started with surviving areas' `ΔP_a` re-seeded from the `delta_p`
mirror; repeat until convergence or exhaustion. Exhaustion while still failing is genuine
network non-convergence plus a terminal diagnostic (`_report_area_interchange_failure`).
Relaxation is never silent: an `@error` at each de-enrollment and a solve-end summary;
converging after a relax still returns `true`.

Resets to the full pristine enrollment before each time step's attempt
(`_ensure_pristine_area_set!`) so a previous step's relax never carries over. The
never-enrolled short-circuit deliberately tests the PRISTINE set, not the WORKING one —
a previous time step's relax may have emptied the working set, and short-circuiting on it
would permanently disable area control for the rest of `data`'s lifetime.
"""
function _ac_power_flow_with_area_relax!(
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    time_step::Int64;
    kwargs...,
)
    aid = data.area_interchange
    isempty(aid.pristine_areas) &&
        return _ac_power_flow(data, pf, time_step; kwargs...)
    _ensure_pristine_area_set!(data, time_step)
    relaxed_this_step = RelaxedAreaRecord[]
    converged = false
    while true
        converged = _ac_power_flow(data, pf, time_step; kwargs...)
        converged && break
        iszero(n_controlled_areas(data)) && break
        gaps = _area_residual_gaps(data, time_step)
        gap, worst_ix = findmax(abs, gaps)
        area = data.area_interchange.areas[worst_ix]
        @error "Area interchange: Newton did not converge with area \"$(area.name)\" " *
               "controlled (target PDES = $(area.pdes), NI gap at the failed iterate = " *
               "$gap); de-enrolling it and re-solving with the remaining " *
               "$(n_controlled_areas(data) - 1) controlled area(s)."
        push!(relaxed_this_step, RelaxedAreaRecord(area.name, area.pdes))
        _deenroll_area!(data, worst_ix)
    end
    if !converged
        _report_area_interchange_failure(data, time_step)
        return converged
    end
    _sync_pristine_delta_p!(data, time_step)
    _warn_area_violations(data, time_step)
    isempty(relaxed_this_step) && return converged
    data.area_interchange.relaxed[time_step] = relaxed_this_step
    pristine_tail_of = Dict(a.name => a.tail_ix for a in aid.pristine_areas)
    relaxed_detail = join(
        (
            let tail_ix = pristine_tail_of[r.name],
                ni_solved = _area_net_interchange(
                    aid.pristine_ties, aid.pristine_dc_ties, tail_ix, data,
                    time_step,
                )

                "$(r.name) (ni_solved=$(ni_solved), pdes=$(r.pdes), " *
                "gap=$(ni_solved - r.pdes))"
            end
            for r in relaxed_this_step
        ),
        ", ",
    )
    @error "Area interchange: time step $time_step converged only after relaxing " *
           "$(length(relaxed_this_step)) area(s): $relaxed_detail. Their schedules were " *
           "infeasible given network/tie capacity."
    return converged
end

# Each time step's solve starts with no flip-flops counted and no bus capped.
function _reset_q_limit_flip_flops!(data::ACPowerFlowData, time_step::Int)
    st = data.q_limit_switches
    filter!(kv -> kv[1][2] != time_step, st.flip_flops)
    filter!(k -> k[2] != time_step, st.capped)
    _reset_remote_voltage_returns!(data, time_step)
    return
end

"""
    _check_q_limit_bounds!(data, time_step) -> Bool

The reactive-limit check after a converged solve, as PSS/E applies it (VARLIM):

- a PV bus whose generation Q is outside its bounds is released to PQ with Q clamped to the
  violated bound, and the |V| it held (its setpoint) is recorded;
- a bus released that way returns to PV at its setpoint when its voltage has crossed to the other
  side: Q at Qmax with |V| above the setpoint, or at Qmin with |V| below it. A unit truly at its
  limit sits on the near side (at Qmax it cannot hold |V| up to the setpoint) and stays PQ.

Without the return a release is one-way, so a unit pushed past a limit by a transient (a
discrete-control continuation's early passes) ends at that limit on the wrong side of its
setpoint. Every inner solve of the continuation runs this check, so the return happens there too,
before the discrete snap.

The bounds are the bus's (the sum over its units, as for the release), so a multi-unit bus
switches as one. A unit that regulates a remote bus by IREG (`_generator_remote_bus`) is part of a
remote-regulation group (`remote_voltage_control.jl`), checked by `_remote_voltage_q_limits!`: a
plant at a limit drops out of its group's share, the regulated bus becomes PQ when every plant is
fixed, and returns to PV at its setpoint by the same rule as below. With `remote_voltage_control
= false` (or the fast decoupled solver) such a unit holds its own terminal, and its setpoint here
is its terminal's stored |V|. A bus with no Q range (Qmin = Qmax) never returns. A return the next solve of the same
Q-limit loop undoes is a flip-flop; after `MAX_Q_LIMIT_FLIP_FLOPS` in one time step's solve the
bus stays PQ, with a warning.

Returns `true` when no bus switched.
"""
function _check_q_limit_bounds!(
    data::ACPowerFlowData,
    time_step::Int64,
)
    st = data.q_limit_switches
    within_limits = true
    bus_types = view(data.bus_type, :, time_step)
    rvc = data.remote_voltage_control
    for (ix, bt) in enumerate(bus_types)
        # Remote-regulation groups switch as groups (`_remote_voltage_q_limits!`).
        _rv_role(rvc, ix) && continue
        if bt == PSY.ACBusTypes.PV
            _q_limit_release!(data, st, ix, time_step) && (within_limits = false)
        elseif bt == PSY.ACBusTypes.PQ
            _q_limit_return!(data, st, ix, time_step) && (within_limits = false)
        end
    end
    _remote_voltage_q_limits!(data, time_step) && (within_limits = false)
    return within_limits
end

function _q_limit_release!(data::ACPowerFlowData, st::QLimitSwitchState, ix::Int, ts::Int)
    Q_gen = data.bus_reactive_power_injections[ix, ts]
    Q_min, Q_max = data.bus_reactive_power_bounds[ix, ts]
    if Q_min - BOUNDS_TOLERANCE <= Q_gen <= Q_max + BOUNDS_TOLERANCE
        @debug "Within Limits"
        return false
    end
    bus_name = data.power_network_matrix.axes[1][ix]
    key = (ix, ts)
    if key in st.returned_this_loop
        n = st.flip_flops[key] = get(st.flip_flops, key, 0) + 1
        if n >= MAX_Q_LIMIT_FLIP_FLOPS && !(key in st.capped)
            push!(st.capped, key)
            @warn "Bus $bus_name switched PQ→PV→PQ $n times at time step $ts \
                (MAX_Q_LIMIT_FLIP_FLOPS); it stays PQ for the rest of this solve"
        end
    end
    @info "Bus $bus_name changed to PSY.ACBusTypes.PQ"
    st.vset[key] = data.bus_magnitude[ix, ts]
    data.bus_type[ix, ts] = PSY.ACBusTypes.PQ
    data.bus_reactive_power_injections[ix, ts] = clamp(Q_gen, Q_min, Q_max)
    return true
end

function _q_limit_return!(data::ACPowerFlowData, st::QLimitSwitchState, ix::Int, ts::Int)
    key = (ix, ts)
    vset = get(st.vset, key, NaN)
    (isnan(vset) || key in st.capped) && return false
    Q_min, Q_max = data.bus_reactive_power_bounds[ix, ts]
    Q_max - Q_min > BOUNDS_TOLERANCE || return false
    Q = data.bus_reactive_power_injections[ix, ts]
    V = data.bus_magnitude[ix, ts]
    at_max = Q >= Q_max - BOUNDS_TOLERANCE
    at_min = Q <= Q_min + BOUNDS_TOLERANCE
    tol = Q_LIMIT_RETURN_VOLTAGE_TOLERANCE
    ((at_max && V > vset + tol) || (at_min && V < vset - tol)) || return false
    bus_name = data.power_network_matrix.axes[1][ix]
    @info "Bus $bus_name changed back to PSY.ACBusTypes.PV: Q at its \
        $(at_max ? "maximum" : "minimum") with |V| = $V $(at_max ? "above" : "below") its \
        setpoint $vset"
    data.bus_type[ix, ts] = PSY.ACBusTypes.PV
    data.bus_magnitude[ix, ts] = vset
    push!(st.returned_this_loop, key)
    st.returns[] += 1
    key in st.stored_starts && push!(st.stored_start_returns, key)
    return true
end

"""Tolerance (p.u. Q, 0.01 MVAr on a 100 MVA base) within which a local unit's stored reactive
power counts as at its limit for `_start_local_units_at_stored_limits!`. Tight on purpose: PSS/E
writes a unit that cannot hold its setpoint exactly at its limit, so a unit off its setpoint and
further from its limit than this is a stored state to report, not one to start from."""
const STORED_Q_LIMIT_TOLERANCE = 1e-4

_stored_limit_start_enabled(pf::PowerFlowEvaluationModel) =
    get_start_at_stored_q_limits(pf) && pf.check_reactive_power_limits
_stored_limit_start_enabled(::ACPolarPowerFlow{<:FastDecoupledACPowerFlow}) = false

"""The scheduled voltage of the local units at each bus index: the units' VS when every available
generator at the bus regulates that bus and states the same VS, and those units account for the
bus's Q bounds. A bus that fails this is absent from the result: without its own VS a unit has no
side to be on, and the VS of a unit with a remote IREG belongs to the bus it regulates."""
function _local_unit_setpoints(data, pf, sys)
    bus_lookup = get_bus_lookup(data)
    rmap = PNM.get_reverse_bus_search_map(
        PNM.get_network_reduction_data(data.power_network_matrix))
    units = Dict{Int, Vector{Any}}()
    for g in PSY.get_available_components(PSY.Generator, sys)
        bix = _resolve_bus_ix(bus_lookup, rmap, PSY.get_number(PSY.get_bus(g)))
        isnothing(bix) || push!(get!(units, bix, Any[]), g)
    end
    setpoints = Dict{Int, Float64}()
    for (bix, gs) in units
        all(gs) do g
            r = _generator_remote_bus(g)
            isnothing(r) || _resolve_bus_ix(bus_lookup, rmap, r) == bix
        end || continue
        vs = [_generator_remote_setpoint(g) for g in gs]
        any(isnothing, vs) && continue
        maximum(vs) - minimum(vs) > REMOTE_VOLTAGE_VS_AGREEMENT && continue
        _, lo, hi = _rv_plant_q_and_bounds(gs, pf)
        all(ts -> begin
                b = data.bus_reactive_power_bounds[bix, ts]
                abs(b[1] - lo) <= BOUNDS_TOLERANCE && abs(b[2] - hi) <= BOUNDS_TOLERANCE
            end, axes(data.bus_reactive_power_bounds, 2)) || continue
        setpoints[bix] = first(vs)
    end
    return setpoints
end

"""
    _start_local_units_at_stored_limits!(data, pf, sys)

The groups' saturated start (`_rv_stored_saturation`), for single units. Limits apply only after a
converged solve, but a local unit stored at a Q limit on PSS/E's side of its VS is PQ in PSS/E
(VARLIM 0) from the first iteration. Held PV at its stored |V| it can screen a remote-regulation
group's regulated bus from the group's plants, so that the Jacobian has no route from the group's
Q total to that bus's row, and the first solve never converges.

So such a bus starts PQ at the limit, with its VS kept as its setpoint for `_q_limit_return!`:
Qmax with |V| below VS, or Qmin with |V| above it, within `STORED_Q_LIMIT_TOLERANCE` of the limit.
A unit the solve moves across its VS returns to PV there. The final state is consistent with
VARLIM whichever way the start was read: at a limit on the near side of VS, or PV at VS.

Reads `data` as built, so the stored Q and stored |V|: with `start_at_stored_q_limits = false` (or
Q limits off, or a fast decoupled solver) every such unit stays PV at its stored |V| and limits
apply after a converged solve. Leaves alone a bus in a remote-regulation group, and a bus whose
units do not all state the same VS for themselves (`_local_unit_setpoints`). Called once, at build,
after `_enroll_remote_voltage_control!`.
"""
function _start_local_units_at_stored_limits!(data, pf, sys)
    _stored_limit_start_enabled(pf) || return
    setpoints = _local_unit_setpoints(data, pf, sys)
    isempty(setpoints) && return
    st = data.q_limit_switches
    rvc = data.remote_voltage_control
    tol = Q_LIMIT_RETURN_VOLTAGE_TOLERANCE
    for (ix, vs) in setpoints, ts in axes(data.bus_type, 2)
        data.bus_type[ix, ts] == PSY.ACBusTypes.PV || continue
        _rv_role(rvc, ix) && continue
        V = data.bus_magnitude[ix, ts]
        Q = data.bus_reactive_power_injections[ix, ts]
        lo, hi = data.bus_reactive_power_bounds[ix, ts]
        lim = V < vs - tol ? hi : V > vs + tol ? lo : NaN
        (isfinite(lim) && abs(Q - lim) <= STORED_Q_LIMIT_TOLERANCE) || continue
        key = (ix, ts)
        st.vset[key] = vs
        push!(st.stored_starts, key)
        data.bus_type[ix, ts] = PSY.ACBusTypes.PQ
        data.bus_reactive_power_injections[ix, ts] = lim
    end
    n = count(k -> k[2] == 1, st.stored_starts)
    n > 0 && @info "Q limits: $n local unit bus(es) start PQ at the limit they are stored at, \
        on PSS/E's side of their scheduled voltage."
    return
end

function bus_type_idx(
    data::ACPowerFlowData,
    time_step::Int64 = 1,
    bus_types::Tuple{Vararg{PSY.ACBusTypes}} = (
        PSY.ACBusTypes.REF,
        PSY.ACBusTypes.PV,
        PSY.ACBusTypes.PQ,
    ),
)
    # Find indices for each bus type
    return [
        findall(==(bus_type), data.bus_type[:, time_step]) for bus_type in bus_types
    ]
end
