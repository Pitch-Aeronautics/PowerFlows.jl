# Remote voltage regulation by generators (PSS/E IREG, shared by RMPCT).
#
# In PSS/E a plant whose IREG names another bus X holds |V_X| at its scheduled voltage VS, not
# its own terminal, and the plants that name the same X (with X's own plant, when X has one)
# share the reactive power that takes in proportion to their RMPCT. PowerSystems has no field
# for either, so they are read from the generator's `ext` (the data contract below), and
# PowerFlows solves them in the polar formulation without new state:
#
# - X is PV in the solve: |V_X| is held at the group's setpoint, and its first state slot,
#   x[2X−1], holds Q_tot, the reactive power of the group's sharing plants.
# - Each member plant's bus G is PQ: |V_G| is free, and G's reactive injection is
#   (s_G / Σ s) · Q_tot, where s are the RMPCT shares of the members still sharing.
# - X's own injection keeps its fixed part (`q_base`, X's loads and non-regulating sources) and,
#   when X has a regulating plant of its own, that plant's share.
#
# The Jacobian column of Q_tot carries −s_G/Σs at each sharing member's Q row (a structural
# entry is added where G is not X's neighbour), and 0 at X's own Q row unless X's plant shares.
#
# Q limits (with `check_reactive_power_limits`): a member whose share passes its bus's summed
# bounds is fixed at that bound and the shares renormalise over the rest; when every member is
# fixed X becomes PQ (the group saturates). The reverse follows PSS/E's VARLIM return (GAP-027):
# a fixed member rejoins when the share it would get falls back inside its bound, and a
# saturated group returns to PV at its setpoint when |V_X| has crossed it (above with the
# members at Qmax, below at Qmin).
# A group saturated in the stored state (the plants' total at their summed limit on the side PSS/E
# leaves it, Σ Qmax with |V_X| below the setpoint or Σ Qmin above) starts saturated, each plant at
# its stored Q and X PQ at its stored |V|, so a warm start reproduces the stored point; the return
# rule frees it if |V_X| crosses the setpoint.
#
# Data contract (each key in `ext[key]` or `ext["psse"][key]`, upper or lower case under "psse"):
# - `IREG`: the regulated bus number; 0, absent or the unit's own bus is local
#   (`_generator_remote_bus`).
# - `RMPCT`: the unit's percent share; absent ⇒ 100 (PSS/E's default); not positive ⇒ 100 with a
#   warning. A plant is the available generators at a bus: its share is its RMPCT counted once
#   per PSS/E bus (`_rv_plant_share`), its
#   limits their summed Q limits; other injections at the bus (FACTS, sources) stay fixed.
# - `VS`: the scheduled voltage of the regulated bus. Absent ⇒ X's stored |V| (PowerSystems'
#   v33 reader keeps IREG and RMPCT but drops VS, and its v35 reader drops IREG and RMPCT).
#   The largest-share plant's VS is used, X's own plant's included.

"""One remote-regulation group: the plants (`members`, bus indices) that hold the voltage of
`regulated_ix` at `vset`, with their RMPCT shares. When the regulated bus has a regulating plant
of its own it is a member too. A member's status at time step `ts` is
`status[offset + k, ts]` in [`RemoteVoltageControl`](@ref)."""
struct RemoteVoltageGroup
    regulated_ix::Int
    regulated_number::Int
    members::Vector{Int}
    member_numbers::Vector{Int}
    shares::Vector{Float64}
    offset::Int
    vset::Float64
end

"""Remote voltage regulation state of an `ACPowerFlowData` (GAP-015). Empty when no generator
regulates a remote bus, or when `remote_voltage_control` is off.

- `regulated_group[ix]`: the group whose regulated bus is `ix`, else 0.
- `member_group[ix]`: the group whose member plant sits at `ix` (the regulated bus's own plant
  included), else 0.
- `status[row, ts]`: 0 sharing, +1 fixed at its bus's Qmax, −1 fixed at its Qmin, `RV_RELEASED`
  out of its group (`remote_voltage_release!`: its plant out of service, its bus dead or away from
  X, or X gone): no share, no injection written, no limit check, its bus an ordinary bus again.
- `q_base[g, ts]`: the regulated bus's reactive injection outside the group (its stored injection
  when it has no regulating plant of its own, else 0).
- `q_fixed[row, ts]`: a member bus's reactive injection outside its plant (FACTS, sources and
  other non-generator injections), kept fixed; the plant is the bus's available generators.
- `q_bounds[row]`: the plant's reactive limits, summed over its generators.
- `q_held[row, ts]`: the Q a fixed plant holds: the limit the reactive-limit check fixed it at, or
  its stored Q when its group started saturated (`_rv_stored_saturation`).
- `returns`: rejoins per (member row, time step) in the running time step's solve, capped at
  `MAX_Q_LIMIT_FLIP_FLOPS`."""
mutable struct RemoteVoltageControl
    groups::Vector{RemoteVoltageGroup}
    regulated_group::Vector{Int}
    member_group::Vector{Int}
    status::Matrix{Int8}
    q_base::Matrix{Float64}
    q_fixed::Matrix{Float64}
    q_bounds::Vector{Tuple{Float64, Float64}}
    q_held::Matrix{Float64}
    returns::Dict{Tuple{Int, Int}, Int}
end

"""Member status of a plant taken out of its group (`remote_voltage_release!`)."""
const RV_RELEASED = Int8(2)

RemoteVoltageControl() = RemoteVoltageControl(
    RemoteVoltageGroup[], Int[], Int[], zeros(Int8, 0, 0), zeros(0, 0), zeros(0, 0),
    Tuple{Float64, Float64}[], zeros(0, 0), Dict{Tuple{Int, Int}, Int}())

has_remote_voltage_groups(rvc::RemoteVoltageControl) = !isempty(rvc.groups)
_rv_status_column(rvc::RemoteVoltageControl, ts::Int) =
    isempty(rvc.groups) ? Int8[] : rvc.status[:, ts]

# True when bus `ix` is a regulated bus or a member plant's bus of some group.
@inline function _rv_role(rvc::RemoteVoltageControl, ix::Int)
    isempty(rvc.regulated_group) && return false
    @inbounds return rvc.regulated_group[ix] != 0 || rvc.member_group[ix] != 0
end

@inline function _rv_active_share(rvc::RemoteVoltageControl, grp::RemoteVoltageGroup, ts::Int)
    S = 0.0
    @inbounds for k in eachindex(grp.members)
        rvc.status[grp.offset + k, ts] == 0 && (S += grp.shares[k])
    end
    return S
end

# The Q a fixed plant (status ≠ 0) holds.
@inline _rv_held(rvc::RemoteVoltageControl, row::Int, ts::Int) = rvc.q_held[row, ts]

# Member `k`'s plant reactive power: its bus's injection less what is not its plant (`q_fixed`:
# FACTS, sources and other non-generator injections at the bus; and, at the regulated bus,
# `q_base`, which is 0 when X has a plant of its own).
@inline function _rv_member_q(data, rvc, g::Int, grp::RemoteVoltageGroup, k::Int, ts::Int)
    b = grp.members[k]
    q = data.bus_reactive_power_injections[b, ts] - rvc.q_fixed[grp.offset + k, ts]
    return b == grp.regulated_ix ? q - rvc.q_base[g, ts] : q
end

"""Read a key from a component's `ext`, or from `ext["psse"]` under the key or its lower case."""
function _ext_psse_value(c, key::String)
    ext = PSY.get_ext(c)
    v = get(ext, key, nothing)
    v isa Real && return v
    psse = get(ext, "psse", nothing)
    psse isa AbstractDict || return nothing
    v = get(psse, key, get(psse, lowercase(key), nothing))
    return v isa Real ? v : nothing
end

"""A generator's RMPCT share (percent): `ext["RMPCT"]` or `ext["psse"]["RMPCT"/"rmpct"]`;
100 (PSS/E's default) when absent, and when not positive (with a warning)."""
function _generator_rmpct(g)
    r = _ext_psse_value(g, "RMPCT")
    isnothing(r) && return 100.0
    if !(isfinite(r) && r > 0)
        @warn "Generator $(PSY.get_name(g)): RMPCT $r is not positive; using 100." maxlog =
            PF_MAX_LOG
        return 100.0
    end
    return Float64(r)
end

"""A plant's RMPCT share, counted as PSS/E counts it: once per PSS/E bus (every machine record at a
bus carries its plant's RMPCT, so two machines at 20 % share as one plant at 20 %), the largest
over that bus's available machines (with a warning when they disagree), summed over the PSS/E buses
a zero-impedance reduction merged into this one."""
function _rv_plant_share(gens)
    per_bus = Dict{Int, Float64}()
    for g in gens
        n = PSY.get_number(PSY.get_bus(g))
        r = _generator_rmpct(g)
        prev = get(per_bus, n, nothing)
        if !isnothing(prev) && abs(prev - r) > 1e-9
            @warn "Generators at bus $n carry different RMPCT ($prev, $r); PSS/E keeps one per \
                plant: using the largest." maxlog = PF_MAX_LOG
        end
        per_bus[n] = isnothing(prev) ? r : max(prev, r)
    end
    return sum(values(per_bus); init = 0.0)
end

"""A generator's scheduled voltage for its regulated bus (`ext["VS"]` or
`ext["psse"]["VS"/"vs"]`), or `nothing`."""
function _generator_remote_setpoint(g)
    v = _ext_psse_value(g, "VS")
    (isnothing(v) || !(isfinite(v) && v > 0)) && return nothing
    return Float64(v)
end

_remote_voltage_enabled(pf::PowerFlowEvaluationModel) = get_remote_voltage_control(pf)
_remote_voltage_enabled(
    ::ACPolarPowerFlow{<:FastDecoupledACPowerFlow},
) = false

# The fast decoupled solvers do not model remote regulation (their explicit PV update closes a
# regulated bus's own Q row with its Q_tot slot), so a solve of data that holds remote-regulation
# groups with one of them is refused rather than solved wrong. Data built for a fast decoupled solver
# has no groups (`_enroll_remote_voltage_control!`).
_check_remote_voltage_solver(::PowerFlowEvaluationModel, data) = nothing
function _check_remote_voltage_solver(::ACPolarPowerFlow{<:FastDecoupledACPowerFlow}, data)
    has_remote_voltage_groups(data.remote_voltage_control) || return
    throw(ArgumentError("this PowerFlowData holds remote voltage regulation groups (built with \
        remote_voltage_control = true), which the fast decoupled solver does not model; solve it \
        with Newton-Raphson, trust region or Levenberg-Marquardt, or build the data for the fast \
        decoupled solver (or with remote_voltage_control = false)."))
end

function _any_remote_generator(sys)
    for g in PSY.get_available_components(PSY.Generator, sys)
        isnothing(_generator_remote_bus(g)) || return true
    end
    return false
end

"""Build the remote-regulation groups of `data` from `sys` and set the bus types they solve
under (each regulated bus PV at its setpoint, each member plant's bus PQ), for every time step.
Called once, after `initialize_power_flow_data!`."""
function _enroll_remote_voltage_control!(data, pf, sys)
    if !_remote_voltage_enabled(pf)
        if get_remote_voltage_control(pf) && _any_remote_generator(sys)
            @warn "Remote voltage control (IREG) is not supported by the fast decoupled \
                solver: each generator holds its own terminal. Use Newton-Raphson, trust \
                region or Levenberg-Marquardt." maxlog = 1
        end
        return
    end
    nrd = PNM.get_network_reduction_data(data.power_network_matrix)
    bus_lookup = get_bus_lookup(data)
    rmap = PNM.get_reverse_bus_search_map(nrd)
    units = Dict{Int, Vector{Any}}()
    any_remote = false
    for g in PSY.get_available_components(PSY.Generator, sys)
        bix = _resolve_bus_ix(bus_lookup, rmap, PSY.get_number(PSY.get_bus(g)))
        isnothing(bix) && continue
        push!(get!(units, bix, Any[]), g)
        any_remote |= !isnothing(_generator_remote_bus(g))
    end
    any_remote || return
    n_buses, n_ts = size(data.bus_type)
    numbers = data.power_network_matrix.axes[1]

    # Each plant (the available units at one bus) regulates one bus: PSS/E keeps IREG per plant.
    target = Dict{Int, Int}()
    share = Dict{Int, Float64}()
    vs = Dict{Int, Union{Nothing, Float64}}()
    for bix in sort!(collect(keys(units)))
        gs = sort!(units[bix]; by = PSY.get_name)
        per = Dict{Int, Vector{Any}}()
        for g in gs
            r = _generator_remote_bus(g)
            rix = isnothing(r) ? bix : _resolve_bus_ix(bus_lookup, rmap, r)
            if isnothing(rix)
                @warn "Generator $(PSY.get_name(g)): its regulated bus $r (IREG) is not in \
                    the (reduced) network; it holds its own terminal." maxlog = PF_MAX_LOG
                rix = bix
            end
            push!(get!(per, rix, Any[]), g)
        end
        t = argmax(k -> (_rv_plant_share(per[k]), -k), collect(keys(per)))
        if length(per) > 1
            @warn "Generators at bus $(numbers[bix]) regulate different buses (IREG); the \
                plant regulates bus $(numbers[t]), the target of the largest RMPCT." maxlog =
                PF_MAX_LOG
        end
        data.bus_type[bix, 1] == PSY.ACBusTypes.PV || continue
        target[bix] = t
        share[bix] = _rv_plant_share(gs)
        vs[bix] = nothing
        for g in gs
            v = _generator_remote_setpoint(g)
            isnothing(v) || (vs[bix] = v; break)
        end
    end
    # A plant whose own bus other plants regulate holds that bus (no chains of remote control).
    regulated = Set(t for (b, t) in target if t != b)
    for b in sort!(collect(keys(target)))
        t = target[b]
        if t != b && b in regulated
            @warn "The plant at bus $(numbers[b]) regulates bus $(numbers[t]) (IREG), but \
                other plants regulate its own bus; it holds its own bus." maxlog = PF_MAX_LOG
            target[b] = b
        end
    end
    regulated = sort!(collect(Set(t for (b, t) in target if t != b)))
    isempty(regulated) && return

    island = zeros(Int, n_buses)
    for (k, buses) in enumerate(values(
        PNM.find_subnetworks(data.power_network_matrix.data, collect(1:n_buses))))
        for i in buses
            island[i] = k
        end
    end

    groups = RemoteVoltageGroup[]
    offset = 0
    for X in regulated
        btX = data.bus_type[X, 1]
        plants = sort!([b for (b, t) in target if t == X && b != X])
        if btX == PSY.ACBusTypes.REF
            @warn "Bus $(numbers[X]) is a reference bus; the plants that regulate it by IREG \
                ($(join(numbers[plants], ", "))) hold their own terminals." maxlog = PF_MAX_LOG
            continue
        end
        x_member = btX == PSY.ACBusTypes.PV && get(target, X, 0) == X
        if btX == PSY.ACBusTypes.PV && !x_member
            @warn "Bus $(numbers[X]) is held by a source that does not share by RMPCT; the \
                plants that regulate it by IREG ($(join(numbers[plants], ", "))) hold their \
                own terminals." maxlog = PF_MAX_LOG
            continue
        end
        away = filter(b -> island[b] != island[X], plants)
        if !isempty(away)
            @warn "Plants at buses $(join(numbers[away], ", ")) regulate bus $(numbers[X]) \
                (IREG) in another island; they hold their own terminals." maxlog = PF_MAX_LOG
            filter!(b -> island[b] == island[X], plants)
        end
        isempty(plants) && continue
        members = x_member ? vcat(X, plants) : plants
        # The setpoint: the plants' VS (X's own plant's included), else X's stored |V|.
        vset = data.bus_magnitude[X, 1]
        setpoints = [(share[b], vs[b]) for b in members if !isnothing(vs[b])]
        if !isempty(setpoints)
            vset = last(argmax(first, setpoints))
            if any(abs(v - vset) > 1e-4 for (_, v) in setpoints)
                @warn "Plants regulating bus $(numbers[X]) disagree on its scheduled \
                    voltage (VS); using $vset, the largest share's." maxlog = PF_MAX_LOG
            end
        end
        push!(groups, RemoteVoltageGroup(X, numbers[X], members, numbers[members],
            [share[b] for b in members], offset, vset))
        offset += length(members)
    end
    isempty(groups) && return

    rvc = data.remote_voltage_control
    rvc.groups = groups
    rvc.regulated_group = zeros(Int, n_buses)
    rvc.member_group = zeros(Int, n_buses)
    rvc.status = zeros(Int8, offset, n_ts)
    rvc.q_base = zeros(length(groups), n_ts)
    rvc.q_fixed = zeros(offset, n_ts)
    rvc.q_bounds = fill((0.0, 0.0), offset)
    rvc.q_held = zeros(offset, n_ts)
    for grp in groups, (k, b) in enumerate(grp.members)
        q_plant, lo, hi = _rv_plant_q_and_bounds(units[b], pf)
        rvc.q_bounds[grp.offset + k] = (lo, hi)
        rvc.q_fixed[grp.offset + k, :] .= view(data.bus_reactive_power_injections, b, :) .-
                                          q_plant
    end
    empty!(rvc.returns)
    for (g, grp) in enumerate(groups)
        X = grp.regulated_ix
        rvc.regulated_group[X] = g
        for b in grp.members
            rvc.member_group[b] = g
        end
        x_member = first(grp.members) == X
        for ts in 1:n_ts
            rvc.q_base[g, ts] = x_member ? 0.0 : data.bus_reactive_power_injections[X, ts]
            for b in grp.members
                b == X || (data.bus_type[b, ts] = PSY.ACBusTypes.PQ)
            end
            side = pf.check_reactive_power_limits ?
                   _rv_stored_saturation(data, rvc, g, grp, ts) : 0
            if iszero(side)
                data.bus_type[X, ts] = PSY.ACBusTypes.PV
                data.bus_magnitude[X, ts] = grp.vset
            else
                # Saturated in the stored state (VARLIM 0): the plants' total at their summed limit
                # on the side PSS/E leaves it, X at its stored |V| off VS. It starts fixed, X PQ,
                # and the return rule brings X back to PV once |V_X| crosses VS.
                # Each plant holds its stored Q: the group's total is at its summed limit, and the
                # stored split (which PSS/E may leave uneven) reproduces the stored point.
                for k in eachindex(grp.members)
                    rvc.status[grp.offset + k, ts] = side
                    rvc.q_held[grp.offset + k, ts] = _rv_member_q(data, rvc, g, grp, k, ts)
                end
                data.bus_type[X, ts] = PSY.ACBusTypes.PQ
                _rv_set_fixed_injections!(data, rvc, g, grp, ts)
            end
        end
    end
    n_sat = count(g -> !iszero(rvc.status[rvc.groups[g].offset + 1, 1]), eachindex(groups))
    n_sat > 0 && @info "Remote voltage control: $n_sat regulated bus(es) start saturated: \
        the plants' total at their summed reactive limit in the stored state, the bus off its \
        setpoint."
    slack = [numbers[b] for grp in groups for b in grp.members if b != grp.regulated_ix &&
             any(!iszero, view(data.bus_slack_participation_factors, b, :))]
    if !isempty(slack)
        @warn "Remote-regulating plants at buses $(join(slack, ", ")) take no distributed \
            slack: their buses are PQ in the solve." maxlog = 1
    end
    @info "Remote voltage control: $(length(groups)) regulated bus(es), \
        $(sum(grp -> count(!=(grp.regulated_ix), grp.members), groups)) remote plant(s)."
    return
end

"""Tolerance (p.u. Q, 1 MVAr on a 100 MVA base) within which a plant's stored reactive power
counts as at its limit for `_rv_stored_saturation`."""
const REMOTE_VOLTAGE_STORED_LIMIT_TOLERANCE = 1e-2

"""Whether group `grp` is saturated in the stored state at `ts`, as PSS/E's VARLIM leaves a group
that cannot hold its bus: |V_X| off the setpoint by more than `Q_LIMIT_RETURN_VOLTAGE_TOLERANCE`
and the plants' summed stored Q within `REMOTE_VOLTAGE_STORED_LIMIT_TOLERANCE` of their summed
limits on PSS/E's side: Σ Qmax with |V_X| below the setpoint (+1), Σ Qmin with it above (−1).
The test is on the group's total, not each plant: a fixed-Q plant (Qmin = Qmax) is at both
limits, and PSS/E can leave the total at the summed limit with the plants split otherwise.
0 (the group starts sharing at the setpoint) otherwise, or with an unbounded plant. Reads
`data` as initialised (X's stored |V|)."""
function _rv_stored_saturation(data, rvc::RemoteVoltageControl, g::Int,
    grp::RemoteVoltageGroup, ts::Int)
    V = data.bus_magnitude[grp.regulated_ix, ts]
    tol = Q_LIMIT_RETURN_VOLTAGE_TOLERANCE
    side = V < grp.vset - tol ? 1 : V > grp.vset + tol ? -1 : 0
    iszero(side) && return 0
    q, lim = 0.0, 0.0
    for k in eachindex(grp.members)
        q += _rv_member_q(data, rvc, g, grp, k, ts)
        lo, hi = rvc.q_bounds[grp.offset + k]
        lim += side > 0 ? hi : lo
    end
    isfinite(lim) || return 0
    return abs(q - lim) <= REMOTE_VOLTAGE_STORED_LIMIT_TOLERANCE ? side : 0
end

"""A plant's stored reactive power and its summed limits, over the generators at its bus that
inject reactive power (as `_get_injections!` and `_get_reactive_power_bound!` count them); an
unbounded generator makes the plant unbounded."""
function _rv_plant_q_and_bounds(gens, pf)
    q, lo, hi = 0.0, 0.0, 0.0
    for g in gens
        (contributes_reactive_power(g) &&
         reactive_power_contribution_type(g) == PowerContributionType.INJECTION) || continue
        q += PSY.get_reactive_power(g)
        lims = get_reactive_power_limits_for_power_flow(g)
        if isnothing(lims) || isinf(lims.min) || isinf(lims.max)
            lo, hi = -Inf, Inf
        else
            lo += lims.min
            hi += lims.max
        end
    end
    return q, lo, hi
end

"""Write each sharing group's injections from its Q_tot slot `x[2X−1]` into `data` (and into
`Q_net`, the residual's net injections, unless `nothing`). A group whose regulated bus is PQ
(saturated) has every member fixed, and is left as it is."""
function _apply_remote_voltage_injections!(
    Q_net::Union{Nothing, Vector{Float64}},
    x::Vector{Float64},
    data,
    ts::Int,
)
    rvc = data.remote_voltage_control
    @inbounds for (g, grp) in enumerate(rvc.groups)
        X = grp.regulated_ix
        data.bus_type[X, ts] == PSY.ACBusTypes.PV || continue
        Qt = x[2 * X - 1]
        S = _rv_active_share(rvc, grp, ts)
        qx = rvc.q_base[g, ts]
        for (k, b) in enumerate(grp.members)
            st = rvc.status[grp.offset + k, ts]
            if st == RV_RELEASED
                # An ordinary bus now, its row its own; X keeps only its non-plant part.
                b == X && (qx += rvc.q_fixed[grp.offset + k, ts])
                continue
            end
            q = if st == 0
                S > 0.0 ? grp.shares[k] / S * Qt : 0.0
            else
                _rv_held(rvc, grp.offset + k, ts)
            end
            q += rvc.q_fixed[grp.offset + k, ts]
            if b == X
                qx += q
            else
                isnothing(Q_net) ||
                    (Q_net[b] += q - data.bus_reactive_power_injections[b, ts])
                data.bus_reactive_power_injections[b, ts] = q
            end
        end
        data.bus_reactive_power_injections[X, ts] = qx
        isnothing(Q_net) ||
            (Q_net[X] = qx - get_bus_reactive_power_total_withdrawals(data, X, ts))
    end
    return
end

"""Fill each sharing group's Q_tot slot `x[2X−1]` from `data`: the sum of its sharing members'
injections (inverse of [`_apply_remote_voltage_injections!`](@ref))."""
function _remote_voltage_state!(x::Vector{Float64}, data, ts::Int)
    rvc = data.remote_voltage_control
    @inbounds for (g, grp) in enumerate(rvc.groups)
        X = grp.regulated_ix
        data.bus_type[X, ts] == PSY.ACBusTypes.PV || continue
        Qt = 0.0
        for (k, b) in enumerate(grp.members)
            rvc.status[grp.offset + k, ts] == 0 || continue
            Qt += _rv_member_q(data, rvc, g, grp, k, ts)
        end
        x[2 * X - 1] = Qt
    end
    return
end

# Structural entries for the Q_tot column at each member's Q row not already X's neighbour.
function _create_jacobian_matrix_structure_remote_voltage!(
    rows::Vector{J_INDEX_TYPE},
    columns::Vector{J_INDEX_TYPE},
    values::Vector{Float64},
    data,
)
    for grp in data.remote_voltage_control.groups
        X = grp.regulated_ix
        for b in grp.members
            (b == X || b in data.neighbors[X]) && continue
            push!(rows, J_INDEX_TYPE(2 * b))
            push!(columns, J_INDEX_TYPE(2 * X - 1))
            push!(values, 0.0)
        end
    end
    return
end

# Q_tot column values: −s_k/Σs at each sharing member's Q row, 0 at a fixed member's and at X's
# own Q row unless X's plant shares (overwriting the PV diagonal's −1). A saturated group's X is
# PQ, whose column the neighbour loop fills; the added entries are zeroed.
function _set_entries_for_remote_voltage!(
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    data,
    ts::Int,
)
    rvc = data.remote_voltage_control
    @inbounds for grp in rvc.groups
        X = grp.regulated_ix
        col = 2 * X - 1
        if data.bus_type[X, ts] == PSY.ACBusTypes.PV
            S = _rv_active_share(rvc, grp, ts)
            Jv[2 * X, col] = 0.0
            for (k, b) in enumerate(grp.members)
                share_on = rvc.status[grp.offset + k, ts] == 0 && S > 0.0
                Jv[2 * b, col] = share_on ? -grp.shares[k] / S : 0.0
            end
        else
            for b in grp.members
                (b == X || b in data.neighbors[X]) && continue
                Jv[2 * b, col] = 0.0
            end
        end
    end
    return
end

# A member's status changes the Q_tot column's values without changing any bus type, so the
# persisted NR factorization's pivots were chosen under other values. Marking the bus types it
# recorded as stale makes the next NR solve repivot (`_refresh_pivots!`), as after a PV↔PQ flip;
# the numeric factor itself is left for that solve to drop.
function _rv_drop_pivots!(data)
    e = data.polar_nr_cache[]
    (e isa PolarNRCache && !isempty(e.bus_types)) || return
    e.bus_types[1] = PSY.ACBusTypes.ISOLATED
    return
end

"""The reactive-limit check for the remote-regulation groups, after a converged solve; see the
file header. Returns `true` when a member or a group switched."""
function _remote_voltage_q_limits!(data, ts::Int)
    rvc = data.remote_voltage_control
    has_remote_voltage_groups(rvc) || return false
    switched = false
    names = data.power_network_matrix.axes[1]
    for (g, grp) in enumerate(rvc.groups)
        X = grp.regulated_ix
        if data.bus_type[X, ts] == PSY.ACBusTypes.PV
            fixed = false
            for (k, b) in enumerate(grp.members)
                row = grp.offset + k
                rvc.status[row, ts] == 0 || continue
                q = _rv_member_q(data, rvc, g, grp, k, ts)
                lo, hi = rvc.q_bounds[grp.offset + k]
                st = q > hi + BOUNDS_TOLERANCE ? 1 : q < lo - BOUNDS_TOLERANCE ? -1 : 0
                st == 0 && continue
                rvc.status[row, ts] = st
                rvc.q_held[row, ts] = st > 0 ? hi : lo
                fixed = true
                @debug "Remote voltage control: plant at bus $(names[b]) fixed at its \
                    $(st > 0 ? "maximum" : "minimum") Q for bus $(grp.regulated_number)"
            end
            if fixed
                switched = true
                _rv_set_fixed_injections!(data, rvc, g, grp, ts)
                if iszero(_rv_active_share(rvc, grp, ts))
                    data.bus_type[X, ts] = PSY.ACBusTypes.PQ
                    @info "Bus $(grp.regulated_number): every plant regulating it is at a \
                        reactive limit; it changed to PQ" maxlog = PF_MAX_LOG
                end
                continue
            end
            # Rejoins: a fixed member whose share would sit back inside its bound.
            S = _rv_active_share(rvc, grp, ts)
            S > 0.0 || continue
            Qt = 0.0
            for (k, b) in enumerate(grp.members)
                rvc.status[grp.offset + k, ts] == 0 &&
                    (Qt += _rv_member_q(data, rvc, g, grp, k, ts))
            end
            λ = Qt / S
            for (k, b) in enumerate(grp.members)
                row = grp.offset + k
                st = rvc.status[row, ts]
                (st == 0 || st == RV_RELEASED) && continue
                lo, hi = rvc.q_bounds[grp.offset + k]
                hi - lo > BOUNDS_TOLERANCE || continue     # a fixed-Q plant cannot share
                q = grp.shares[k] * λ
                back = (st > 0 && q < hi - BOUNDS_TOLERANCE) ||
                       (st < 0 && q > lo + BOUNDS_TOLERANCE)
                back && _rv_rejoin!(rvc, row, ts, names[b]) && (switched = true)
            end
        else
            any(k -> rvc.status[grp.offset + k, ts] == 0, eachindex(grp.members)) &&
                continue
            # Dissolved (every plant released): X stays PQ, nothing to return.
            all(k -> rvc.status[grp.offset + k, ts] == RV_RELEASED, eachindex(grp.members)) &&
                continue
            V = data.bus_magnitude[X, ts]
            tol = Q_LIMIT_RETURN_VOLTAGE_TOLERANCE
            side = V > grp.vset + tol ? 1 : V < grp.vset - tol ? -1 : 0
            side == 0 && continue
            back = false
            for (k, b) in enumerate(grp.members)
                row = grp.offset + k
                rvc.status[row, ts] == side || continue
                lo, hi = rvc.q_bounds[grp.offset + k]
                hi - lo > BOUNDS_TOLERANCE || continue
                _rv_rejoin!(rvc, row, ts, names[b]) && (back = true)
            end
            if back
                switched = true
                data.bus_type[X, ts] = PSY.ACBusTypes.PV
                data.bus_magnitude[X, ts] = grp.vset
                @info "Bus $(grp.regulated_number) changed back to PV: |V| = $V \
                    $(side > 0 ? "above" : "below") its setpoint $(grp.vset) with its \
                    plants at their $(side > 0 ? "maximum" : "minimum") Q" maxlog =
                    PF_MAX_LOG
            end
        end
    end
    switched && _rv_drop_pivots!(data)
    return switched
end

function _rv_rejoin!(rvc::RemoteVoltageControl, row::Int, ts::Int, name)
    key = (row, ts)
    n = get(rvc.returns, key, 0)
    if n >= MAX_Q_LIMIT_FLIP_FLOPS
        n == MAX_Q_LIMIT_FLIP_FLOPS &&
            @warn "Remote voltage control: the plant at bus $name rejoined its group $n \
                times at time step $ts (MAX_Q_LIMIT_FLIP_FLOPS); it stays at its limit for \
                the rest of this solve"
        rvc.returns[key] = n + 1
        return false
    end
    rvc.returns[key] = n + 1
    rvc.status[row, ts] = 0
    return true
end

# Fixed members' injections at their bounds; the regulated bus's own injection with them.
function _rv_set_fixed_injections!(data, rvc, g::Int, grp::RemoteVoltageGroup, ts::Int)
    X = grp.regulated_ix
    for (k, b) in enumerate(grp.members)
        st = rvc.status[grp.offset + k, ts]
        st == 0 && continue
        if st == RV_RELEASED
            # A released plant's row is its own; at X only the non-plant part stays.
            b == X && (data.bus_reactive_power_injections[X, ts] =
                rvc.q_base[g, ts] + rvc.q_fixed[grp.offset + k, ts])
            continue
        end
        q = _rv_held(rvc, grp.offset + k, ts) + rvc.q_fixed[grp.offset + k, ts]
        data.bus_reactive_power_injections[b, ts] = b == X ? rvc.q_base[g, ts] + q : q
    end
    return
end

function _reset_remote_voltage_returns!(data, ts::Int)
    filter!(kv -> kv[1][2] != ts, data.remote_voltage_control.returns)
    return
end

"""The bus type a bus has as the system describes it, for writing results back: a member plant
is PV while it shares (PQ once fixed at a limit, as a Q-limit release), and a regulated bus
without a plant of its own is PQ. Every other bus has its solve type."""
function get_reported_bus_type(data, ix::Int, ts::Int)
    bt = data.bus_type[ix, ts]
    rvc = data.remote_voltage_control
    _rv_role(rvc, ix) || return bt
    g = rvc.member_group[ix]
    if g != 0
        grp = rvc.groups[g]
        k = findfirst(==(ix), grp.members)
        rvc.status[grp.offset + k, ts] == RV_RELEASED && return bt
        sharing = rvc.status[grp.offset + k, ts] == 0 &&
                  data.bus_type[grp.regulated_ix, ts] == PSY.ACBusTypes.PV
        return sharing ? PSY.ACBusTypes.PV : PSY.ACBusTypes.PQ
    end
    return PSY.ACBusTypes.PQ
end

"""The remote-regulation groups of `data` ([`RemoteVoltageGroup`](@ref)); empty when none."""
get_remote_voltage_groups(data) = data.remote_voltage_control.groups

"""Each group's member status at `time_step`, by regulated bus number: a vector of
`(bus number, share, status)` with status 0 sharing, +1 at Qmax, −1 at Qmin."""
function get_remote_voltage_status(data, time_step::Int = 1)
    rvc = data.remote_voltage_control
    return Dict(
        grp.regulated_number => [
            (grp.member_numbers[k], grp.shares[k], Int(rvc.status[grp.offset + k, time_step]))
            for k in eachindex(grp.members)
        ] for grp in rvc.groups)
end

# ── Editing groups in place (contingency analysis) ───────────────────────────────────────────
# A contingency engine that applies outages to a built `PowerFlowData` in place (instead of
# rebuilding it from the edited System) uses these to keep the groups right. None of them adds or
# removes a Jacobian structural entry, so the cached structure stays valid; each one marks the NR
# factorization's pivots stale. The role vectors (`member_group`, `regulated_group`) are shared by
# every time step: in multi-period data, apply an edit at every time step.

"""Whether bus row `ix` is a regulated bus or a member plant's bus of a remote-regulation group."""
is_remote_voltage_bus(data, ix::Int) = _rv_role(data.remote_voltage_control, ix)

"""
    remote_voltage_release!(data, ix, ts = 1; hold_terminal = false, at = :present) -> Symbol

Take the plant at bus row `ix` out of its group (status `RV_RELEASED`, `member_group[ix] = 0`): it
has no share and the group writes no injection for it, so its bus is an ordinary bus again, whose
injection, bounds and type are the caller's. With `hold_terminal` (its bus live, a source left on
it) a remote plant's bus becomes PV and holds its own terminal, at its present |V| (`at = :present`,
what a fresh build gives a plant whose regulated bus is gone) or at the group's setpoint
(`at = :vs`, PSS/E's rule for a plant whose IREG bus is not a type 1 or 2 bus, as recalled from the
POM's generator data notes). The plant at X itself leaves X regulated by the others; X keeps only
its non-plant injection.

Then the group: `:sharing` (a plant still shares; X stays PV), `:saturated` (every plant left is at
a limit: X PQ at its present |V|, and the return rule may bring it back), `:dissolved` (no plant
left: X PQ at its present |V| for good, `regulated_group[X] = 0`), or `:none` (`ix` is in no
group).
"""
function remote_voltage_release!(data, ix::Int, ts::Int = 1; hold_terminal::Bool = false,
    at::Symbol = :present)
    at in (:present, :vs) || throw(ArgumentError("at must be :present or :vs; got :$at"))
    rvc = data.remote_voltage_control
    (isempty(rvc.member_group) || rvc.member_group[ix] == 0) && return :none
    g = rvc.member_group[ix]
    grp = rvc.groups[g]
    k = findfirst(==(ix), grp.members)
    X = grp.regulated_ix
    was_pv = data.bus_type[X, ts] == PSY.ACBusTypes.PV
    rvc.status[grp.offset + k, ts] = RV_RELEASED
    rvc.member_group[ix] = 0
    if ix != X && hold_terminal
        data.bus_type[ix, ts] = PSY.ACBusTypes.PV
        at === :vs && (data.bus_magnitude[ix, ts] = grp.vset)
    end
    _rv_set_fixed_injections!(data, rvc, g, grp, ts)
    _rv_drop_pivots!(data)
    released(k) = rvc.status[grp.offset + k, ts] == RV_RELEASED
    if all(released, eachindex(grp.members))
        data.bus_type[X, ts] = PSY.ACBusTypes.PQ
        rvc.regulated_group[X] = 0
        return :dissolved
    end
    if was_pv && iszero(_rv_active_share(rvc, grp, ts))
        data.bus_type[X, ts] = PSY.ACBusTypes.PQ
    end
    return data.bus_type[X, ts] == PSY.ACBusTypes.PV ? :sharing : :saturated
end

# The available generators at bus row `ix` (the enrolment's plant), re-read from `sys`.
function _rv_available_units(data, sys, ix::Int)
    nrd = PNM.get_network_reduction_data(data.power_network_matrix)
    bus_lookup = get_bus_lookup(data)
    rmap = PNM.get_reverse_bus_search_map(nrd)
    return [g for g in PSY.get_available_components(PSY.Generator, sys) if
            _resolve_bus_ix(bus_lookup, rmap, PSY.get_number(PSY.get_bus(g))) == ix]
end

"""
    remote_voltage_refresh!(data, sys, ix, ts = 1) -> Symbol

Re-read the plant at bus row `ix` from `sys` after its units changed (in system base units, as
`PowerFlowData` is built): its share (the RMPCT sum of its available generators), its limits and its
non-plant part (`q_fixed`: the row's present injection, which the caller has recomputed from the
System, less the plant's stored Q). A plant with no available generator left is released
(`remote_voltage_release!`, which this returns). At a regulated bus without a plant of its own,
refreshes `q_base` from the row's injection. Returns `:sharing`, `:refreshed` (X's `q_base`), or
`:none`.
"""
function remote_voltage_refresh!(data, sys, ix::Int, ts::Int = 1)
    rvc = data.remote_voltage_control
    isempty(rvc.regulated_group) && return :none
    out = :none
    gx = rvc.regulated_group[ix]
    if gx != 0 && rvc.member_group[ix] != gx
        rvc.q_base[gx, ts] = data.bus_reactive_power_injections[ix, ts]
        out = :refreshed
    end
    g = rvc.member_group[ix]
    g == 0 && return out
    gens = _rv_available_units(data, sys, ix)
    isempty(gens) && return remote_voltage_release!(data, ix, ts)
    grp = rvc.groups[g]
    k = findfirst(==(ix), grp.members)
    row = grp.offset + k
    grp.shares[k] = _rv_plant_share(gens)
    q_plant, lo, hi = _rv_plant_q_and_bounds(gens, nothing)
    rvc.q_bounds[row] = (lo, hi)
    ix == grp.regulated_ix && (rvc.q_base[g, ts] = 0.0)
    rvc.q_fixed[row, ts] = data.bus_reactive_power_injections[ix, ts] - q_plant
    _rv_drop_pivots!(data)
    return :sharing
end

"""
    remote_voltage_islands!(data, island, dead, ts = 1; swing = falses(length(dead)))

After a topology change, with `island[i]` the component of bus row `i`, `dead[i]` a row out of the
solve and `swing[i]` a row made a reference bus: a group whose X is dead or a reference bus is
dissolved, each live plant released holding its own terminal at its present |V| (the fresh build's
rules for a regulated bus out of the network or a reference bus); a dead plant is released; a plant
that became a reference bus is released (the row is the caller's REF); a live plant in another
island than X is released holding its own terminal (enrolment's "another island" rule). Returns
the number of plants released.
"""
function remote_voltage_islands!(data, island::AbstractVector{<:Integer},
    dead::AbstractVector{Bool}, ts::Int = 1; swing::AbstractVector{Bool} = falses(length(dead)))
    rvc = data.remote_voltage_control
    n = 0
    for grp in rvc.groups
        X = grp.regulated_ix
        rvc.regulated_group[X] == 0 && continue          # already dissolved
        x_gone = dead[X] || swing[X]
        for (k, b) in enumerate(grp.members)
            rvc.status[grp.offset + k, ts] == RV_RELEASED && continue
            live = !dead[b] && !swing[b]
            if x_gone || !live || island[b] != island[X]
                remote_voltage_release!(data, b, ts; hold_terminal = live && b != X)
                n += 1
            end
        end
    end
    return n
end

"""Everything an in-place edit may change in `data`'s remote-regulation groups, for an exact
restore with [`restore_remote_voltage!`](@ref): member status, rejoin counts, shares, plant
limits, fixed parts, held Q and the role vectors."""
function remote_voltage_snapshot(data)
    rvc = data.remote_voltage_control
    return (status = copy(rvc.status), returns = copy(rvc.returns),
        shares = [copy(grp.shares) for grp in rvc.groups], q_bounds = copy(rvc.q_bounds),
        q_fixed = copy(rvc.q_fixed), q_base = copy(rvc.q_base), q_held = copy(rvc.q_held),
        regulated_group = copy(rvc.regulated_group), member_group = copy(rvc.member_group))
end

"""Put back a [`remote_voltage_snapshot`](@ref) of the same `data` (bus types, magnitudes and
injections are the caller's to restore)."""
function restore_remote_voltage!(data, s)
    rvc = data.remote_voltage_control
    copyto!(rvc.status, s.status)
    empty!(rvc.returns)
    merge!(rvc.returns, s.returns)
    for (grp, sh) in zip(rvc.groups, s.shares)
        copyto!(grp.shares, sh)
    end
    copyto!(rvc.q_bounds, s.q_bounds)
    copyto!(rvc.q_fixed, s.q_fixed)
    copyto!(rvc.q_base, s.q_base)
    copyto!(rvc.q_held, s.q_held)
    copyto!(rvc.regulated_group, s.regulated_group)
    copyto!(rvc.member_group, s.member_group)
    _rv_drop_pivots!(data)
    return
end
