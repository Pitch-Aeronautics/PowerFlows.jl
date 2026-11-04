"""Shunt susceptance invariants; `false` (with a `@warn`) de-enrolls the device, leaving
it locked at its current setting (the safe posture for bad control data)."""
function _validate_shunt(
    name::String,
    b_min::Float64,
    b0::Float64,
    b_max::Float64,
    steps::Vector{Int},
    dB::Vector{Float64},
)::Bool
    if !(b_min <= b0 <= b_max)
        @warn "ControlledSwitchedShunt \"$name\": b0=$b0 is outside \
            [b_min=$b_min, b_max=$b_max]; leaving the shunt locked at its current setting."
        return false
    end
    if b_min == b_max
        @warn "ControlledSwitchedShunt \"$name\": no controllable susceptance range \
            (b_min == b_max == $b_min); leaving the shunt locked at its current setting."
        return false
    end
    for k in eachindex(steps, dB)
        if iszero(steps[k]) && !iszero(dB[k])
            @warn "ControlledSwitchedShunt \"$name\": block $k has zero steps but \
                nonzero dB=$(dB[k]) — malformed metadata; leaving the shunt locked."
            return false
        end
    end
    return true
end

"""Tap invariants; `false` (with a `@warn`) de-enrolls the device, leaving the tap locked
at its current ratio (the safe posture for bad control data)."""
function _validate_tap(
    name::String,
    p_min::Float64,
    p_max::Float64,
    ntp::Int,
)::Bool
    if p_min > p_max
        @warn "ControlledTap \"$name\": p_min=$p_min exceeds p_max=$p_max — malformed \
            tap-ratio limits; leaving the tap locked at its current ratio."
        return false
    end
    if p_min == p_max
        @warn "ControlledTap \"$name\": no controllable tap-ratio range \
            (p_min == p_max == $p_min); leaving the tap locked at its current ratio."
        return false
    end
    if ntp < 2
        # Missing/degenerate position data means a locked changer; fabricating a default
        # grid here would silently turn a locked device into an active one.
        @warn "ControlledTap \"$name\": fewer than 2 tap positions (ntp=$ntp); \
            leaving the tap locked at its current ratio."
        return false
    end
    return true
end

"""Voltage-setpoint plausibility gate shared by all voltage-controlling devices."""
function _validate_vset(kind::String, name::String, vset::Float64)::Bool
    if !(CONTROL_VSET_MIN <= vset <= CONTROL_VSET_MAX)
        @warn "$kind \"$name\": voltage setpoint $vset p.u. is outside \
            [$CONTROL_VSET_MIN, $CONTROL_VSET_MAX] — implausible control data (for parsed \
            systems PSY's admittance_limits holds the PSS/E VSWLO/VSWHI voltage band; \
            other sources may not). Leaving the device locked at its current setting."
        return false
    end
    return true
end

# Resolve a raw PSY bus number to a network index through the reduction's parent map
# (merged bus → surviving parent); `nothing` when the bus is not in the reduced network.
_resolve_bus_ix(
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
    n::Int,
) = get(bus_lookup, get(reverse_bus_search_map, n, n), nothing)

"""Tap-control metadata for one `TapTransformer`, read from its first-class PSY fields.
`get_tap_limits` is already in tap-ratio units (the PSS/E parser scales RMI1/RMA1 by WINDV2);
`get_regulated_bus_number` is 0 for local (to-bus) control."""
function _tap_metadata(tx, to_bus::Int)
    lims = PSY.get_tap_limits(tx)
    reg = PSY.get_regulated_bus_number(tx)
    cbus = to_bus
    if !iszero(reg)
        cbus = reg
    end
    return (
        cbus = cbus,
        pmin = lims.min,
        pmax = lims.max,
        ntp = PSY.get_number_of_tap_positions(tx),
        vset = PSY.get_voltage_setpoint(tx),
    )
end

# `PSY.ImpedanceCorrectionData` on each `TapTransformer`, keyed by component UUID. One pass
# over the attribute pairs, as PNM's `build_impedance_correction_factors!` does: a per-tap
# supplemental-attribute query hits the association store and is too slow for large cases.
function _tap_correction_tables(sys)
    tables = Dict{Base.UUID, PSY.ImpedanceCorrectionData}()
    isempty(PSY.get_supplemental_attributes(PSY.ImpedanceCorrectionData, sys)) &&
        return tables
    for pair in PSY.get_component_supplemental_attribute_pairs(
        PSY.TapTransformer,
        PSY.ImpedanceCorrectionData,
        sys,
    )
        tables[IS.get_uuid(pair.component)] = pair.supplemental_attribute
    end
    return tables
end

"""`(yt, correction)` for a tap device: the series admittance `1/(r+jx)` and, when an
impedance correction table is attached, either the table itself (`TAP_RATIO` mode, re-evaluated
at each tap the controller applies) or its factor folded into `yt` (`PHASE_SHIFT_ANGLE` mode,
fixed for a `TapTransformer`). Both are evaluated exactly as PNM's `Ybus` evaluates them, so the
device's terms at the enrollment tap equal the Y-bus block it delta-updates."""
function _tap_series_admittance(tx, correction_tables)
    yt = 1.0 / (PSY.get_r(tx) + PSY.get_x(tx) * im)
    ict = get(correction_tables, IS.get_uuid(tx), nothing)
    isnothing(ict) && return yt, nothing
    if PSY.get_transformer_control_mode(ict) ==
       PSY.ImpedanceCorrectionTransformerControlMode.TAP_RATIO
        return yt, PSY.get_impedance_correction_curve(ict)
    end
    return yt / PNM._evaluate_correction_table(tx, ict), nothing
end

# Susceptance model of a switched shunt. The PSS/E parser (MODSW 0/1/2) stores
# `Y = BINIT` (the TOTAL in-service admittance) and ZEROES `initial_status` to avoid
# double counting, so the reachable set is spanned by the blocks alone (base 0) with the
# current point at BINIT. API-built components follow the PSY docstring instead: `Y` is
# the fixed N=0 base and `initial_status` is meaningful. The presence of the parser's
# MODSW key distinguishes the two conventions.
# BINIT convention ⇔ the PSS/E parser produced this shunt: it stores the TOTAL in-service
# admittance in `Y` and zeroes a full-length `initial_status` (see pm_io/psse.jl). API-built
# shunts carry the switched part in a nonzero (or empty) `initial_status`. This is the write-back
# convention too (see `ControlledSwitchedShunt.psse_convention`).
function _is_binit_shunt(init_status::Vector{Int})
    return !isempty(init_status) && all(iszero, init_status)
end

function _shunt_susceptance_model(
    name::String,
    Y0::Complex{Float64},
    steps::Vector{Int},
    dB::Vector{Float64},
    init_status::Vector{Int},
)
    if _is_binit_shunt(init_status)   # PSS/E parser (BINIT) convention
        b_fixed = 0.0
        current = imag(Y0)
    else                              # PSY API convention
        b_fixed = imag(Y0)
        current = imag(Y0) + sum(init_status .* dB; init = 0.0)
    end
    b_min = b_fixed + sum(min.(steps .* dB, 0.0); init = 0.0)
    b_max = b_fixed + sum(max.(steps .* dB, 0.0); init = 0.0)
    if !(b_min - BOUNDS_TOLERANCE <= current <= b_max + BOUNDS_TOLERANCE)
        @warn "ControlledSwitchedShunt \"$name\": initial susceptance $current p.u. lies \
            outside the block-reachable range [$b_min, $b_max]; clamping the control \
            baseline into the range."
        current = clamp(current, b_min, b_max)
    end
    return b_fixed, current, b_min, b_max
end

"""Initial susceptance of a FACTS device: its stored delivered reactive power
`reactive_power_required` (MVA) over `base_mva·|V|²` at its own bus's stored magnitude,
clamped to `[-b0, b0]`; 0 without a stored Q."""
function _facts_warm_start(fd, base_mva::Float64, b0::Float64)
    q = PSY.get_reactive_power_required(fd)
    iszero(q) && return 0.0
    v = max(PSY.get_magnitude(PSY.get_bus(fd)), 1e-3)
    return clamp(q / (base_mva * v^2), -b0, b0)
end

"""Debit each warm-started FACTS device's initial susceptance into the constant-impedance
reactive withdrawal of its bus, every time step, so the device store and the withdrawal agree
(the inverse of `write_device_settings!`; `apply_parameter!` moves it from there)."""
function _debit_facts_warm_start!(data, set)
    isnothing(set) && return
    for d in set.facts
        iszero(d.current) && continue
        data.bus_reactive_power_constant_impedance_withdrawals[d.bus_ix, :] .-= d.current
    end
    return
end

const TAP_DIRECTIONS = (:sensitivity, :cont_sign)

function _validate_tap_direction(tap_direction::Symbol)
    tap_direction in TAP_DIRECTIONS || throw(
        ArgumentError(
            "tap_direction must be one of $(TAP_DIRECTIONS); got :$tap_direction",
        ),
    )
    return
end

"""The signed PSS/E CONT1 of a `TapTransformer`: `ext["CONT1"]` (the PSS/E parser keeps it,
sign included, while `regulated_bus_number` holds its magnitude) or `ext["psse"]["CONT1"]`;
`nothing` when absent or zero."""
function _tap_cont1(tx)
    ext = PSY.get_ext(tx)
    c = get(ext, "CONT1", nothing)
    if !(c isa Real)
        psse = get(ext, "psse", nothing)
        c = psse isa AbstractDict ? get(psse, "CONT1", nothing) : nothing
    end
    (c isa Real && !iszero(c)) || return nothing
    return c
end

"""Orientation of a tap under PSS/E's CONT convention. The tap ratio is the winding-1 (`from`)
ratio, so for a bus on the winding-2 side raising it lowers the voltage (−1), and for one on the
winding-1 side it raises it (+1). PSS/E places the controlled bus by position first: a CONT1
equal to a terminal bus is on that terminal's own winding (the `from` bus +1, the `to` bus −1)
whatever its sign. Only a remote bus, neither terminal, is placed by the sign (raw-data notes on
CONT1): positive as if on the winding-2 side (−1), negative as if on the winding-1 side (+1).
So a system as PowerSystems parses it (the raw CONT1, sign included, in `ext`) orients as PSS/E
does. 0 (orient by measured sensitivity) without a CONT1."""
function _tap_cont_orientation(tx)
    c = _tap_cont1(tx)
    isnothing(c) && return 0
    arc = PSY.get_arc(tx)
    bus = abs(c)
    bus == PSY.get_number(PSY.get_from(arc)) && return 1
    bus == PSY.get_number(PSY.get_to(arc)) && return -1
    return c > 0 ? -1 : 1
end

"""PSS/E `ADJM` of a switched shunt (`ext`, from the PSS/E parser): 1 switches to the next
higher or lower total admittance, 0 (the default, and for API-built shunts) in input order."""
function _shunt_adjm(sa)
    adjm = get(PSY.get_ext(sa), "ADJM", 0)
    adjm isa Real || return 0
    return Int(adjm) == 1 ? 1 : 0
end

"""The PSS/E controlled-voltage band `[VMI1, VMA1]` of a parsed `TapTransformer`, or `nothing`.

PowerSystems 5 keeps the band only in `ext` (the PSS/E parser's `VMI1`/`VMA1`) and stores its
midpoint as `voltage_setpoint`. The band is used only while the two agree: an API-built
transformer has no band, and a setpoint moved off the midpoint after parsing is a point
setpoint the caller chose, which a stale `ext` band must not override."""
function _tap_voltage_band(tx, vset::Float64)
    ext = PSY.get_ext(tx)
    vmi = get(ext, "VMI1", nothing)
    vma = get(ext, "VMA1", nothing)
    (vmi isa Real && vma isa Real) || return nothing
    vmi, vma = Float64(vmi), Float64(vma)
    vmi < vma || return nothing
    isapprox(vset, (vmi + vma) / 2; atol = 1e-9) || return nothing
    return (vmi, vma)
end

"""Build the type-stable device set from a `PSY.System`.

`bus_lookup` maps PSY bus number → network index in the (possibly reduced) network;
`reverse_bus_search_map` maps reduction-merged bus numbers to their surviving parent;
`ybus` is the assembled `AC_Ybus_Matrix` from `data.power_network_matrix`. `n_time_steps`
sizes the returned set's per-ts shunt/FACTS state store (see `ControlledDeviceSet`).

Per-device data problems (unresolvable buses, degenerate ranges, unsupported control
modes) de-enroll the device with a `@warn` — the device stays at its current setting
(a warn-and-lock posture) — and never abort construction.

`tap_voltage_band = true` (the default) gives a PSS/E-parsed voltage-controlling tap its
`[VMI1, VMA1]` band: held while the regulated voltage is inside it, stepped toward it only when
outside, stopping at the first discrete step inside (see `_tap_voltage_band`). `false` drives
every tap to its `voltage_setpoint` (the band midpoint for parsed data), the behaviour before
the band was implemented. Taps without a band always regulate to `voltage_setpoint`.

`shunt_voltage_band = true` (the default) does the same for a voltage-controlling switched
shunt and its `[VSWLO, VSWHI]` band (`admittance_limits`): held inside, stepped block by block
toward the band when outside, stopping at the first step inside (a continuous MODSW 2 shunt
ends on the violated edge), with the blocks in PSS/E's order (`ext["ADJM"]`, see
[`_shunt_levels`](@ref)). `false` is the rule before: regulate toward the band midpoint, hold
once inside, snap to the nearest step of the input-order chains.

`tap_direction = :sensitivity` (the default) orients every tap by its measured dV/dp, with
parallel taps regulating one bus aligned when they measure opposite signs (see
`_align_parallel_taps!`). `:cont_sign` orients a tap with a CONT1 in `ext` as PSS/E does instead
(a terminal bus on its own winding's side, a remote bus by CONT1's sign; see
`_tap_cont_orientation`); taps without one stay on the measured sign."""
function build_controlled_device_set(
    sys,
    bus_lookup::Dict{Int, Int},
    ybus;
    reverse_bus_search_map::Dict{Int, Int} = Dict{Int, Int}(),
    n_time_steps::Int = 1,
    tap_voltage_band::Bool = true,
    shunt_voltage_band::Bool = true,
    tap_direction::Symbol = :sensitivity,
)
    _validate_tap_direction(tap_direction)
    taps = ControlledTap[]
    correction_tables = _tap_correction_tables(sys)
    for tx in PSY.get_available_components(PSY.TapTransformer, sys)
        PSY.get_control_objective(tx) == PSY.TransformerControlObjective.VOLTAGE ||
            continue
        name = PSY.get_name(tx)
        arc = PSY.get_arc(tx)
        fb = PSY.get_number(PSY.get_from(arc))
        tb = PSY.get_number(PSY.get_to(arc))
        md = _tap_metadata(tx, tb)
        fix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, fb)
        tix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, tb)
        cix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, md.cbus)
        if isnothing(fix) || isnothing(tix)
            missing_bus = tb
            if isnothing(fix)
                missing_bus = fb
            end
            @warn "ControlledTap \"$name\": terminal bus $missing_bus \
                is not in the (reduced) network; leaving the tap locked."
            continue
        end
        if isnothing(cix)
            @warn "ControlledTap \"$name\": controlled bus $(md.cbus) is not in the \
                (reduced) network; leaving the tap locked."
            continue
        end
        if fix == tix
            @warn "ControlledTap \"$name\": arc collapsed by a network reduction \
                (from == to after bus merging); leaving the tap locked."
            continue
        end
        _validate_tap(name, md.pmin, md.pmax, md.ntp) || continue
        _validate_vset("ControlledTap", name, md.vset) || continue
        yt, correction = _tap_series_admittance(tx, correction_tables)
        band = tap_voltage_band ? _tap_voltage_band(tx, md.vset) : nothing
        v_lo, v_hi = isnothing(band) ? (md.vset, md.vset) : band
        tap0 = PSY.get_tap(tx)
        if !(md.pmin - BOUNDS_TOLERANCE <= tap0 <= md.pmax + BOUNDS_TOLERANCE)
            @warn "ControlledTap \"$name\": initial tap ratio $tap0 lies \
                outside the tap-ratio band [$(md.pmin), $(md.pmax)]; leaving the tap \
                locked at its current ratio."
            continue
        end
        push!(
            taps,
            ControlledTap(
                name,
                fix,
                tix,
                cix,
                md.vset,
                yt,
                PSY.get_α(tx),   # −(π/6)·winding_group_number: PNM stamps t = p·e^{iα}
                md.pmin,
                md.pmax,
                collect(range(md.pmin, md.pmax; length = md.ntp)),
                _ybus_block_offsets(ybus, fix, tix),
                tap0,   # initial (reporting)
                tap0,   # synced (arc-admittance rows reflect this tap)
                tap0,   # current
                correction,
                !isnothing(band),
                v_lo,
                v_hi,
                0,      # side (continuation scratch)
                tap0,   # anchor (continuation scratch)
                tap_direction == :cont_sign ? _tap_cont_orientation(tx) : 0,
                0,      # aligned (continuation scratch)
                0.0,    # offset (continuation scratch)
            ),
        )
    end

    shunts = ControlledSwitchedShunt[]
    for sa in PSY.get_available_components(PSY.SwitchedAdmittance, sys)
        name = PSY.get_name(sa)
        bus = PSY.get_number(PSY.get_bus(sa))
        mode = PSY.get_control_mode(sa)
        if mode == PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE
            continuous = false
        elseif mode == PSY.SwitchedAdmittanceControlMode.CONTINUOUS_VOLTAGE
            continuous = true
        elseif mode == PSY.SwitchedAdmittanceControlMode.FIXED
            @debug "ControlledSwitchedShunt $name: control_mode FIXED (locked); \
                treated as fixed admittance, not enrolled."
            continue
        else
            @warn "ControlledSwitchedShunt \"$name\": control_mode $mode (remote \
                reactive-power / remote-device control) is not supported — only \
                DISCRETE_VOLTAGE and CONTINUOUS_VOLTAGE are implemented. Leaving the \
                shunt locked at its current setting."
            continue
        end
        bix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, bus)
        if isnothing(bix)
            @warn "ControlledSwitchedShunt \"$name\": bus $bus is not in the (reduced) \
                network; leaving the shunt locked."
            continue
        end
        # `regulated_bus_number` is 0 for local control (PSS/E SWREM/NREG map to it in the parser).
        reg = PSY.get_regulated_bus_number(sa)
        cbus = bus
        if !iszero(reg)
            cbus = reg
        end
        cix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, cbus)
        if isnothing(cix)
            @warn "ControlledSwitchedShunt \"$name\": controlled bus $cbus is not in \
                the (reduced) network; leaving the shunt locked."
            continue
        end
        lims = PSY.get_admittance_limits(sa)
        vset = (lims.min + lims.max) / 2.0
        _validate_vset("ControlledSwitchedShunt", name, vset) || continue
        Y0 = PSY.get_Y(sa)
        steps = PSY.get_number_of_steps(sa)
        dB = imag.(PSY.get_Y_increase(sa))
        init_status = PSY.get_initial_status(sa)
        b_fixed, current_b, bmin, bmax = _shunt_susceptance_model(
            name, Y0, steps, dB, init_status)
        _validate_shunt(name, bmin, b_fixed, bmax, steps, dB) || continue
        adjm = shunt_voltage_band ? _shunt_adjm(sa) : 0
        levels, level_blocks = _shunt_levels(b_fixed, steps, dB, adjm; name = name)
        band = shunt_voltage_band && lims.min < lims.max
        push!(
            shunts,
            ControlledSwitchedShunt(
                name,
                bix,
                cix,
                vset,
                lims.min,   # VSWLO: deadband lower edge
                lims.max,   # VSWHI: deadband upper edge
                real(Y0),
                b_fixed,
                steps,
                dB,
                bmin,
                bmax,
                zeros(Int, length(dB)),
                continuous,
                current_b,   # initial (reporting)
                current_b,   # current
                _is_binit_shunt(init_status),   # psse_convention: true ⇒ parser/BINIT
                levels,
                level_blocks,
                band,
                0,           # side (continuation scratch)
                current_b,   # anchor (continuation scratch)
            ),
        )
    end

    facts = ControlledFACTS[]
    _enroll_facts!(facts, sys, bus_lookup, reverse_bus_search_map)
    _share_regulated_setpoints!(shunts, taps, facts, sys, bus_lookup,
        reverse_bus_search_map; band = shunt_voltage_band)

    return ControlledDeviceSet(taps, shunts, facts, n_time_steps)
end

"""The bus a generator regulates remotely, by PSS/E's IREG, or `nothing`. PowerSystems has no
field for it (a generator holds its own terminal); the PSS/E parser keeps IREG in `ext` (v33),
and `ext["psse"]["IREG"]` (or `"ireg"`) is read too. 0 or the unit's own bus is local."""
function _generator_remote_bus(g)
    ext = PSY.get_ext(g)
    r = get(ext, "IREG", nothing)
    if !(r isa Real)
        psse = get(ext, "psse", nothing)
        r = psse isa AbstractDict ? get(psse, "IREG", get(psse, "ireg", nothing)) : nothing
    end
    (r isa Real && !iszero(r)) || return nothing
    r = Int(r)
    r == PSY.get_number(PSY.get_bus(g)) && return nothing
    return r
end

"""Shared regulated buses (GAP-056). A continuous switched shunt with a point setpoint
(VSWLO = VSWHI) whose regulated bus is also regulated by another enrolled device (a tap, shunt
or FACTS) or by a generator by IREG (`_generator_remote_bus`) is given a hold band of
±`CONTROL_SHARED_SETPOINT_BAND` around its setpoint: it keeps its stored susceptance while the
voltage is inside, and outside it regulates to the nearer edge and ends there (the band rule of
a banded continuous shunt).

Why: with a point setpoint the sigmoid law drives every such shunt to the susceptance its
voltage error maps to, whatever it stored, and the exact finish then puts the bus on its
setpoint. A bus held by several devices is over-determined that way: two shunts on one bus
were pulled to a common position (5 MVAr each with no voltage error at all), and a shunt
sharing its bus with a remote-regulating generator, which PowerFlows models as holding its own
terminal, took the generator's part, a large ΔB for a ~1e-4 p.u. error. PSS/E adjusts a
switched shunt only while its voltage is off setpoint; with another device holding the bus
there, it stays. Under the band the stored split is kept, and a device still acts when the
others cannot hold the bus."""
function _share_regulated_setpoints!(shunts, taps, facts, sys, bus_lookup,
    reverse_bus_search_map; band::Bool = true)
    any(d -> d.continuous && d.vset_lo == d.vset_hi, shunts) || return
    counts = Dict{Int, Int}()
    _count_controlled_buses!(counts, taps)
    _count_controlled_buses!(counts, shunts)
    _count_controlled_buses!(counts, facts)
    for g in PSY.get_available_components(PSY.Generator, sys)
        r = _generator_remote_bus(g)
        isnothing(r) && continue
        rix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, r)
        isnothing(rix) || (counts[rix] = get(counts, rix, 0) + 1)
    end
    δ = CONTROL_SHARED_SETPOINT_BAND
    for d in shunts
        (d.continuous && d.vset_lo == d.vset_hi) || continue
        get(counts, d.controlled_ix, 0) > 1 || continue
        d.vset_lo = d.vset - δ
        d.vset_hi = d.vset + δ
        d.band = band
        @debug "ControlledSwitchedShunt $(d.name): regulated bus shared with another \
            device; held within ±$δ p.u. of its setpoint."
    end
    return
end

# Continuous shunt FACTS (SVC/STATCOM) voltage control. `rating` (SHMX) bounds the SVC
# susceptance-at-unity or the STATCOM current; `q_cap` is an independent MVA ceiling. Both
# combine into the |V|-dependent limit `_facts_b_limit`. FCREG (`regulated_bus_number`)
# selects local vs. remote-bus regulation.
function _enroll_facts!(
    facts::Vector{ControlledFACTS},
    sys,
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
)
    base_mva = PSY.get_base_power(sys)
    for fd in PSY.get_available_components(PSY.FACTSControlDevice, sys)
        name = PSY.get_name(fd)
        mode = PSY.get_control_mode(fd)
        # OOS or no control mode ⇒ not a voltage-controlling shunt; not enrolled.
        if isnothing(mode) || mode == PSY.FACTSOperationModes.OOS
            @debug "ControlledFACTS $name: control_mode=$(mode) is not \
                voltage-controlling; not enrolled."
            continue
        end
        bus = PSY.get_number(PSY.get_bus(fd))
        bix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, bus)
        if isnothing(bix)
            @warn "ControlledFACTS \"$name\": bus $bus is not in the (reduced) network; \
                device not enrolled."
            continue
        end
        reg = PSY.get_regulated_bus_number(fd)
        cix = bix
        if !iszero(reg)
            cix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, reg)
        end
        if isnothing(cix)
            @warn "ControlledFACTS \"$name\": regulated bus $reg is not in the \
                (reduced) network; device not enrolled."
            continue
        end
        # `rating` (SHMX) is MVA at unity voltage ⇒ the SVC susceptance-at-unity bound or
        # the STATCOM current limit, on system base. `q_cap` is an independent MVA ceiling.
        rating = PSY.get_max_shunt_current(fd) / base_mva
        q_cap = PSY.get_max_reactive_power(fd) / base_mva
        svc = PSY.get_shunt_control_type(fd) == PSY.FACTSShuntControlType.SVC
        if rating <= 0.0
            @warn "ControlledFACTS \"$name\": max_shunt_current must be positive \
                (series-only FACTS records parse with 0.0); device not enrolled."
            continue
        end
        vset = PSY.get_voltage_setpoint(fd)
        _validate_vset("ControlledFACTS", name, vset) || continue
        b0 = rating   # enrollment-time bound at unity voltage
        # Warm start from the stored solution: `reactive_power_required` (MVA, delivered
        # Q = b·|V|² at the device's own bus, what `write_device_settings!` writes back) at the
        # bus's stored |V|. The withdrawal debit that matches it is applied when PowerFlowData
        # is built (`_debit_facts_warm_start!`).
        b_init = _facts_warm_start(fd, base_mva, b0)
        push!(
            facts,
            ControlledFACTS(
                name,
                bix,
                cix,
                vset,
                svc,
                rating,
                q_cap,
                b0,
                base_mva,
                b_init,            # initial (reporting)
                b_init,            # warm start (0 without a stored Q)
                false,             # saturated (set post-solve)
            ),
        )
    end
    return
end
