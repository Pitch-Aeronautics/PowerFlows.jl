# Discrete Control Devices via λ-Continuation

PowerFlows.jl supports three families of controlled devices, solved by a common
outer-loop continuation: **tap-changing transformers**, **switched (stepping)
shunts**, and **shunt FACTS devices** (SVC/STATCOM). This page explains what
those devices are, why an outer-loop continuation strategy is used instead of
embedding the control law in the Jacobian, how the sigmoid control law and its
steepness ramp work, and what happens when the continuous solution is snapped
to discrete settings.

## What and why

Commercial power flow tools (PSS/E, PSLF) simulate discrete control devices as
part of the steady-state solution: a tap-changing transformer adjusts its turns
ratio to regulate the voltage at a designated bus; a switched shunt switches
reactive admittance blocks in or out for the same purpose. Without modeling
these devices, the solved bus voltages can differ materially from what a
commercial tool reports, because the network seen by the solver is wrong.

The natural mathematical formulation (Agarwal, Pandey, Jereminov & Pileggi,
arXiv 1811.02000, §IV.C; Pandey PhD thesis 2018, §5.2.8.1) embeds a
differentiable sigmoid approximation of the discrete control law directly in
the residual and Jacobian, so the Newton iteration solves for device parameters
simultaneously with bus voltages. PowerFlows.jl deliberately does **not** use
that implicit embedding for the initial implementation. Instead it wraps the
existing inner solvers in an **outer-loop continuation** (the sigmoid law and
steepness schedule follow the paper; the outer-loop reinterpretation and its
convergence safeguards are original to PowerFlows.jl):

  - The inner solver ([`NewtonRaphsonACPowerFlow`](@ref),
    [`TrustRegionACPowerFlow`](@ref)) is called without any modification.
  - Between outer iterations only the following fields of `data` are mutated:
    Y-bus `nzval` entries for tap devices; the reactive constant-impedance
    withdrawal matrix for shunt/FACTS devices.
  - The outer loop is **formulation-agnostic**: it works identically for
    [`ACPolarPowerFlow`](@ref), [`ACRectangularPowerFlow`](@ref), and
    [`ACMixedPowerFlow`](@ref) because it calls `_solve_with_q_limits!`, the
    existing Q-limit loop, as a black box.

The dispatch point is `_ac_power_flow` in `solve_ac_power_flow.jl`: when
`data.controlled_devices` is non-empty it delegates to
`_control_continuation!`; otherwise the existing path is taken unchanged
(regression invariant).

## [Scope and guards](@id discrete-control-guards)

`control_discrete_devices = true` is validated at construction and rejected
with an `ArgumentError` for combinations that are not yet supported:

  - **Solvers:** only `NewtonRaphsonACPowerFlow` and `TrustRegionACPowerFlow`
    inner solvers. (FastDecoupled factors B′/B″ once and would silently reuse
    them after a tap move; LM/GD/Homotopy are unvalidated as continuation inner
    solvers.)

`time_steps > 1` is supported for every device family (see
[Multiperiod solves](@ref discrete-control-multiperiod)).

Per-device data problems (unresolvable controlled buses, degenerate tap/step
ranges, unsupported MODSW modes, implausible voltage setpoints) never abort
construction: the device is de-enrolled with a `@warn` and stays at its current
setting — the same *warn and lock* posture PSS/E takes for bad control data.

`ControlledFACTS` is enrolled unconditionally (no flag); its metadata sourcing
caveats are noted in [Metadata sourcing](@ref discrete-control-metadata).

## [Multiperiod solves](@id discrete-control-multiperiod)

Each time step regulates independently. `ControlledDeviceSet` carries a
per-time-step store (one column per step, sized at construction); around each
step's continuation the solver loads that step's device state into the scalar
scratch the engine mutates and saves it back afterwards. Shunt/FACTS state is a
plain scalar swap — their control deltas land in the per-column withdrawal
matrices. Taps mutate the *shared* Y-bus, so each step instead resets the tap to
its enrollment baseline via a Y-bus delta-update before regulating
(reset-to-baseline design), and branch flows are computed inside the time-step
loop so every step's flows reflect that step's tap position.

Two multiperiod-specific behaviors:

  - **No PSY write-back:** a PSY component holds a single scalar setting, so
    [`write_device_settings!`](@ref) skips (with a warning) for
    `time_steps > 1`. Per-step settings are reported by
    [`get_controlled_device_results`](@ref), one row per device per time step
    (`time_step` column).
  - **Same-snapshot seeding:** time-invariant inputs (constant-power,
    constant-current, and constant-impedance withdrawals, injections, reactive
    limits) are seeded into every column from the system snapshot; callers with
    time-varying data overwrite columns before solving.

## Device abstraction

Two abstract families sit under `AbstractControlledDevice`:

  - `AbstractBranchControl` — devices that mutate the branch's 2×2 Y-bus block:
    `ControlledTap` (voltage-controlling `TapTransformer`) and
    `ControlledPhaseShifter` (MW-controlling `PhaseShiftingTransformer`, PSS/E COD1 = 3).
  - `AbstractShuntControl` — devices that mutate the bus reactive
    constant-impedance withdrawal: `ControlledSwitchedShunt`
    (voltage-controlling `SwitchedAdmittance`) and `ControlledFACTS`
    (SVC/STATCOM).

The runtime container is `ControlledDeviceSet`, which holds one concretely
typed `Vector` per family. All outer-loop traversal iterates each vector
separately, so each element has a concrete type and no dynamic dispatch occurs;
the per-device kernels
(`apply_parameter!`, `snap_to_discrete`) are allocation-free (the outer loop
itself allocates small snapshot buffers, which is immaterial next to the inner
solves).

For `ControlledTap`, `apply_parameter!` rewrites three of the four cached `nzval`
entries of the sparse Y-bus in place (Y11, Y12, Y21) using cached linear offsets
(`nz_offsets::NTuple{4,Int}`) resolved once at device-set construction; Y22 = Yt
is tap-independent and is skipped. The complex tap includes the winding-group
phase shift `α = PSY.get_α(tx)`, matching PowerNetworkMatrices' stamping
`t = p·e^{iα}`. The delta update `nzval[k] += Y_new − Y_old` preserves any
parallel-branch contributions already in the shared slot; the device's stored
`current` value (not the lossy `nzval`) is the authoritative source for the old
parameter. For
`ControlledSwitchedShunt`, `apply_parameter!` applies the analogous reactive
delta to `data.bus_reactive_power_constant_impedance_withdrawals[bus_ix, ts]`,
so co-located constant-impedance contributions on the same bus are preserved.

## The sigmoid control law

The continuous target for each device is a sigmoid function of the regulated
quantity ``y`` (the controlled-bus voltage magnitude):

```math
\sigma(\ell, h, S, y, y_\text{set}) = \frac{h - \ell}{1 + e^{S(y - y_\text{set})}} + \ell
```

The steepness ``S`` controls how closely the smooth sigmoid approximates the
step function.

```@eval
using Markdown, PowerFlows
Markdown.parse(
    "It starts at `INITIAL_CONTROL_STEEPNESS = $(PowerFlows.INITIAL_CONTROL_STEEPNESS)` and is " *
    "ramped by `CONTROL_STEEPNESS_GROWTH = $(PowerFlows.CONTROL_STEEPNESS_GROWTH)` after each " *
    "settling phase, up to `MAX_CONTROL_STEEPNESS = $(PowerFlows.MAX_CONTROL_STEEPNESS)` " *
    "(values from paper equation 10 / thesis).",
)
```

The ramp happens only after the devices have settled at the current ``S``, so the
solver is never asked to handle a stiff sigmoid before the network state is
compatible with it.

### Plant-sign orientation and gain tracking

The orientation of the control law comes **entirely from the measured plant
sensitivity**, not from the device's wiring. `_plant_sign` measures
``dy/dp`` by a small parameter perturbation at the converged base point (one
inner solve per device; the full pre-probe state — voltages, bus types, and
injections — is restored afterward, so a probe can never permanently flip a
marginal generator PV→PQ). `_control_target` then evaluates the sigmoid with
its limits ordered so the closed-loop gain ``\sigma'(y)\cdot dy/dp`` is
non-positive (negative feedback): measured ``dy/dp > 0`` selects the
decreasing orientation ``(\ell, h) = (p_\min, p_\max)``; measured
``dy/dp \le 0`` selects the increasing orientation ``(h, \ell)``. There is no
stored flip flag and no wiring-based (primary/secondary) orientation logic.

The gain is not frozen at its probed value: every accepted step refreshes it
with a **secant estimate** from the parameter/response pair the step just
produced (zero extra solves). Three conditions freeze a device at its current
parameter with a warning (PSS/E lock-and-continue): a failed probe solve
(orientation unknown), a full-range effectiveness below `CONTROL_GAIN_FLOOR`
(e.g. a PV-pinned controlled bus, which probes exactly 0 — stepping it would
slam the parameter to a rail with no feedback), and a detected **sign
reversal** of the sensitivity along the trajectory (OLTC reverse action —
continuing would be positive feedback). Devices sharing a controlled bus
split the correction (``\omega / n_\text{shared}``), since N co-located
controllers stepping the full error together have an in-phase gain ≈ N× the
measured self-gain.

## The continuation engine

The outer loop `_control_continuation!` runs a steepness ladder of ~7 stages,
each with its own budget of `MAX_CONTROL_PASSES_PER_STAGE = 20` passes. Each
pass:

 1. Computes the sigmoid target ``p^*`` for every device given the current
    regulated quantity and steepness ``S``.
 2. Applies an **adaptive under-relaxation** step
    ``p \leftarrow p + \omega(p^* - p)``.
 3. Applies each update via `_continuation_to!`, the incremental robust
    applicator.

**Under-relaxation.** The damped iteration ``p \leftarrow p + \omega(p^* - p)`` has
local slope ``m = 1 + \omega(g' - 1)``, where ``g' = \sigma'(y)\cdot dy/dp \le 0``
after sign correction. The factor ``\omega`` is chosen to keep ``m`` non-negative
(monotone, ``0 \le m < 1``) with target ``m \ge \theta``, i.e.

```math
\omega \leq \frac{1 - \theta}{1 + |g'|}
```

where `CONTROL_CONTRACTION = 0.5` is the contraction target ``\theta`` and ``g'`` is
the closed-loop gain bound ``|h - \ell| \cdot S/4 \cdot |dy/dp|``
(the maximum sigmoid derivative times the plant gain). Note this formula
already caps ``\omega \le 1-\theta = 0.5``, and the guarantee is conditional on
the plant gain measured at the base point remaining representative along the
trajectory — it is a safeguard, not an unconditional convergence proof.

**Deadband.** A switched shunt whose controlled voltage (local, or the remote
SWREG bus) is anywhere inside its `[VSWLO, VSWHI]` band is held — the PSS/E
semantics. One outside latches the violated edge and regulates toward it; a
discrete (MODSW 1) shunt then stops at the first block step inside the band, the
steps taken in PSS/E's order (see the snap below), and a continuous (MODSW 2)
shunt ends on the edge. `shunt_voltage_band = false` restores the earlier rule:
regulate toward the band midpoint, hold as soon as the voltage crosses in, snap
to the nearest step (which can leave the bus one step short of the band). A
zero-width band (`VSWLO == VSWHI`) is a setpoint. A voltage-controlling tap parsed from PSS/E keeps its
`[VMI1, VMA1]` band (`ext`, while `voltage_setpoint` is still its midpoint): it
is held while its regulated voltage (local or remote) lies inside the band; one
outside latches the violated edge, regulates toward it, and the snap then takes
the first discrete tap position inside the band, stepping one more position
while the voltage is still outside (the tap limits hold). `tap_voltage_band =
false` on the power flow restores the earlier behaviour, driving every tap to
`voltage_setpoint` with no deadband. FACTS devices, and taps without a parsed
band, carry a point setpoint and always regulate.

**Tap orientation.** Every device is oriented by its measured dy/dp, so the
loop is negative feedback whatever the wiring. Two refinements for taps:
parallel taps regulating one bus (same terminals, same regulated bus) whose
measured signs disagree — each unit's own effect is weak, or a reactive
current circulates between units at unequal ratios — are aligned to the sign
of their summed sensitivity, so they move together instead of walking apart
(a tap fight); their measured magnitudes still size the steps. And
`tap_direction = :cont_sign` orients a tap by PSS/E's convention for the sign
of CONT1 instead (`ext["CONT1"]`, or `ext["psse"]["CONT1"]`): positive, the
ratio is adjusted as if the regulated bus were on the winding-2 side (raising
the winding-1 ratio lowers it); negative, as if on the winding-1 side. Where
the convention and the network disagree the tap runs to a limit, as PSS/E's
would. The default, `:sensitivity`, keeps the measured sign.

**Impedance correction.** A tap transformer with a `TAP_RATIO` impedance
correction table (`PSY.ImpedanceCorrectionData`) re-evaluates the table's
factor at every tap the controller applies, so each step is sized on, and the
Y-bus re-stamped with, the corrected impedance at the new tap (PNM's `Ybus`
evaluates it at the build-time tap). Reported branch flows use the same
corrected block at the solved tap.

**Incremental applicator.** `_continuation_to!` tries the full damped move
first — in the common case one warm-started inner solve accepts it. Only when
that fails does it fall back to bisection sub-stepping (starting at half the
interval, growing by `CONTROL_STEP_GROWTH = 1.5` on success and halving on
failure). If the step falls below `MIN_LAMBDA_STEP = 1e-3` the parameter stays
at the last converged point; a device that could not move at all despite a
requested move is frozen with a warning rather than counted as settled.
Intermediate steepness stages solve at a relaxed tolerance
(`CONTROL_STAGE_TOL = 1e-6`); the final stage and the snap/restore solves use
the full tolerance. `get_control_inner_solve_count(data)` reports the
continuation's inner-solve count — the metric its performance is measured by.

**Settling and steepness ramp.** A pass is considered settled when every device
moves by less than its scale-aware tolerance
``\max(10^{-5},\ 10^{-4}\cdot(p_\max - p_\min))``. Each steepness stage has
its own budget of `MAX_CONTROL_PASSES_PER_STAGE = 20` passes, so a
slow-settling early stage cannot starve the stiffer later stages; ``S`` is
advanced when the stage settles (or its budget runs out). If the final stage
does not settle, an honest `@warn` is emitted (regulation may be loose); the
solver does not claim failure on account of steepness alone.

**Oscillation safeguard.** If the direction of a device's parameter update
reverses sign more than `CONTROL_OSCILLATION_LIMIT = 3` times **within one
steepness stage** (the counter and direction memory reset at each ramp, since
a ramp legitimately reverses the update direction once; sub-tolerance moves
carry no direction information), the device is frozen at its current parameter
with a `@warn`. Frozen devices count as settled — one locked device does not
block the steepness ramp for the healthy ones. This mirrors the PSS/E behavior
of locking oscillating adjustments after a fixed iteration count.

## Snap and feasibility restoration

After the continuous outer loop converges (or reaches its iteration limit),
`snap_and_restore!` discretizes every device:

  - **Tap.** The continuous parameter is snapped to the nearest value in the
    pre-computed `levels` vector (a uniform grid of `NTP` tap positions from
    `p_min` to `p_max`).
  - **Shunt.** The target susceptance is snapped onto the shunt's `levels`, the
    realizable totals built once at enrollment (`_shunt_levels`). With PSS/E
    `ADJM = 0` blocks switch on in listed order and off in reverse, capacitor
    and reactor blocks each forming a chain away from the all-off base, so the
    totals are the prefix sums of each chain; with `ADJM = 1` every combination
    of steps is a level, so a step goes to the next higher or lower total
    admittance. A banded shunt regulating toward an edge snaps on in the
    direction it moved to the first level beyond its continuous value;
    otherwise the nearest level. `block_n` records the per-block step counts,
    and `snap_to_discrete` makes no heap allocations. (Continuous devices —
    MODSW=2 shunts, FACTS — clamp instead of snapping.)
  - **Band settle.** For banded taps and discrete shunts the snapped network is
    then checked: a device still outside on its latched side steps on, one
    whose step back toward its anchor would still be inside steps back (jointly,
    then alone for devices that pushed each other out), and a final forward
    pass, so each device ends at its first position inside.
  - **Continuous finish.** A regulating continuous device (a continuous shunt
    with a zero-width band or latched on an edge, and every FACTS) then takes
    Newton steps on its susceptance, with one joint solve per round, until its
    voltage is within `CONTROL_FINISH_VTOL` of its target or the step clamps
    at a susceptance limit (a FACTS's |V|-dependent bound refreshed every
    round), where it then sits exactly, so a FACTS that cannot hold its
    setpoint is classified saturated at |Q| = SHMX·V. Without it the sigmoid
    law's finite steepness leaves such a device ~1e-4 p.u. off its setpoint,
    or short of the limit it is pinned at. During the continuation itself a
    continuous device whose target is within its settle tolerance of a bound
    while it is off setpoint is put on the bound (the rail snap), instead of
    creeping toward it in ever smaller damped steps.

After snapping all devices, the inner solver is called on the discretized
network. If it converges, the procedure is complete. If not, each device is
individually restored toward its pre-snap continuous value using the same
bisection sub-stepping as `_continuation_to!` (`_restore_one!`). If any
device cannot be restored to a converged state, `data.converged[ts] = false`
is set and an `@error` is emitted with the device names; no non-physical
solution is silently returned.

### State checkpoint and rollback during continuation

The outer loop maintains a checkpoint at the beginning of each continuation
attempt: V/θ, `bus_type`, the two bus injection columns, and, for systems
carrying HVDC, the VSC DC-network columns and the LCC converter columns
(per-time-step taps, thyristor angles, and DC current). It does not cover
Ybus tap deltas, shunt/FACTS withdrawal vectors, or device parameters — those
are restored separately, at the call sites, by `apply_parameter!` resetting
each device to its pre-attempt value. The orientation-probe phase
(`_plant_sign`) captures and restores this checkpoint unconditionally on
every control-enabled solve, not only when an attempt fails; when a
continuation attempt itself fails to converge, the checkpoint is restored to
roll back the bus/DC-network/LCC mutations the attempt made. When the
checkpoint is restored, the derived LCC caches (`phi`, `branch_admittances`)
are re-derived from the restored converter parameters at the restored
network state, rather than snapshotted.

## [Metadata sourcing](@id discrete-control-metadata)

Device parameters are sourced from the PSY component's first-class fields when
the installed PowerSystems.jl provides them (PSY ≥ 6 via PSY #1705 / the psy6
development branch — detected once at load time), and otherwise from the PSS/E
parser's `ext` dictionary with documented defaults. The `ext` keys the parser
actually writes are winding-suffixed; unsuffixed spellings are accepted as
user-facing overrides.

### `TapTransformer` → `ControlledTap`

Only transformers with `get_control_objective(tx) == VOLTAGE` are included.

| Parameter        | `ext` keys (parser first)            | First-class field (PSY ≥ 6)               | Default on PSY 5.x |
|:---------------- |:------------------------------------ |:----------------------------------------- |:------------------ |
| Controlled bus   | `CONT1`, `NREG`                      | `regulated_bus_number` (0 ⇒ local/to-bus) | to-bus             |
| Tap ratio min    | `RMI1`, `RMI`                        | `tap_limits.min`                          | `0.9`              |
| Tap ratio max    | `RMA1`, `RMA`                        | `tap_limits.max`                          | `1.1`              |
| Tap positions    | `NTP1`, `NTP`                        | `number_of_tap_positions`                 | `33`               |
| Voltage setpoint | `VSET` (never written by the parser) | `voltage_setpoint`                        | `1.0`              |
| Voltage band     | `VMI1`, `VMA1` (PSS/E parser)        | (none on PSY 5.x)                         | no band            |

Precedence: a nonzero `regulated_bus_number` wins for the controlled bus;
for the numeric parameters the `ext` keys win over the first-class fields
(so parsed PSS/E data keeps working until the parser populates the fields).
The discrete tap levels are `range(p_min, p_max; length=NTP)`, collected once
at construction. The winding-group phase shift is taken from `PSY.get_α(tx)`.
Setpoints outside `[0.5, 1.5]` p.u. de-enroll the device with a warning.

### `SwitchedAdmittance` → `ControlledSwitchedShunt`

| Parameter         | Source                                                                                                           |
|:----------------- |:---------------------------------------------------------------------------------------------------------------- |
| Control mode      | `ext["MODSW"]`: 0 ⇒ locked (skipped); 1 ⇒ discrete; 2 ⇒ continuous; ≥3 ⇒ unsupported (warn + lock). Absent ⇒ 1.  |
| Controlled bus    | `ext["NREG"]` (v35) or `ext["SWREM"]` (v32/33), else own bus                                                     |
| Voltage setpoint  | midpoint of `get_admittance_limits` — the VSWLO/VSWHI band for parsed systems                                    |
| Susceptance range | spanned by the blocks: `[Σ min(steps·dB, 0), Σ max(steps·dB, 0)]` (plus the fixed base for API-built components) |
| Block structure   | `get_number_of_steps`, `get_Y_increase`, `get_initial_status`                                                    |
| Voltage band      | `get_admittance_limits` = `[VSWLO, VSWHI]` (with `shunt_voltage_band`, the default)                              |
| Block order       | `ext["ADJM"]`: 0 (default) ⇒ input order, by sign; 1 ⇒ next higher/lower total admittance                        |

Two `Y`/`initial_status` conventions exist and are auto-detected via the
presence of the parser's `MODSW` key: the **PSS/E parser** stores
`Y = BINIT` (the *total* in-service admittance) and zeroes `initial_status`,
so the reachable range is spanned by the blocks alone with the current point
at BINIT; **API-built** components follow the PSY docstring (`Y` = fixed N=0
base, `initial_status` meaningful).

### `FACTSControlDevice` → `ControlledFACTS`

A FACTS device starts from its stored solution: `reactive_power_required`
(MVA, what `write_device_settings!` writes) over `base_power·|V|²` at its bus's
stored magnitude, clamped to `±rating`, and that susceptance is debited from the
bus's constant-impedance reactive withdrawal when `PowerFlowData` is built.
Without a stored Q it starts at 0.

A `FACTSControlDevice` in a non-out-of-service control mode enrolls
unconditionally (no flag) as a *continuous* shunt: it varies a symmetric
susceptance `b` to hold its regulated bus at `voltage_setpoint`, injecting
`Q = b·|V|²`.

| Parameter          | `ext` keys (parser first)       | First-class field (PSY ≥ 6)                    | Default on PSY 5.x |
|:------------------ |:------------------------------- |:---------------------------------------------- |:------------------ |
| Regulated bus      | `FCREG` (v35), `REMOT` (v32/33) | `regulated_bus_number` (0 ⇒ local/sending bus) | local bus          |
| Voltage setpoint   | `VSET`                          | `voltage_setpoint`                             | `1.0`              |
| Shunt current cap  | `SHMX`                          | `max_shunt_current` (MVA at unity voltage)     | `9999`             |
| Reactive power cap | —                               | `max_reactive_power` (MVA)                     | `9999`             |
| Device class       | —                               | `shunt_control_type` (SVC / STATCOM)           | STATCOM            |

`SHMX` is the shunt *current* capability, **not** a reactive-power limit; the
series-branch current `IMX` and `RMPCT` are out of scope and left in `ext`.
`reactive_power_required` is a solver *output* (the delivered `Q`), not a parsed
input.

**Voltage-dependent limit.** The effective susceptance bound `b ∈ [−b_lim, b_lim]`
is refreshed each outer iteration from the measured regulated-bus voltage `V`,
with `rating = SHMX/S_base` and `q_cap = max_reactive_power/S_base`:

  - **SVC** (susceptance-limited): `b_lim = min(rating, q_cap/V²)`
  - **STATCOM** (current-limited): `b_lim = min(rating/V, q_cap/V²)`

so a STATCOM's reactive capability falls off linearly with voltage (`Q ≈ V·SHMX`
at the current limit) while an SVC's falls off as `V²`. At the bound the clamp
holds `b` there — the homotopy analogue of the PV→PQ Q-limit release. To keep the
refreshing bound from chattering, a device that has been oscillation-frozen holds
its last `b_lim` instead of continuing to track voltage.

**Saturation reporting.** After the solve each FACTS device is classified: one
pinned at `b_lim` while its regulated bus remains off `voltage_setpoint` is
flagged `saturated = true` (with a warning). `get_controlled_device_results`
carries that flag plus `delivered_q_mvar = b·|V|²·S_base` on the FACTS rows, and
under active controls the solved `Q` is written back to the component's
`reactive_power_required` (single time step only; see
[Multiperiod solves](@ref discrete-control-multiperiod)).

### `PhaseShiftingTransformer` → `ControlledPhaseShifter`

A phase shifter under PSS/E active-power control (COD1 = 3) adjusts its angle in steps to hold
its MW flow inside a band, and stops at the first step inside (PSS/E's PHSHFT adjustment). It is
the fourth family: it is indexed after the FACTS devices, goes through the same batched pass,
band latch, directional snap, band settle and parallel-unit grouping as a tap, and is turned
off by `phase_shifter_control = false`. PSS/E reuses the voltage fields for COD 3: `VMA1`/`VMI1`
are the band in MW and `RMA1`/`RMI1` the angle limits in degrees. PowerSystems 5's reader keeps
them in `ext` only (it never sets `phase_angle_limits`), so `ext` comes first. Each key is read
from `ext[key]` or `ext["psse"][key]`.

| Parameter     | Source                                                                                                           |
|:------------- |:---------------------------------------------------------------------------------------------------------------- |
| Enrolment     | `control_objective == ACTIVE_POWER_FLOW` (the reader sets it from COD1), or `UNDEFINED` with `COD1 = 3`          |
| Angle limits  | `RMI1`, `RMA1` (degrees); else `phase_angle_limits` (radians) when it is not PowerSystems' default ±π; else lock |
| MW band       | `VMI1`, `VMA1` (MW, over the system base); required. `VMI1 = VMA1` is a point setpoint                           |
| Positions     | `NTP1` over `[RMI1, RMA1]`; absent ⇒ 33 (PSS/E's default); below 2 ⇒ continuous                                  |
| Metered end   | `NMETR = 1` ⇒ winding 2; otherwise winding 1 (PSS/E's default `NMETR = 2`)                                       |
| Initial angle | `get_α` (radians); outside the limits ⇒ lock                                                                     |

COD1 = −3 is PSS/E's "adjustment suppressed" (`ACTIVE_POWER_FLOW_DISABLED`): the angle is held.
The regulated quantity is the winding-1 → winding-2 active power at the metered end: at
winding 1 the power entering the transformer from its `from` bus (magnetizing shunt included,
as the reported flow), at winding 2 the power leaving it into the `to` bus. Raising α (PNM's
stamp `t = tap·e^{jα}` on the `from` side) lowers that flow; the direction is taken from the
measured sensitivity dP/dα, which the polar path computes from the sensitivity context as
∂P/∂α + ∇ₓP·dx/dα, one triangular solve per device. Parallel units (the same `from` and `to`
buses) are one control: each measures the group's mean flow and keeps its angle offset, so they
step together; the band settle never backs one off alone. A `PHASE_SHIFT_ANGLE` impedance
correction table is re-evaluated at every angle. The solved angle is reported in
`get_controlled_device_results` (family `"PhaseShiftingTransformer"`), used for the reported
flows, and written back with `set_α!` by `solve_and_store_power_flow!`. DC power flow keeps the
stored angle.

## Remote voltage regulation by generators

A generator that regulates another bus X (PSS/E's IREG) is not a discrete device; it is part of
the polar solve itself (`remote_voltage_control = true`, the default, on `ACPolarPowerFlow`).
The fields come from the generator's `ext` (or `ext["psse"]`): `IREG` the regulated bus,
`RMPCT` its percent share (default 100), `VS` X's scheduled voltage (default X's stored
magnitude; PowerSystems' v33 reader keeps IREG and RMPCT but not VS, its v35 reader none of
them). X is PV in the solve at that setpoint and its first state slot holds the group's total
reactive power Q_tot; each plant regulating X has its own bus PQ, with |V| free and reactive
injection s/Σs · Q_tot, s its RMPCT share. X's own regulating plant, if it has one, shares too.
With `check_reactive_power_limits`, a plant past its bus's Q bound is fixed there and the shares
renormalise; when every plant is fixed X becomes PQ, and it returns to PV at its setpoint when
|V_X| crosses it (the PQ→PV return of the reactive-limit check, applied to the group).

A device regulating X (a tap, shunt or FACTS) sees no voltage change while the group holds X,
so its probe freezes it at its stored setting; when the group is saturated at the base solve, X
is PQ and the device regulates it alone, to its own setpoint. Remote-regulating generators then
no longer count as co-regulators for the shared-setpoint band of a point-setpoint shunt. The
fast decoupled solvers and the rectangular and mixed formulations do not model remote
regulation (each unit holds its own terminal).

## Reserved seams for future work

`stamp_control!(d::AbstractControlledDevice, args...)` is defined on the
abstract base and calls `error("implicit embedding not implemented for ...")`.
It is the attachment point for a future per-formulation implicit embedding
(sigmoid term added to the residual and corresponding Jacobian row), which
would allow the inner Newton iteration to solve for device parameters
simultaneously with bus voltages. Such an embedding would be added per
formulation type by overloading `stamp_control!`; the outer-loop code and the
device hierarchy would require no changes.
