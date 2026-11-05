# A local unit stored at a Q limit, on PSS/E's side of its VS, starts PQ at that limit with VS kept
# as its setpoint (`_start_local_units_at_stored_limits!`, GAP-015, GAP-027): the groups' saturated
# start applied to single units. Limits otherwise apply only after a converged solve, and a unit
# held PV can screen a remote-regulation group's regulated bus from its plants, so the first solve
# cannot converge.

"""REF 1 ─ unit 2 ─ PQ load 3. The unit's |V| is stored at `v`, its Q at `q`, its limits are
`qmin`, `qmax`, its scheduled voltage `vs` (`nothing`: no VS stated)."""
function _stored_limit_system(; v = 1.0, q = 0.0, qmin = -0.1, qmax = 0.1, vs = 1.03,
    units = 1, ireg = nothing, source = false)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230, v, 0.0)
    for u in 1:units
        g = _add_simple_thermal_standard!(sys, b2, 0.2 / units, q / units)
        set_reactive_power_limits!(g, (min = qmin / units, max = qmax / units))
        psse = Dict{String, Any}()
        isnothing(vs) || (psse["vs"] = vs isa Function ? vs(u) : vs)
        isnothing(ireg) || (psse["ireg"] = ireg)
        get_ext(g)["psse"] = psse
    end
    source && set_reactive_power_limits!(_add_simple_source!(sys, b2, 0.0, 0.0),
        (min = -0.5, max = 0.5))
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_load!(sys, b3, 50.0, 20.0)
    _add_simple_line!(sys, b1, b2, 0.005, 0.1, 0.0)
    _add_simple_line!(sys, b2, b3, 0.005, 0.05, 0.0)
    return sys
end

_stored_limit_pf(; kw...) = ACPolarPowerFlow(; check_reactive_power_limits = true,
    remote_voltage_control = false, kw...)

function _stored_limit_data(; pf = _stored_limit_pf(), kw...)
    data = PowerFlowData(pf, _stored_limit_system(; kw...))
    return data, PF.get_bus_lookup(data)[2]
end

"""The state a solver that holds the unit at `q` reaches: its |V| at that Q, for storing."""
function _v_at_q(q; kw...)
    sys = _stored_limit_system(; q, qmin = q, qmax = q, vs = nothing, kw...)
    data = PowerFlowData(_stored_limit_pf(; enhanced_flat_start = false), sys)
    b2 = PF.get_bus_lookup(data)[2]
    data.bus_type[b2, 1] = ACBusTypes.PQ
    @test solve_power_flow!(data)
    return data.bus_magnitude[b2, 1]
end

@testset "stored Q limit start: a unit at its limit on PSS/E's side starts PQ" begin
    @testset "at Qmax with |V| below VS" begin
        v = _v_at_q(0.1)
        data, b2 = _stored_limit_data(; v, q = 0.1, vs = v + 0.02)
        @test data.bus_type[b2, 1] == ACBusTypes.PQ
        @test data.bus_reactive_power_injections[b2, 1] == 0.1
        @test PF.get_q_limit_stored_start_buses(data) == [b2]
        @test data.q_limit_switches.vset[(b2, 1)] == v + 0.02
        @test solve_power_flow!(data)
        @test data.bus_type[b2, 1] == ACBusTypes.PQ
        @test data.bus_magnitude[b2, 1] ≈ v atol = 1e-8
        @test isempty(PF.get_q_limit_stored_start_returned_buses(data))
        @test PF.get_q_limit_return_count(data) == 0
    end
    @testset "at Qmin with |V| above VS" begin
        v = _v_at_q(-0.1)
        data, b2 = _stored_limit_data(; v, q = -0.1, vs = v - 0.02)
        @test data.bus_type[b2, 1] == ACBusTypes.PQ
        @test data.bus_reactive_power_injections[b2, 1] == -0.1
        @test solve_power_flow!(data)
        @test data.bus_magnitude[b2, 1] ≈ v atol = 1e-8
    end
    @testset "a limit summed over the units at the bus" begin
        v = _v_at_q(0.1)
        data, b2 = _stored_limit_data(; v, q = 0.1, vs = v + 0.02, units = 3)
        @test PF.get_q_limit_stored_start_buses(data) == [b2]
    end
    @testset "a fixed-Q unit is at both limits; it never returns" begin
        v = _v_at_q(0.1)
        data, b2 = _stored_limit_data(; v, q = 0.1, qmin = 0.1, qmax = 0.1, vs = v - 0.02)
        @test PF.get_q_limit_stored_start_buses(data) == [b2]
        @test solve_power_flow!(data)
        @test data.bus_type[b2, 1] == ACBusTypes.PQ
    end
    @testset "every time step" begin
        v = _v_at_q(0.1)
        pf = _stored_limit_pf(; time_steps = 2, time_step_names = ["a", "b"])
        data, b2 = _stored_limit_data(; pf, v, q = 0.1, vs = v + 0.02)
        @test PF.get_q_limit_stored_start_buses(data, 1) == [b2]
        @test PF.get_q_limit_stored_start_buses(data, 2) == [b2]
        @test all(==(ACBusTypes.PQ), view(data.bus_type, b2, :))
    end
end

@testset "stored Q limit start: where it must not fire" begin
    v = _v_at_q(0.1)
    # every case below keeps the unit PV at its stored |V|
    function held(; kw...)
        data, b2 = _stored_limit_data(; kw...)
        return data.bus_type[b2, 1] == ACBusTypes.PV &&
               isempty(PF.get_q_limit_stored_start_buses(data)) &&
               !haskey(data.q_limit_switches.vset, (b2, 1)) &&
               data.bus_magnitude[b2, 1] == v
    end
    @test !held(; v, q = 0.1, vs = v + 0.02)   # the control: this one fires
    @test held(; v, q = 0.1, vs = v)                  # at its limit and at VS: it holds VS
    @test held(; v, q = 0.1, vs = v - 0.02)           # above VS at Qmax: not PSS/E's side
    @test held(; v, q = -0.1, vs = v + 0.02)          # below VS at Qmin: not PSS/E's side
    @test held(; v, q = 0.05, vs = v + 0.02)          # off VS and inside its range: a mismatch to report
    @test held(; v, q = 0.1 - 2 * PF.STORED_Q_LIMIT_TOLERANCE, vs = v + 0.02)
    @test !held(; v, q = 0.1 - PF.STORED_Q_LIMIT_TOLERANCE / 2, vs = v + 0.02)
    @test held(; v, q = 0.1, vs = nothing)            # no VS: no side to be on
    @test held(; v, q = 0.1, vs = u -> u == 1 ? v + 0.02 : v + 0.03, units = 2)   # VS disagree
    @test held(; v, q = 0.1, vs = v + 0.02, ireg = 3) # remote IREG, its VS is bus 3's
    @test held(; v, q = 0.1, vs = v + 0.02, source = true)   # a source's limits are not the units'
    @test held(; v, q = 0.1, qmin = -Inf, qmax = Inf, vs = v + 0.02)  # unbounded
    @test held(; pf = _stored_limit_pf(; start_at_stored_q_limits = false), v, q = 0.1,
        vs = v + 0.02)
    @test held(; pf = ACPolarPowerFlow(; check_reactive_power_limits = false,
            remote_voltage_control = false), v, q = 0.1, vs = v + 0.02)
    @test held(; pf = ACPolarPowerFlow{FastDecoupledXB}(; check_reactive_power_limits = true,
            remote_voltage_control = false), v, q = 0.1, vs = v + 0.02)
end

@testset "stored Q limit start: the solve can leave the stored limit" begin
    @testset "a stored state the network does not reach returns to PV at VS" begin
        # Stored at Qmax = 0.1 below VS, but at 1.03 the network wants less than Qmax: PQ at Qmax
        # puts |V| above VS, the return brings it to PV at VS, off the limit.
        v = _v_at_q(0.1)
        vs = v - 0.005
        @test vs < v
        # at Qmax |V| = v > vs although it is stored below: store |V| below VS to fire
        data, b2 = _stored_limit_data(; v = vs - 0.01, q = 0.1, vs)
        @test PF.get_q_limit_stored_start_buses(data) == [b2]
        @test solve_power_flow!(data)
        @test PF.get_q_limit_stored_start_returned_buses(data) == [b2]
        @test data.bus_type[b2, 1] == ACBusTypes.PV
        @test data.bus_magnitude[b2, 1] == vs
        @test data.bus_reactive_power_injections[b2, 1] < 0.1
    end
    @testset "released at the other limit after it returns" begin
        # Stored at Qmax below VS, but holding VS takes Q under Qmin: PQ at Qmax rises past VS,
        # the return makes it PV at VS, and the check then releases it at Qmin.
        v = _v_at_q(0.1)
        vs = v - 0.03
        data, b2 = _stored_limit_data(; v = vs - 0.01, q = 0.1, qmin = 0.05, vs)
        @test solve_power_flow!(data)
        @test data.bus_type[b2, 1] == ACBusTypes.PQ
        @test data.bus_reactive_power_injections[b2, 1] == 0.05
        @test data.bus_magnitude[b2, 1] > vs
    end
    @testset "a start that is not the stored state reaches the same solution" begin
        v = _v_at_q(0.1)
        ref, b2 = _stored_limit_data(; v, q = 0.1, vs = v + 0.02)
        @test solve_power_flow!(ref)
        data, b2 = _stored_limit_data(; v, q = 0.1, vs = v + 0.02)
        data.bus_magnitude[:, 1] .= [data.bus_magnitude[1, 1], 1.0, 1.0]
        data.bus_angles[:, 1] .= 0.0
        @test solve_power_flow!(data)
        @test data.bus_magnitude ≈ ref.bus_magnitude atol = 1e-8
        @test data.bus_type[b2, 1] == ACBusTypes.PQ
    end
    @testset "a second solve on the same data, as a contingency runner makes" begin
        v = _v_at_q(0.1)
        data, b2 = _stored_limit_data(; v, q = 0.1, qmin = -0.5, vs = v + 0.02)
        @test solve_power_flow!(data)
        b3 = PF.get_bus_lookup(data)[3]
        # the load drops: |V| rises past VS at Qmax, and the unit regulates again
        data.bus_active_power_withdrawals[b3, 1] = 0.0
        data.bus_reactive_power_withdrawals[b3, 1] = -0.2
        @test solve_power_flow!(data)
        @test PF.get_q_limit_stored_start_returned_buses(data) == [b2]
        @test data.bus_type[b2, 1] == ACBusTypes.PV
        @test data.bus_magnitude[b2, 1] == v + 0.02
        @test -0.5 < data.bus_reactive_power_injections[b2, 1] < 0.1
    end
end

"""REF 1 ─ X 2 ─ held 3 ─ plant 4, the plant reaching the network only through 3, regulating X by
IREG. A unit at 3 is stored at Qmax below its VS."""
function _screened_group_system(; v3 = 1.0)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_load!(sys, b2, 50.0, 20.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PV, 230, v3, 0.0)
    g3 = _add_simple_thermal_standard!(sys, b3, 0.0, 0.0)
    set_reactive_power_limits!(g3, (min = -0.1, max = 0.0))
    get_ext(g3)["psse"] = Dict{String, Any}("vs" => v3 + 0.02)
    b4 = _add_simple_bus!(sys, 4, ACBusTypes.PV, 230, 1.0, 0.0)
    g4 = _add_simple_thermal_standard!(sys, b4, 0.2, 0.0)
    set_reactive_power_limits!(g4, (min = -3.0, max = 3.0))
    get_ext(g4)["psse"] = Dict{String, Any}("ireg" => 2, "rmpct" => 100.0, "vs" => 1.0)
    _add_simple_line!(sys, b1, b2, 0.005, 0.1, 0.0)
    _add_simple_line!(sys, b2, b3, 0.005, 0.05, 0.0)
    _add_simple_line!(sys, b3, b4, 0.005, 0.05, 0.0)
    return sys
end

@testset "stored Q limit start: a group's regulated bus behind a held bus" begin
    solve(; kw...) = with_logger(NullLogger()) do
        data = PowerFlowData(ACPolarPowerFlow(; check_reactive_power_limits = true,
                remote_voltage_control = true, kw...), _screened_group_system())
        (; ok = solve_power_flow!(data), data)
    end
    # The unit at 3 PV: Q_tot moves nothing at X, whose row Newton then cannot close.
    @test !solve(; start_at_stored_q_limits = false).ok
    r = solve()
    @test r.ok
    bl = PF.get_bus_lookup(r.data)
    @test r.data.bus_type[bl[3], 1] == ACBusTypes.PQ
    @test r.data.bus_type[bl[2], 1] == ACBusTypes.PV
    @test r.data.bus_magnitude[bl[2], 1] == 1.0
    @test PF.get_q_limit_stored_start_buses(r.data) == [bl[3]]
end
