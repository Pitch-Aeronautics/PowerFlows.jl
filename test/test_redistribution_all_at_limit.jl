# Issue #231: the write-back shares a bus's solved generation among its units, and a
# residual that no unit can take used to fail `@assert length(remaining_unit_index) == 1`
# with an empty vector. A solve can hold a bus PV, or the reference, past its units'
# combined limits (remote voltage control, a regulated bus at its limit, no reactive-limit
# check), so every unit ends at its limit. The residual now goes to one unit, past its limit,
# with a warning: the unit with the widest range of the quantity, or the largest by base
# power when no unit has any range.

"""REF bus 1 ─ bus 2 (`bus2_type`, |V| = `v2`), a load of `load2_mw` + j40 MVA at bus 2.
`units` are bus 2's generators, each `(P, Q limits, base power)`; `ref_units` puts the same
at bus 1 in place of a source. Under PV, bus 2 makes Q ≈ 0.9 pu."""
function _all_at_limit_system(;
    units = [],
    ref_units = nothing,
    bus2_type = ACBusTypes.PV,
    v2 = 1.05,
    load2_mw = 20.0,
)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, bus2_type, 230, v2, 0.0)
    gens = Dict(1 => ThermalStandard[], 2 => ThermalStandard[])
    if isnothing(ref_units)
        _add_simple_source!(sys, b1, 0.0, 0.0)
    else
        for u in ref_units
            push!(gens[1], _add_unit!(sys, b1, u...))
        end
    end
    for u in units
        push!(gens[2], _add_unit!(sys, b2, u...))
    end
    _add_simple_load!(sys, b2, load2_mw, 40.0)
    _add_simple_line!(sys, b1, b2, 0.01, 0.1, 0.0)
    set_units_base_system!(sys, "SYSTEM_BASE")
    return sys, gens
end

function _add_unit!(sys, bus, p, q_limits, base_power)
    g = _add_simple_thermal_standard!(sys, bus, p, 0.0)
    set_reactive_power_limits!(g, q_limits)
    set_rating!(g, 10.0)
    set_base_power!(g, base_power)
    return g
end

"""The solved generation at `bus_number`, in system per unit, from a solve that writes
nothing back."""
function _solved_generation(sys, bus_number)
    df = solve_power_flow(ACPowerFlow(), sys)["bus_results"]
    row = only(eachrow(df[df.bus_number .== bus_number, :]))
    return (p = row.P_gen / 100.0, q = row.Q_gen / 100.0)
end

@testset "Write-back: a residual with every unit at its limit (#231)" begin
    @testset "PV bus, two units whose combined Q limits are below the bus's Q" begin
        sys, gens = _all_at_limit_system(;
            units = [
                (0.25, (min = -0.1, max = 0.1), 100.0),
                (0.25, (min = -0.15, max = 0.15), 100.0),
            ],
        )
        q_bus = _solved_generation(sys, 2).q
        @test q_bus > 0.1 + 0.15  # beyond both units' limits together
        narrow, wide = gens[2]
        @test @test_logs(
            (:warn, r"Every unit at bus bus_2 is at its reactive power limit"),
            match_mode = :any,
            solve_and_store_power_flow!(ACPowerFlow(), sys),
        )
        @test get_reactive_power(narrow) ≈ 0.1 atol = 1e-6
        @test get_reactive_power(wide) ≈ q_bus - 0.1 atol = 1e-6
        @test get_reactive_power(wide) > 0.15
    end

    @testset "PV bus, a unit with no Q capability and a limited one" begin
        # The unit with no capability is the larger by base power: the residual still goes
        # to the unit with a range.
        sys, gens = _all_at_limit_system(;
            units = [
                (0.05, (min = 0.0, max = 0.0), 500.0),
                (0.25, (min = -0.1, max = 0.1), 100.0),
            ],
        )
        q_bus = _solved_generation(sys, 2).q
        none, limited = gens[2]
        @test @test_logs(
            (:warn, r"Every unit at bus bus_2 is at its reactive power limit"),
            match_mode = :any,
            solve_and_store_power_flow!(ACPowerFlow(), sys),
        )
        @test get_reactive_power(none) == 0.0
        @test get_reactive_power(limited) ≈ q_bus atol = 1e-6
    end

    @testset "PV bus, no unit with any Q capability: the largest by base power" begin
        sys, gens = _all_at_limit_system(;
            units = [
                (0.5, (min = 0.0, max = 0.0), 50.0),
                (0.125, (min = 0.0, max = 0.0), 200.0),
            ],
        )
        q_bus = _solved_generation(sys, 2).q
        small, large = gens[2]
        @test @test_logs(
            (:warn, r"Every unit at bus bus_2 is at its reactive power limit"),
            match_mode = :any,
            solve_and_store_power_flow!(ACPowerFlow(), sys),
        )
        @test get_reactive_power(small) == 0.0
        @test get_reactive_power(large) ≈ q_bus atol = 1e-6
    end

    @testset "Reference bus, two units whose combined P limits are below the bus's P" begin
        sys, gens = _all_at_limit_system(;
            ref_units = [
                (0.1, (min = -5.0, max = 5.0), 100.0),
                (0.1, (min = -5.0, max = 5.0), 100.0),
            ],
            bus2_type = ACBusTypes.PQ,
            v2 = 1.0,
            load2_mw = 100.0,
        )
        narrow, wide = gens[1]
        set_active_power_limits!(narrow, (min = 0.0, max = 0.2))
        set_active_power_limits!(wide, (min = 0.0, max = 0.3))
        p_bus = _solved_generation(sys, 1).p
        @test p_bus > 0.2 + 0.3
        @test @test_logs(
            (:warn, r"Every unit at bus bus_1 is at its active power limit"),
            match_mode = :any,
            solve_and_store_power_flow!(ACPowerFlow(), sys),
        )
        @test get_active_power(narrow) ≈ 0.2 atol = 1e-6
        @test get_active_power(wide) ≈ p_bus - 0.2 atol = 1e-6
    end
end
