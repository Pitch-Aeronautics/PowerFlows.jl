# A bus the solve reports PV can have no available source for the write-back to share its
# Q among: a remote-regulation member (`get_reported_bus_type`), or a bus held PV by a
# switched shunt (`must_be_PV` counts one as PSS/E does) whose machines are all out. The
# write-back's PV branch then failed `_reactive_power_redistribution_pv`'s
# `error("No devices in bus …")` after a converged solve. It now skips such buses, with one
# warning that counts them, and the redistribution warns and returns where it has no device.

"""REF bus 1 (a source) ─ PV bus 2 ─ PV bus 3. Buses 2 and 3 each hold a switched shunt,
fixed, an out-of-service generator (stored Q 0.07) and a load: PV by the shunt, with no
available source. Returns the system, the generators and the shunts."""
function _pv_without_source_system()
    sys = System(100.0)
    gens, shunts = ThermalStandard[], SwitchedAdmittance[]
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    prev = b1
    for n in (2, 3)
        b = _add_simple_bus!(sys, n, ACBusTypes.PV, 230, 1.01, 0.0)
        shunt =
            SwitchedAdmittance(; name = "shunt_$n", available = true, bus = b, Y = 0.05im)
        add_component!(sys, shunt)
        g = _add_simple_thermal_standard!(sys, b, 0.3, 0.07)
        set_available!(g, false)
        push!(gens, g)
        push!(shunts, shunt)
        _add_simple_load!(sys, b, 20.0, 10.0)
        _add_simple_line!(sys, prev, b, 0.01, 0.1, 0.0)
        prev = b
    end
    set_units_base_system!(sys, "SYSTEM_BASE")
    return sys, gens, shunts
end

@testset "Write-back: a bus reported PV with no available source" begin
    @testset "the PV branch skips it, with one warning for all of them" begin
        sys, gens, shunts = _pv_without_source_system()
        df = solve_power_flow(ACPowerFlow(), sys)["bus_results"]
        q_mvar = sum(abs, df[in.(df.bus_number, Ref((2, 3))), :Q_gen])
        @test q_mvar > 1.0
        logs, converged = Test.collect_test_logs(; min_level = Logging.Warn) do
            solve_and_store_power_flow!(ACPowerFlow(), sys)
        end
        @test converged
        skipped = filter(l -> occursin("reported PV with no available source", l.message),
            logs)
        @test length(skipped) == 1
        @test occursin("at 2 bus(es)", only(skipped).message)
        # The warning gives the solved Q left unwritten, in Mvar.
        m = match(r"solved Q, ([0-9.]+) Mvar in magnitude", only(skipped).message)
        @test !isnothing(m)
        @test parse(Float64, m[1]) ≈ q_mvar atol = 1e-3
        # Nothing is written to the out-of-service generators or the shunts.
        @test all(get_reactive_power(g) == 0.07 for g in gens)
        @test all(get_Y(s) == 0.05im for s in shunts)
        # The solved angles are still written; the PV buses keep their setpoints.
        for n in (2, 3)
            bus = get_component(ACBus, sys, "bus_$n")
            @test get_bustype(bus) == ACBusTypes.PV
            @test get_magnitude(bus) == 1.01
            @test get_angle(bus) != 0.0
        end
    end

    @testset "the redistribution warns and returns where it has no device" begin
        sys, _, _ = _pv_without_source_system()
        bus = get_component(ACBus, sys, "bus_2")
        @test_logs (:warn, r"No available source at bus bus_2") match_mode = :any begin
            @test isnothing(PF._reactive_power_redistribution_pv(sys, 0.1, bus, 10))
        end
    end
end
