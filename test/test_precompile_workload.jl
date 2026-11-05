# The precompile workload (src/precompile.jl): it swallows its own errors, so that a broken one
# cannot stop the package loading, and only a test can see one. Public data only.

@testset "precompile workload: runs, and its IEEE 14 is the published case" begin
    @test with_logger(NullLogger()) do
        PF._pc_workload()
    end === nothing
    # The 1962 case's published solution (test/test_data/14_bus.raw).
    sys = PF._pc_ieee14()
    data = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(; check_reactive_power_limits = true), sys)
    @test with_logger(NullLogger()) do
        solve_power_flow!(data)
    end
    bl = PF.get_bus_lookup(data)
    for (n, vm, va) in ((4, 1.01767, -10.3128), (9, 1.05593, -14.9385), (14, 1.03553, -16.0336))
        @test data.bus_magnitude[bl[n], 1] ≈ vm atol = 1e-3
        @test rad2deg(data.bus_angles[bl[n], 1]) ≈ va atol = 0.05
    end
    # The remote-regulation synthetic enrols one group and holds the regulated bus at VS.
    data = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(; check_reactive_power_limits = true),
        PF._pc_remote_system())
    @test with_logger(NullLogger()) do
        solve_power_flow!(data)
    end
    @test PF.has_remote_voltage_groups(data.remote_voltage_control)
    @test data.bus_magnitude[PF.get_bus_lookup(data)[3], 1] ≈ 1.02 atol = 1e-8
end

@testset "solve kwargs: sorted by key, so a key set has one type" begin
    a = PF._sorted_kwargs(Dict{Symbol, Any}(:tol => 1e-8, :maxIterations => 30))
    b = PF._sorted_kwargs(Dict{Symbol, Any}(:maxIterations => 30, :tol => 1e-8))
    @test typeof(a) === typeof(b) === @NamedTuple{maxIterations::Int, tol::Float64}
    @test PF._sorted_kwargs(Dict{Symbol, Any}()) === (;)
end
