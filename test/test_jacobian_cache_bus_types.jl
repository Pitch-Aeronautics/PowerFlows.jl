# The Jacobian-structure and polar NR caches persist across solves on one PowerFlowData. A
# bus-type change between two solves (an island's bus turned REF, a new swing, a PV bus turned
# PQ) must not reuse a structure or a pivot sequence chosen under the old types.

const _CACHE_14BUS_RAW = joinpath(TEST_DATA_DIR, "14_bus.raw")

function _cache_test_system()
    return with_logger(SimpleLogger(stderr, Logging.Error)) do
        System(_CACHE_14BUS_RAW; runchecks = false)
    end
end

_quiet_solve!(data; kwargs...) = with_logger(SimpleLogger(stderr, Logging.Error)) do
    solve_power_flow!(data; kwargs...)
end

# Open branch 7-8 in the Y-bus in place (as a contingency runner does) and make bus 8, now an
# island, a REF row: its own subnetwork with its own swing.
function _open_7_8_island!(data, sys)
    br = only(
        b for b in get_components(ACBranch, sys) if Set(PNM.get_arc_tuple(b)) == Set((7, 8))
    )
    bl = PF.get_bus_lookup(data)
    (y11, y12, y21, y22) = PNM.ybus_branch_entries(br, PF.get_network_reduction_data(data))
    f, t = bl[PNM.get_arc_tuple(br)[1]], bl[PNM.get_arc_tuple(br)[2]]
    Y = data.power_network_matrix.data
    Y[f, f] -= y11
    Y[f, t] -= y12
    Y[t, f] -= y21
    Y[t, t] -= y22
    data.bus_type[bl[8], 1] = ACBusTypes.REF
    return
end

@testset "Jacobian caches: a new island swing between two solves on one PowerFlowData" begin
    sys = _cache_test_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    @test _quiet_solve!(data)
    nr_before = data.polar_nr_cache[]
    structure_before = data.ac_jacobian_structure_cache[]
    @test !isnothing(nr_before)
    _open_7_8_island!(data, sys)
    # Before GAP-043 this re-solve stalled (the base case's pivots and structure were reused).
    @test _quiet_solve!(data)
    # The structure was rebuilt for the new REF row and subnetwork.
    @test data.ac_jacobian_structure_cache[] !== structure_before
    # Same answer as a solve on the same data with the caches dropped.
    ref = PowerFlowData(ACPolarPowerFlow(), sys)
    @test _quiet_solve!(ref)
    _open_7_8_island!(ref, sys)
    ref.ac_jacobian_structure_cache[] = nothing
    ref.polar_nr_cache[] = nothing
    @test _quiet_solve!(ref)
    keep = [i for i in axes(data.bus_type, 1) if i != PF.get_bus_lookup(data)[8]]
    @test data.bus_magnitude[keep, 1] ≈ ref.bus_magnitude[keep, 1] atol = 1e-8
    @test data.bus_angles[keep, 1] ≈ ref.bus_angles[keep, 1] atol = 1e-8
end

@testset "Jacobian caches: a PV→PQ change repivots without a new symbolic analysis" begin
    # A generator trip: the PV bus turns PQ and loses its reactive injection.
    function trip!(data)
        pv = findfirst(==(ACBusTypes.PV), view(data.bus_type, :, 1))
        data.bus_type[pv, 1] = ACBusTypes.PQ
        data.bus_reactive_power_injections[pv, 1] = 0.0
        return
    end
    sys = _cache_test_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    @test _quiet_solve!(data)
    nr_before = data.polar_nr_cache[]
    structure_before = data.ac_jacobian_structure_cache[]
    trip!(data)
    @test _quiet_solve!(data)
    # The structure and the symbolic analysis are reused (a PV↔PQ flip leaves the sparsity
    # alone); the pivots were refreshed for the new types.
    @test data.ac_jacobian_structure_cache[] === structure_before
    @test data.polar_nr_cache[] === nr_before
    @test nr_before.bus_types == data.bus_type[:, 1]
    # Same answer as the same sequence with the caches dropped before the second solve.
    ref = PowerFlowData(ACPolarPowerFlow(), sys)
    @test _quiet_solve!(ref)
    trip!(ref)
    ref.ac_jacobian_structure_cache[] = nothing
    ref.polar_nr_cache[] = nothing
    @test _quiet_solve!(ref)
    @test data.bus_magnitude[:, 1] ≈ ref.bus_magnitude[:, 1] atol = 1e-8
    @test data.bus_angles[:, 1] ≈ ref.bus_angles[:, 1] atol = 1e-8
end

@testset "Jacobian caches: time steps with different bus types in one multiperiod solve" begin
    # c_sys14 with Q limits: every generator bus hits its limit at time step 1 (all PQ), not at
    # steps 2 and 3, so consecutive steps factor the Jacobian under different bus types.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    set_units_base_system!(sys, UnitSystem.SYSTEM_BASE)
    function run(clear::Bool)
        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 3,
            check_reactive_power_limits = true, correct_bustypes = true)
        data = PowerFlowData(pf, sys)
        prepare_ts_data!(data, 3)
        for ts in 1:3
            if clear
                data.polar_nr_cache[] = nothing
                data.ac_jacobian_structure_cache[] = nothing
            end
            @test _quiet_solve!(data; time_steps = [ts])
        end
        return data
    end
    data, ref = run(false), run(true)
    @test view(data.bus_type, :, 1) != view(data.bus_type, :, 2)
    @test data.bus_magnitude ≈ ref.bus_magnitude atol = 1e-8
    @test data.bus_angles ≈ ref.bus_angles atol = 1e-8
end
