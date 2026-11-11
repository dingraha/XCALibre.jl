using XCALibre
using Test

function run_planar_example(file)
    m = Module(Symbol("PlanarExample_",splitext(file)[1]))
    Core.eval(m, :(include(path) = Base.include(@__MODULE__,path)))
    Base.include(m,pkgdir(XCALibre,"examples","interface_verification",file))
    Base.invokelatest(getproperty,m,:results)
end

@testset "Shipped planar channel example" begin
    result = run_planar_example("planar_channel.jl")
    metrics = result.metrics
    @test result.continued.converged
    @test result.state.work.iteration[] > result.first.iterations
    @test metrics.continuity < 1e-9
    @test abs(metrics.interface_balance) < 1e-12
    @test abs(metrics.boundary_balance) < 1e-10
    @test metrics.unequal_distances
    @test metrics.interface_flow ≈ 1 atol=0.02
    @test metrics.velocity_error < 0.03
    @test metrics.pressure_error < 0.05
    @test metrics.sections[1] < metrics.sections[2]
    @test metrics.pressure_drop > 0
end

@testset "Shipped planar channel refinement study" begin
    result = run_planar_example("channel_refinement.jl")
    a,b = result.nonconforming,result.conforming
    for cases in (a,b)
        errors = [x.metrics.velocity_error for x in cases]
        @test all(diff(errors) .< 0)
        @test errors[end] < errors[1]/5
        for x in cases
            @test x.metrics.continuity < 1e-9
            @test abs(x.metrics.boundary_balance) < 1e-10
            @test x.metrics.pressure_error < 0.06
        end
    end
    for x in a
        @test abs(x.metrics.interface_balance) < 1e-12
        @test x.metrics.unequal_distances
    end
    @test abs(a[end].metrics.velocity_error-b[end].metrics.velocity_error) < 0.004
    @test abs(a[end].metrics.pressure_error-b[end].metrics.pressure_error) < 0.01
    # Pressure-drop errors are nonmonotonic on coarse grids: report orders,
    # but deliberately make no positivity or second-order assertion.
    @test all(isfinite,result.pressure_orders)
    @test result.strong_mismatch.metrics.velocity_error < 0.06
    @test result.strong_mismatch.metrics.pressure_error < 0.08
    @test result.strong_mismatch.metrics.continuity < 1e-9
    @test abs(result.strong_mismatch.metrics.interface_balance) < 1e-12
end
