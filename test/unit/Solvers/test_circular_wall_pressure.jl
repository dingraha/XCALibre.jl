using XCALibre, Test, StaticArrays, LinearAlgebra
module CircularWallSetup
using XCALibre
include(pkgdir(XCALibre,"examples/interface_verification/circular_setup.jl"))
end
const CWS=CircularWallSetup

@testset "Prescribed normal pressure derivatives" begin
    c=CWS.circular_couette_case(wall_pressure=:analytical)
    h=c.state.geometry.pressure;r=h.reconstruction;n=length(r.points)
    g=SVector(0.31,-0.17,0.);p=[1.2+dot(g,x) for x in r.points]
    values=[dot(g,normalize(r.metrics.face_area_vector[j])) for j in h.exterior_faces]
    gradients=fill(zero(g),n)
    circular_pressure_reconstruct!(gradients,h,p,values)
    @test gradients ≈ fill(g,n) atol=1e-12
    circular_pressure_gradient!(gradients,h,p,values)
    @test gradients ≈ fill(g,n) atol=1e-12
    mobility=[0.5+0.01i for i in 1:n]
    terms=circular_pressure_terms(h,fill(g,n),mobility,values)
    flux=circular_pressure_flux(terms,p)
    @test terms.matrix*p+terms.offset ≈ flux.balance atol=1e-12
    @test all(iszero,terms.boundary_conductance)
    for (k,j) in enumerate(h.exterior_faces)
        a=r.metrics.owner_a[j]
        @test flux.faces[j] ≈ -mobility[a]*norm(r.metrics.face_area_vector[j])*values[k] atol=1e-14
    end
    # Predictor alone is incompatible; predictor + prescribed derivative flux
    # is exactly compatible. Retain saved predictor for every deferred sweep.
    pred=-flux.faces;overlaps=-flux.overlaps
    @test abs(sum(XCALibre.InterfaceCoupling._pressure_balance(h,pred,overlaps)))>1e-5
    projected=circular_pressure_project(h,mobility,values,pred,overlaps;
        initial=p,reference=Dict(1=>p[1]),tolerance=1e-11)
    @test projected.converged
    @test projected.pressure ≈ p atol=1e-10
    @test maximum(abs,projected.faces)<1e-11
    @test maximum(abs,projected.overlaps)<1e-11
    bad=copy(pred);bad[first(h.exterior_faces)]+=0.1
    @test_throws ArgumentError circular_pressure_project(h,mobility,values,bad,overlaps;reference=Dict(1=>p[1]))
    homogeneous=construct_circular_pressure_geometry(c.mesh,c.coupling,fill(:neumann,length(values)))
    @test_throws ArgumentError circular_pressure_reconstruct!(gradients,homogeneous,p,values)
    invalid=copy(values);invalid[1]=NaN;old=copy(gradients)
    @test_throws ArgumentError circular_pressure_reconstruct!(gradients,h,p,invalid)
    @test gradients==old
    # Mixed affine Dirichlet/gradient data, including Neumann matrix nullspace
    # removal by a real Dirichlet anchor rather than a pressure reference.
    kinds=fill(:normal_gradient,length(values));kinds[1]=:dirichlet
    mixed=construct_circular_pressure_geometry(c.mesh,c.coupling,kinds)
    mixedvalues=copy(values);mixedvalues[1]=1.2+dot(g,r.exterior_points[1])
    circular_pressure_gradient!(gradients,mixed,p,mixedvalues)
    @test gradients ≈ fill(g,n) atol=1e-12
    mt=circular_pressure_terms(mixed,fill(g,n),mobility,mixedvalues)
    mf=circular_pressure_flux(mt,p)
    mp=circular_pressure_project(mixed,mobility,mixedvalues,-mf.faces,-mf.overlaps;initial=p)
    @test mp.converged
    @test mp.pressure ≈ p atol=1e-10
end

@testset "Analytical wall gradient coupled consistency" begin
    c=CWS.circular_couette_case(wall_pressure=:analytical)
    r=circular_simple!(c.state;iterations=500)
    @test r.converged
    h=c.state.geometry.pressure
    target=[dot(v,h.reconstruction.metrics.face_area_vector[j]) for (v,j) in zip(c.state.config.velocity_values,h.exterior_faces)]
    @test c.state.faces[h.exterior_faces] ≈ target atol=1e-14
    @test r.history[end].continuity<1e-10
    @test r.history[end].momentum<1e-9
    @test minimum(c.state.config.pressure_values)<0
    shifted=CWS.circular_couette_case(wall_pressure=:analytical,initial_pressure=2.,reference_pressure=2.,relaxation_U=.5,relaxation_p=.2)
    @test circular_simple!(shifted.state;iterations=500).converged
    @test shifted.state.velocity ≈ c.state.velocity atol=1e-7
    @test shifted.state.pressure ≈ c.state.pressure .+ 2 atol=1e-7
    @test shifted.state.overlaps ≈ c.state.overlaps atol=1e-7
    a=CWS.circular_couette_case(wall_pressure=:analytical);b=CWS.circular_couette_case(wall_pressure=:analytical)
    circular_simple!(a.state;iterations=10)
    circular_simple!(b.state;iterations=4);circular_simple!(b.state;iterations=6)
    @test a.state.velocity==b.state.velocity
    @test a.state.pressure==b.state.pressure
    @test a.state.faces==b.state.faces
    @test a.state.overlaps==b.state.overlaps
    @test_throws ArgumentError circular_simple!(b.state;iterations=0)
    @test !b.state.valid
    @test circular_simple!(b.state;iterations=500).converged
    @test b.state.pressure ≈ c.state.pressure atol=1e-7
end

@testset "Wall pressure refinement comparison" begin
    results=NamedTuple[]
    for k in (1,2,4)
        pair=NamedTuple[]
        for mode in (:homogeneous,:analytical)
            c=CWS.circular_couette_case(na=12k,nb=15k,nra=2k,nrb=2k,phase=.13/k,wall_pressure=mode)
            @test circular_simple!(c.state;iterations=600).converged
            push!(pair,CWS.circular_couette_errors(c))
        end
        @test pair[2].pressure<pair[1].pressure
        @test abs(pair[2].velocity-pair[1].velocity)<1e-5
        @test isfinite(pair[2].pressure_gradient)
        push!(results,(;k,homogeneous=pair[1],analytical=pair[2]))
    end
    @info "Prescribed-gradient comparison: modest pressure improvement, plateau persists; no pressure/gradient convergence claim" results
end
