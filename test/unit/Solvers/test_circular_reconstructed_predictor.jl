using XCALibre, Test, StaticArrays, LinearAlgebra
module CircularPredictorSetup
using XCALibre
include(pkgdir(XCALibre,"examples/interface_verification/circular_setup.jl"))
end
const CPS=CircularPredictorSetup

@testset "Shared reconstructed physical predictor" begin
    case=CPS.circular_couette_case();m=case.state.geometry;h=m.velocity;r=h.reconstruction
    origin=r.metrics.coupling.origin
    G=SMatrix{3,3}(0.2,-0.4,0.,0.3,-0.1,0.,0.,0.,0.)
    u0=SVector(0.13,-0.17,0.)
    velocity(x)=u0+G*(x-origin)
    U=velocity.(r.points);values=velocity.(r.exterior_points)
    # Arbitrary varying pressure contribution must retain precisely the old
    # geometric interpolation, not be reconstructed or multiplied by rD twice.
    pressure_velocity=[SVector(0.01i,-0.003i,0.) for i in eachindex(U)]
    H=U+pressure_velocity;oldH=copy(H);oldP=copy(pressure_velocity);oldV=copy(values)
    result=circular_momentum_reconstructed_predictor_flux(m,H,pressure_velocity,values)
    for e in eachindex(h.edge_a)
        a,b,w,x=h.edge_a[e],h.edge_b[e],h.weight_a[e],h.edge_point[e]
        contribution=w*pressure_velocity[a]+(1-w)*pressure_velocity[b]
        j=h.edge_face[e]
        if iszero(j)
            k=h.edge_overlap[e];c=r.metrics.coupling
            # Integral of (u0+G*(x-origin)) dot outward normal.
            expected=dot(u0,c.area_vector[k])+sum(G.*transpose(c.area_moment[k]-origin*c.area_vector[k]'))+dot(contribution,c.area_vector[k])
            @test result.overlaps[k] ≈ expected atol=1e-12
        else
            @test result.faces[j] ≈ dot(velocity(x)+contribution,h.edge_area[e]) atol=1e-12
        end
    end
    @test H==oldH && pressure_velocity==oldP && values==oldV
    @test result.faces[h.exterior_faces] ≈ [dot(values[k],r.metrics.face_area_vector[j]) for (k,j) in enumerate(h.exterior_faces)] atol=1e-14
    balance=zeros(length(U));interface_cell_flux!(balance,r.metrics.coupling,result.overlaps)
    @test abs(sum(balance))<1e-13
    diagnostic=copy(result.faces);interface_face_flux!(diagnostic,r.metrics.coupling,result.overlaps)
    @test diagnostic==result.faces
    constant=fill(u0,length(U));constantvalues=fill(u0,length(values))
    reconstructed=circular_momentum_reconstructed_predictor_flux(m,constant,fill(zero(u0),length(U)),constantvalues)
    low=circular_momentum_predictor_flux(m,constant,constantvalues)
    @test reconstructed.faces ≈ low.faces atol=1e-14
    @test reconstructed.overlaps ≈ low.overlaps atol=1e-14
    @test_throws DimensionMismatch circular_momentum_reconstructed_predictor_flux(m,H,pressure_velocity[1:end-1],values)
    bad=copy(pressure_velocity);bad[1]=SVector(0.,0.,1.)
    @test_throws ArgumentError circular_momentum_reconstructed_predictor_flux(m,H,bad,values)
    bad[1]=SVector(NaN,0.,0.)
    @test_throws ArgumentError circular_momentum_reconstructed_predictor_flux(m,H,bad,values)
    @test_throws ArgumentError CPS.circular_couette_case(predictor=:unsupported)
    # Exact solid-body rotation has pointwise zero circular radial flux. Its
    # integrated affine trace must preserve that on nonmatching segmentation.
    rotation(x)=SVector(-x[2],x[1],0.)
    spin=circular_momentum_reconstructed_predictor_flux(m,rotation.(r.points),fill(zero(u0),length(U)),rotation.(r.exterior_points))
    @test maximum(abs,spin.overlaps)<1e-13
    # Mixed outlet reproduces constant flow and retains strict backflow checks.
    outgoing=[dot(u0,r.metrics.face_area_vector[j])>0 for j in h.exterior_faces]
    hp=construct_circular_pressure_geometry(case.mesh,case.coupling,[out ? :dirichlet : :neumann for out in outgoing])
    mixed=construct_circular_momentum_geometry(hp,[out ? :neumann : :dirichlet for out in outgoing])
    vv=[out ? zero(u0) : u0 for out in outgoing]
    outlet=CircularSimpleState(mixed,constant,zeros(length(U)),vv,zeros(length(vv)),1.;predictor=:reconstructed)
    @test circular_simple!(outlet;iterations=5).converged
    @test outlet.velocity ≈ constant atol=1e-12
    @test_throws ArgumentError circular_momentum_reconstructed_predictor_flux(mixed,-constant,fill(zero(u0),length(U)),vv)
end

isdefined(@__MODULE__,:circular_interface_fixture) || include("../InterfaceCoupling/test_circular_interface.jl")
@testset "Reconstructed predictor storage types" begin
    for (T,I) in ((Float32,Int32),(Float64,Int32),(Float64,Int64))
        mesh=circular_interface_fixture(na=8,nb=11,phase=.13,float=T,integer=I)
        c=circular_coupling(mesh);r=construct_circular_reconstruction(mesh,c)
        h=construct_circular_pressure_geometry(mesh,c,fill(:dirichlet,length(r.exterior_faces)))
        m=construct_circular_momentum_geometry(h,fill(:dirichlet,length(r.exterior_faces)))
        U(x)=SVector{3,T}(.2+.3x[1]-.1x[2],-.1+.2x[1]+.4x[2],0)
        zero_pressure=fill(zero(SVector{3,T}),length(r.points))
        flux=circular_momentum_reconstructed_predictor_flux(m,U.(r.points),zero_pressure,U.(r.exterior_points))
        @test eltype(flux.faces)==T
        @test eltype(flux.overlaps)==T
        for k in eachindex(c.measure)
            x=c.midpoint[k];S=c.area_vector[k];M=c.area_moment[k]-x*S'
            expected=dot(U(x),S)+T(.3)*M[1,1]-T(.1)*M[2,1]+T(.2)*M[1,2]+T(.4)*M[2,2]
            @test flux.overlaps[k] ≈ expected atol=1000eps(T)
        end
    end
end

@testset "Reconstructed coupled relaxation, gauge and lifecycle" begin
    baseline=CPS.circular_couette_case(wall_pressure=:analytical,predictor=:reconstructed)
    @test circular_simple!(baseline.state;iterations=600).converged
    s=baseline.state;h=s.geometry.pressure
    @test s.faces[h.exterior_faces] ≈ [dot(v,h.reconstruction.metrics.face_area_vector[j]) for (v,j) in zip(s.config.velocity_values,h.exterior_faces)] atol=1e-14
    for (ru,rp) in ((.5,.2),(.8,.5))
        c=CPS.circular_couette_case(wall_pressure=:analytical,predictor=:reconstructed,relaxation_U=ru,relaxation_p=rp,initial_pressure=2.,reference_pressure=2.)
        @test circular_simple!(c.state;iterations=600).converged
        @test c.state.velocity ≈ s.velocity atol=1e-7 rtol=1e-7
        @test c.state.pressure ≈ s.pressure .+ 2 atol=1e-7 rtol=1e-7
        @test c.state.faces ≈ s.faces atol=1e-7 rtol=1e-7
        @test c.state.overlaps ≈ s.overlaps atol=1e-7 rtol=1e-7
    end
    a=CPS.circular_couette_case(predictor=:reconstructed);b=CPS.circular_couette_case(predictor=:reconstructed)
    circular_simple!(a.state;iterations=12)
    circular_simple!(b.state;iterations=5);circular_simple!(b.state;iterations=7)
    @test a.state.velocity==b.state.velocity
    @test a.state.pressure==b.state.pressure
    @test a.state.faces==b.state.faces
    @test a.state.overlaps==b.state.overlaps
    function fresh(s)
        c=s.config
        CircularSimpleState(s.geometry,s.velocity,s.pressure,c.velocity_values,c.pressure_values,c.viscosity;
            reference=c.reference,relaxation_U=c.relaxation_U,relaxation_p=c.relaxation_p,predictor=c.predictor)
    end
    @test_throws ArgumentError circular_simple!(b.state;iterations=0)
    @test !b.state.valid
    retry=fresh(b.state);circular_simple!(retry;iterations=2);circular_simple!(b.state;iterations=2)
    @test b.state.velocity==retry.velocity
    @test b.state.overlaps==retry.overlaps
    b.state.velocity[1]+=SVector(.01,0.,0.)
    changed=fresh(b.state);circular_simple!(changed;iterations=2);circular_simple!(b.state;iterations=2)
    @test b.state.velocity==changed.velocity
    @test b.state.faces==changed.faces
    @test b.state.iteration==2
    failed=CPS.circular_couette_case(predictor=:reconstructed,pressure_sweeps=1,pressure_tolerance=1e-18)
    oldU=copy(failed.state.velocity);oldp=copy(failed.state.pressure)
    @test_throws ErrorException circular_simple!(failed.state;iterations=1)
    @test !failed.state.valid
    @test failed.state.velocity==oldU && failed.state.pressure==oldp
    @test failed.state.iteration==0
    for phase in (0.,.71)
        c=CPS.circular_couette_case(phase=phase,predictor=:reconstructed,wall_pressure=:analytical)
        @test circular_simple!(c.state;iterations=600).converged
        error=CPS.circular_couette_errors(c)
        @test all(isfinite,values(error))
        @info "Reconstructed predictor phase sensitivity (no phase-independent accuracy claim)" phase error
    end
end

@testset "Coupled predictor pressure refinement" begin
    errors=NamedTuple[]
    for k in (1,2,4)
        pair=NamedTuple[]
        for predictor in (:low_order,:reconstructed)
            c=CPS.circular_couette_case(na=12k,nb=15k,nra=2k,nrb=2k,phase=.13/k,wall_pressure=:analytical,predictor=predictor)
            result=circular_simple!(c.state;iterations=600)
            @test result.converged
            @test result.history[end].momentum<1e-9
            @test result.history[end].continuity<1e-10
            @test result.history[end].pressure_residual<1e-11
            push!(pair,CPS.circular_couette_errors(c))
        end
        @test pair[2].pressure<pair[1].pressure
        @test pair[2].pressure_gradient<pair[1].pressure_gradient
        @test abs(pair[2].velocity-pair[1].velocity)<0.0001
        push!(errors,(;k,low_order=pair[1],reconstructed=pair[2]))
    end
    @test errors[end].reconstructed.pressure<errors[1].reconstructed.pressure/5
    @test errors[end].reconstructed.pressure<errors[end].low_order.pressure/8
    @info "Reconstructed predictor improves coupled pressure; max gradient error is not monotonic" errors
end
