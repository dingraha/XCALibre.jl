using XCALibre, Test, StaticArrays, LinearAlgebra
module CircularSimpleSetup
using XCALibre
include(pkgdir(XCALibre,"examples/interface_verification/circular_setup.jl"))
end
const CSS=CircularSimpleSetup

@testset "Circular SIMPLE constant flow and validation" begin
    case=CSS.circular_couette_case();m=case.state.geometry;n=length(case.state.velocity);nb=length(m.velocity.exterior_faces)
    constant=SVector(0.1,-0.2,0.)
    s=CircularSimpleState(m,fill(constant,n),zeros(n),fill(constant,nb),zeros(nb),1.;reference=Dict(1=>0.))
    result=circular_simple!(s;iterations=5)
    @test result.converged
    @test s.velocity ≈ fill(constant,n) atol=1e-12
    @test maximum(abs,s.pressure)<1e-12
    @test result.history[end].momentum<1e-12
    @test result.history[end].continuity<1e-12
    @test any(>(0),s.overlaps) && any(<(0),s.overlaps)
    # Mixed prescribed inflow/wall and no-backflow pressure outlet, exact constant
    # flow. Choose only strictly outgoing exterior faces as velocity Neumann.
    mesh=case.mesh;c=case.coupling;r=m.pressure.reconstruction
    outgoing=[dot(constant,r.metrics.face_area_vector[j])>0 for j in r.exterior_faces]
    hp=construct_circular_pressure_geometry(mesh,c,[out ? :dirichlet : :neumann for out in outgoing])
    mixed=construct_circular_momentum_geometry(hp,[out ? :neumann : :dirichlet for out in outgoing])
    vv=[out ? SVector(0.,0.,0.) : constant for out in outgoing]
    outlet=CircularSimpleState(mixed,fill(constant,n),zeros(n),vv,zeros(nb),1.)
    @test circular_simple!(outlet;iterations=5).converged
    @test outlet.velocity ≈ fill(constant,n) atol=1e-12
    @test maximum(abs,outlet.pressure)<1e-12
    @test_throws ArgumentError CircularSimpleState(mixed,fill(-constant,n),zeros(n),vv,zeros(nb),1.)
    # Boundary configuration snapshots are independent, with strict pairing.
    m.pressure.boundary_kind[1]=:dirichlet
    @test_throws ArgumentError CircularSimpleState(m,fill(constant,n),zeros(n),fill(constant,nb),zeros(nb),1.)
    m.pressure.boundary_kind[1]=:neumann
    for kw in ((;relaxation_U=0.),(;relaxation_p=1.1),(;momentum_tolerance=NaN),
               (;continuity_tolerance=0.),(;pressure_sweeps=0),(;reference=Dict(1=>NaN)))
        @test_throws ArgumentError CSS.circular_couette_case(;kw...)
    end
    @test_throws ArgumentError CircularSimpleState(m,fill(constant,n),zeros(n),fill(constant,nb),zeros(nb),0.;reference=Dict(1=>0.))
    @test_throws ArgumentError CircularSimpleState(m,fill(constant,n),zeros(n),fill(constant,nb),zeros(nb),1.)
    unbalanced=fill(constant,nb);j=m.velocity.exterior_faces[1]
    unbalanced[1]+=normalize(m.velocity.reconstruction.metrics.face_area_vector[j])
    @test_throws ArgumentError CircularSimpleState(m,fill(constant,n),zeros(n),unbalanced,zeros(nb),1.;reference=Dict(1=>0.))
    @test_throws DimensionMismatch CircularSimpleState(m,fill(constant,n),zeros(n),fill(constant,nb-1),zeros(nb),1.;reference=Dict(1=>0.))
end

@testset "Circular Couette coupling, gauge and relaxation" begin
    baseline=CSS.circular_couette_case()
    result=circular_simple!(baseline.state;iterations=500)
    @test result.converged
    @test result.history[end].momentum<1e-9
    @test result.history[end].pressure_residual<1e-11
    @test result.history[end].continuity<1e-10
    @test CSS.circular_couette_errors(baseline).velocity<0.08
    @test all(iszero,getindex.(baseline.state.velocity,3))
    @test abs(sum(baseline.state.overlaps))<1e-12
    # Small nonzero radial overlap flows reflect the nonconforming discretisation;
    # net sums must not replace the retained individual overlaps.
    @test length(baseline.state.overlaps)==length(baseline.coupling.measure)
    for (u,p) in ((0.5,0.2),(0.8,0.5))
        c=CSS.circular_couette_case(relaxation_U=u,relaxation_p=p)
        @test circular_simple!(c.state;iterations=500).converged
        @test c.state.velocity ≈ baseline.state.velocity atol=1e-7 rtol=1e-7
        @test c.state.pressure ≈ baseline.state.pressure atol=1e-7 rtol=1e-7
        @test c.state.faces ≈ baseline.state.faces atol=1e-7 rtol=1e-7
        @test c.state.overlaps ≈ baseline.state.overlaps atol=1e-7 rtol=1e-7
    end
    shifted=CSS.circular_couette_case(initial_pressure=3.,reference_pressure=3.)
    @test circular_simple!(shifted.state;iterations=500).converged
    @test shifted.state.velocity ≈ baseline.state.velocity atol=1e-7
    @test shifted.state.pressure ≈ baseline.state.pressure .+ 3 atol=1e-7
    @test shifted.state.overlaps ≈ baseline.state.overlaps atol=1e-7
    # External pressure not initially at the reference is uniformly gauge shifted.
    other=CSS.circular_couette_case(initial_pressure=1.)
    @test circular_simple!(other.state;iterations=500).converged
    @test other.state.velocity ≈ baseline.state.velocity atol=1e-7
    @test other.state.pressure[1]==0
end

@testset "Circular continuation, failure invalidation and recovery" begin
    a=CSS.circular_couette_case();b=CSS.circular_couette_case()
    @test !circular_simple!(a.state;iterations=12).converged
    circular_simple!(b.state;iterations=5);r=circular_simple!(b.state;iterations=7)
    @test r.total_iterations==12
    @test a.state.velocity==b.state.velocity
    @test a.state.pressure==b.state.pressure
    @test a.state.faces==b.state.faces
    @test a.state.overlaps==b.state.overlaps
    @test_throws ArgumentError circular_simple!(b.state;iterations=0)
    @test !b.state.valid
    # Compare retry against a fresh state starting from current fields, not stale q.
    function fresh(s)
        cfg=s.config
        CircularSimpleState(s.geometry,s.velocity,s.pressure,cfg.velocity_values,cfg.pressure_values,cfg.viscosity;
            reference=cfg.reference,relaxation_U=cfg.relaxation_U,relaxation_p=cfg.relaxation_p)
    end
    retry=fresh(b.state)
    circular_simple!(b.state;iterations=3);circular_simple!(retry;iterations=3)
    @test b.state.velocity==retry.velocity
    @test b.state.overlaps==retry.overlaps
    @test b.state.iteration==3
    # Changed fields reset the conservative flux snapshot.
    b.state.velocity[1]+=SVector(0.01,0.,0.)
    changed=fresh(b.state)
    circular_simple!(b.state;iterations=2);circular_simple!(changed;iterations=2)
    @test b.state.velocity==changed.velocity
    @test b.state.faces==changed.faces
    @test b.state.iteration==2
    saved=copy(b.state.velocity);b.state.velocity[1]=SVector(0.,0.,1.)
    @test_throws ArgumentError circular_simple!(b.state;iterations=1)
    @test !b.state.valid
    copyto!(b.state.velocity,saved)
    recovered=fresh(b.state)
    circular_simple!(b.state;iterations=2);circular_simple!(recovered;iterations=2)
    @test b.state.velocity==recovered.velocity
    # A frozen pressure solve is not adequate: force inner sweep exhaustion.
    failed=CSS.circular_couette_case(pressure_sweeps=1,pressure_tolerance=1e-18)
    oldU=copy(failed.state.velocity);oldp=copy(failed.state.pressure)
    @test_throws ErrorException circular_simple!(failed.state;iterations=1)
    @test !failed.state.valid
    @test failed.state.velocity==oldU && failed.state.pressure==oldp
    @test failed.state.iteration==0
end

@testset "Fixed-domain Couette refinement and frozen alignment" begin
    errors=NamedTuple[]
    for k in (1,2,4)
        c=CSS.circular_couette_case(na=12k,nb=15k,nra=2k,nrb=2k,phase=0.13/k)
        r=circular_simple!(c.state;iterations=500)
        @test r.converged
        @test r.history[end].continuity<1e-10
        push!(errors,CSS.circular_couette_errors(c))
    end
    @info "Static coupled Couette errors (pressure wall BC mismatch; no pressure order claim)" errors
    @test all(diff(getproperty.(errors,:velocity)).<0)
    @test errors[end].velocity<errors[1].velocity/8
    # Homogeneous p-Neumann omits physical radial dp/dr at the exterior walls;
    # report pressure errors rather than imposing a spurious convergence assertion.
    @test all(e -> isfinite(e.pressure) && e.pressure<0.003,errors)
    aligned=CSS.circular_couette_case(phase=0.)
    shifted=CSS.circular_couette_case(phase=0.71)
    @test circular_simple!(aligned.state;iterations=500).converged
    @test circular_simple!(shifted.state;iterations=500).converged
    @test abs(CSS.circular_couette_errors(aligned).velocity-CSS.circular_couette_errors(shifted).velocity)<0.01
end
