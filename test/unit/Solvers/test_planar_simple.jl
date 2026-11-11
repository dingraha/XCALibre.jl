using XCALibre
using Test
using StaticArrays
using LinearAlgebra
using Accessors
using Statistics

function planar_simple_case(; na=4,nb=5,split=true,channel=false,iterations=500,convergence=1e-8)
    P,line = XCALibre.Point,XCALibre.line!
    if split
        pts = [P(x,y,0.0) for (x,y) in ((0.,0.),(1.,0.),(0.,1.),(1.,1.),(1.,0.),(2.,0.),(1.,1.),(2.,1.))]
        edges = [line(pts,1,2,3),line(pts,3,4,3),line(pts,1,3,na),line(pts,2,4,na),
            line(pts,5,6,3),line(pts,7,8,3),line(pts,5,7,nb),line(pts,6,8,nb)]
        patches = [Patch(:interface_a,[4]),Patch(:interface_b,[7]),Patch(:inlet,[3]),Patch(:outlet,[8]),Patch(:walls,[1,2,5,6])]
        blocks = [quad(edges,[1,2,3,4]),quad(edges,[5,6,7,8])]
    else
        pts = [P(x,y,0.0) for (x,y) in ((0.,0.),(2.,0.),(0.,1.),(2.,1.))]
        edges = [line(pts,1,2,6),line(pts,3,4,6),line(pts,1,3,nb),line(pts,2,4,nb)]
        patches = [Patch(:inlet,[3]),Patch(:outlet,[4]),Patch(:walls,[1,2])]
        blocks = [quad(edges,[1,2,3,4])]
    end
    mesh = UNV2.update_mesh_format(generate!(MeshBuilder2D(pts,edges,patches,blocks)),Int64,Float64)
    model = Physics(time=Steady(),fluid=Fluid{Incompressible}(nu=0.1),turbulence=RANS{Laminar}(),energy=Energy{Isothermal}(),domain=mesh)
    velocity(x,t,i) = SVector(channel ? 6*x[2]*(1-x[2]) : 1.0,0.0,0.0)
    if split
        pair = construct_planar_transport(mesh,:interface_a,:interface_b)
        pressure = construct_planar_pressure(pair)
    else
        pair,pressure = (),()
    end
    BCs = assign((U=[pair...,DirichletFunction(:inlet,velocity),Zerogradient(:outlet),
        channel ? Dirichlet(:walls,[0.,0.,0.]) : Zerogradient(:walls)],
        p=[pressure...,Zerogradient(:inlet),Dirichlet(:outlet,0.0),Zerogradient(:walls)]);region=mesh)
    setup(relax) = SolverSetup(solver=Bicgstab(),preconditioner=Jacobi(),relax=relax,convergence=convergence,rtol=1e-11,atol=1e-12)
    cfg = Configuration(solvers=(U=setup(0.7),p=setup(0.3)),schemes=(U=Schemes(divergence=Upwind),p=Schemes()),boundaries=BCs,
        runtime=Runtime(iterations=iterations,write_interval=-1,time_step=1),hardware=Hardware(backend=CPU(),workgroup=16))
    initialise!(model.momentum.U,[0.7,0.,0.]); initialise!(model.momentum.p,0.0)
    model,cfg
end

@testset "Static planar SIMPLE integration" begin
    model,cfg = planar_simple_case()
    @test_throws ArgumentError run!(model,cfg;progress=false)
    state = PlanarSimpleState(model,cfg;continuity=1e-9)
    result = planar_simple!(state)
    @test result.converged
    @test maximum(abs,model.momentum.U.x.values .- 1) < 1e-6
    @test maximum(abs,model.momentum.U.y.values) < 1e-6
    @test maximum(abs,model.momentum.p.values) < 1e-6
    @test result.continuity[end] < 1e-9
    @test result.p[end] <= cfg.solvers.p.convergence || result.p_absolute[end] <= state.work.continuity
    w = state.work
    c = w.transport.coupling
    @test sum(w.transport.values[model.domain.boundaries[c.patch_a].IDs_range]) ≈ -sum(w.transport.values[model.domain.boundaries[c.patch_b].IDs_range]) atol=1e-12
    @test model.momentum.mdotf.values == w.transport.values
    conforming,ccfg = planar_simple_case(split=false)
    run!(conforming,ccfg;progress=false,transpose_stress=false)
    @test maximum(abs,conforming.momentum.U.x.values .- 1) < 1e-6
    @test mean(model.momentum.U.x.values) ≈ mean(conforming.momentum.U.x.values) atol=1e-6

    # Preserve overlap state across chunks; disable early stopping for exact comparison.
    m1,c1 = planar_simple_case(convergence=0.0)
    s1 = PlanarSimpleState(m1,c1)
    r1 = planar_simple!(s1;iterations=12)
    m2,c2 = planar_simple_case(convergence=0.0)
    s2 = PlanarSimpleState(m2,c2)
    planar_simple!(s2;iterations=6)
    r2 = planar_simple!(s2;iterations=6)
    @test m1.momentum.U.x.values == m2.momentum.U.x.values
    @test m1.momentum.p.values == m2.momentum.p.values
    @test s1.work.transport.overlaps == s2.work.transport.overlaps
    @test r1.Ux[7:12] == r2.Ux
    @test r1.p[7:12] == r2.p
    @test r1.continuity[7:12] == r2.continuity
    @test r1.p_absolute[7:12] == r2.p_absolute
    @test s1.work.coefficient.conductance == s2.work.coefficient.conductance
    initialise!(m2.momentum.U,[0.7,0.,0.]); initialise!(m2.momentum.p,0.0)
    restarted = planar_simple!(s2;iterations=12)
    @test s2.work.iteration[] == 12
    @test m1.momentum.U.x.values == m2.momentum.U.x.values
    @test r1.Ux == restarted.Ux
    @test_throws ArgumentError planar_simple!(s2;iterations=0)
    @test_throws ArgumentError PlanarSimpleState(model,cfg;pref=0.0)
    @test_throws ArgumentError PlanarSimpleState(model,cfg;continuity=-1)
    bad = cfg
    @reset bad.schemes.U = Schemes(divergence=Linear)
    @test_throws ArgumentError PlanarSimpleState(model,bad)
    bad = cfg
    @reset bad.runtime = Runtime(iterations=2,write_interval=1,time_step=1)
    @test_throws ArgumentError PlanarSimpleState(model,bad)
    @test_throws ArgumentError PlanarSimpleState(conforming,ccfg)
    bad = cfg
    @reset bad.schemes.p = Schemes(gradient=LeastSquares)
    @test_throws ArgumentError PlanarSimpleState(model,bad)
    bad = cfg
    @reset bad.schemes.U = Schemes(limiter=1.0)
    @test_throws ArgumentError PlanarSimpleState(model,bad)
    initialise!(model.momentum.U,[1.,0.,1.])
    @test_throws ArgumentError PlanarSimpleState(model,cfg)
    initialise!(model.momentum.U,[1.,0.,0.])
    badmodel,badcfg = planar_simple_case()
    @reset badmodel.fluid = Fluid{Incompressible}(nu=0.1,rho=2.0)(badmodel.domain)
    @test_throws ArgumentError PlanarSimpleState(badmodel,badcfg)
end

@testset "Planar SIMPLE Neumann compatibility" begin
    model,cfg = planar_simple_case(convergence=0.0)
    pair = construct_planar_transport(model.domain,:interface_a,:interface_b)
    pressure = construct_planar_pressure(pair)
    BCs = assign((U=[pair...,Dirichlet(:inlet,[1.,0.,0.]),Dirichlet(:outlet,[1.,0.,0.]),Dirichlet(:walls,[1.,0.,0.])],
        p=[pressure...,Zerogradient(:inlet),Zerogradient(:outlet),Zerogradient(:walls)]);region=model.domain)
    @reset cfg.boundaries = BCs
    @test_throws ArgumentError PlanarSimpleState(model,cfg)
    state = PlanarSimpleState(model,cfg;pref=0.0)
    result = planar_simple!(state;iterations=100)
    @test result.continuity[end] < 1e-9
    @test maximum(abs,model.momentum.U.x.values .- 1) < 1e-6
    # Incompatible prescribed exterior flow must fail rather than hide the error at reference.
    BCs = assign((U=[pair...,Dirichlet(:inlet,[1.,0.,0.]),Dirichlet(:outlet,[2.,0.,0.]),Dirichlet(:walls,[1.,0.,0.])],
        p=[pressure...,Zerogradient(:inlet),Zerogradient(:outlet),Zerogradient(:walls)]);region=model.domain)
    @reset cfg.boundaries = BCs
    invalid = PlanarSimpleState(model,cfg;pref=0.0)
    @test_throws ArgumentError planar_simple!(invalid;iterations=1)
    @test !invalid.work.valid[]
end

@testset "Planar SIMPLE viscous channel" begin
    model,cfg = planar_simple_case(na=8,nb=9,channel=true,iterations=800,convergence=1e-7)
    state = PlanarSimpleState(model,cfg;continuity=1e-8)
    result = planar_simple!(state)
    @test result.converged
    @test result.continuity[end] < 1e-8
    U,p,mesh = model.momentum.U,model.momentum.p,model.domain
    exact = [6*x[2]*(1-x[2]) for x in mesh.cell_centre]
    @test norm(U.x.values-exact)/norm(exact) < 0.12
    @test maximum(abs,U.y.values) < 0.1
    @test maximum(p.values)-minimum(p.values) > 0.5
    conforming,ccfg = planar_simple_case(split=false,nb=9,channel=true,iterations=800,convergence=1e-7)
    run!(conforming,ccfg;progress=false,transpose_stress=false)
    inlet = only(filter(bc -> bc.ID == findfirst(b -> b.name == :inlet,mesh.boundaries),cfg.boundaries.U))
    @test maximum(abs,model.momentum.U.x.values) > minimum(model.momentum.U.x.values)+0.5
    @test sum(state.work.transport.values[only(filter(bc -> bc isa Zerogradient && bc.ID != inlet.ID,cfg.boundaries.U)).IDs_range]) ≈ 1.0 atol=0.02
    XCALibre.Calculate.div!(state.work.divergence,FaceScalarField(state.work.transport.values,mesh),cfg)
    @test maximum(abs,state.work.divergence.values .* mesh.cell_volume) < 1e-8
    correction_from_velocity = similar(state.work.transport.overlaps)
    interface_predictor_flux!(correction_from_velocity,state.work.pair[1],U)
    @test norm(correction_from_velocity-state.work.transport.overlaps) > 1e-8
    @test sum(state.work.transport.values[inlet.IDs_range]) ≈ -1.0 atol=0.02
    @test maximum(p.values) ≈ maximum(conforming.momentum.p.values) rtol=0.25
end
