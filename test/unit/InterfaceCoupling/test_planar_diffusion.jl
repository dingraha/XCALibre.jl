using XCALibre
using Test
using LinearAlgebra
using Accessors

# The geometry suite defines the shared tiny-mesh fixture when running the unit tier.
if !isdefined(@__MODULE__, :planar_interface_fixture)
    include("test_planar_interface.jl")
end

function planar_diffusion_config(mesh, BCs)
    Configuration(
        solvers=(T=SolverSetup(solver=Cg(),preconditioner=Jacobi(),relax=1.0,convergence=1e-10,rtol=1e-12,atol=1e-12),),
        schemes=(T=Schemes(laplacian=Linear),),
        runtime=Runtime(iterations=1,write_interval=-1,time_step=1),
        hardware=Hardware(backend=CPU(),workgroup=4),boundaries=BCs)
end

function planar_diffusion_equation(mesh, BCs, gamma=ConstantScalar(2.0))
    phi = ScalarField(mesh)
    cfg = planar_diffusion_config(mesh,BCs)
    eqn = (-Laplacian{Linear}(gamma,phi) == Source(ConstantScalar(0.0))) → ScalarEquation(phi,BCs.T)
    discretise!(eqn,phi,cfg)
    apply_boundary_conditions!(eqn,BCs.T,nothing,0.0,cfg)
    eqn,phi,cfg
end

@testset "Implicit static planar diffusion" begin
    for I in (Int64,Int32)
        mesh = planar_interface_fixture(integer=I)
        pair = construct_planar_diffusion(mesh,:interface_a,:interface_b)
        c = pair[1].value.coupling
        BCs = assign((T=[pair...,Zerogradient(:outside)],);region=mesh)
        eqn,phi,cfg = planar_diffusion_equation(mesh,BCs)
        A = Matrix(eqn.equation.A.parent)
        @test A ≈ transpose(A)
        @test A*ones(length(mesh.cells)) ≈ zeros(length(mesh.cells)) atol=1e-14
        @test eqn.equation.b == zeros(length(mesh.cells))
        @test eltype(eqn.equation.diag_nz) == I
        @test eltype(eqn.equation.A.parent.colval) == I
        G = 2 .* c.measure # owner normal separation is one
        for k in eachindex(G)
            @test A[c.cell_a[k],c.cell_b[k]] ≈ -G[k]
            @test A[c.cell_b[k],c.cell_a[k]] ≈ -G[k]
        end
        # Subtract the insulated-block operator to isolate interface contributions.
        insulated = assign((T=[Zerogradient(:interface_a),Zerogradient(:interface_b),
            Zerogradient(:outside)],);region=mesh)
        base,_,_ = planar_diffusion_equation(mesh,insulated)
        D = A-Matrix(base.equation.A.parent)
        @test D ≈ transpose(D)
        @test D*ones(length(mesh.cells)) ≈ zeros(length(mesh.cells)) atol=1e-14
        @test eigmin(Symmetric(D)) >= -1e-14
        phi.values .= [0.3,-0.7,1.1,-0.2,0.8]
        flux = zeros(length(c.measure))
        interface_diffusion_flux!(flux,pair[1],phi,ConstantScalar(2.0))
        reference = zeros(length(mesh.cells))
        interface_cell_flux!(reference,c,flux)
        @test D*phi.values ≈ reference
        @test abs(sum(reference)) < 1e-14
        @test interface_diffusion_flux!(similar(flux),pair[2],phi,ConstantScalar(2.0)) == flux
        initialise!(phi,3.0)
        @test interface_diffusion_flux!(flux,pair[1],phi,ConstantScalar(2.0)) == zeros(length(flux))
        @test_throws ArgumentError interface_diffusion_flux!(flux,pair[1],phi,ConstantScalar(-1.0))
        @test_throws DimensionMismatch interface_diffusion_flux!(zeros(1),pair[1],phi,ConstantScalar(2.0))
        # Reassembly must clear added sparse entries instead of accumulating old values.
        discretise!(eqn,phi,cfg)
        apply_boundary_conditions!(eqn,BCs.T,nothing,0.0,cfg)
        @test Matrix(eqn.equation.A.parent) == A
        # Normal-linear field must give the common interface trace on every face.
        phi.values .= [p[1] for p in mesh.cell_centre]
        phif = FaceScalarField(mesh)
        correct_boundaries!(phif,phi,BCs.T,0.0,cfg)
        @test phif.values[c.face_adjacency.IDs] ≈ ones(length(c.face_adjacency.IDs))
        interface_diffusion_flux!(flux,pair[1],phi,ConstantScalar(2.0))
        @test flux ≈ -2 .* c.measure
    end

    # A single A owner gathers several donors into the same matrix row.
    many = planar_interface_fixture(na=1,nb=3)
    many_pair = construct_planar_diffusion(many,:interface_a,:interface_b)
    many_bcs = assign((T=[many_pair...,Zerogradient(:outside)],);region=many)
    many_eqn,_,_ = planar_diffusion_equation(many,many_bcs)
    many_A = Matrix(many_eqn.equation.A.parent)
    @test many_A ≈ transpose(many_A)
    @test many_A*ones(length(many.cells)) ≈ zeros(length(many.cells)) atol=1e-14
    @test many_A[1,2:4] ≈ fill(-2/3,3)

    # Manufactured phi=x with exact exterior Dirichlet values. Existing boundary
    # function supplies the value at each physical face centre.
    mesh = planar_interface_fixture()
    pair = construct_planar_diffusion(mesh,:interface_a,:interface_b)
    exterior = DirichletFunction(:outside,(coords,time,i)->coords[1])
    BCs = assign((T=[pair...,exterior],);region=mesh)
    eqn,phi,cfg = planar_diffusion_equation(mesh,BCs)
    exact = [p[1] for p in mesh.cell_centre]
    A,b = Matrix(eqn.equation.A.parent),eqn.equation.b
    @test A*exact ≈ b
    @test A\b ≈ exact
    @test isposdef(Symmetric(A))
    # Exercise the existing preconditioner/workspace and scalar solve seam.
    @reset eqn.preconditioner = set_preconditioner(cfg.solvers.T,eqn)
    @reset eqn.solver = _workspace(cfg.solvers.T,eqn)
    solve_equation!(eqn,phi,BCs.T,cfg.solvers.T,cfg;time=0.0)
    @test phi.values ≈ exact rtol=1e-7
    c = pair[1].value.coupling
    flux = zeros(length(c.measure))
    interface_diffusion_flux!(flux,pair[1],phi,ConstantScalar(2.0))
    @test flux ≈ -2 .* c.measure rtol=1e-7

    @test_throws ArgumentError assign((T=[pair[1],Zerogradient(:interface_b),Zerogradient(:outside)],);region=mesh)
    wrong = PlanarDiffusion(:interface_b,pair[1].value,0:0)
    @test_throws ArgumentError assign((T=[pair[1],wrong,Zerogradient(:outside)],);region=mesh)
    unsupported = FaceScalarField(mesh)
    initialise!(unsupported,2.0)
    @test_throws Exception planar_diffusion_equation(mesh,BCs,unsupported)
    # Transient terms contribute nothing on physical interface faces.
    time_eqn = (Time{Euler}(ConstantScalar(1.0),phi)-Laplacian{Linear}(ConstantScalar(2.0),phi)
        == Source(ConstantScalar(0.0))) → ScalarEquation(phi,BCs.T)
    discretise!(time_eqn,phi,cfg)
    apply_boundary_conditions!(time_eqn,BCs.T,nothing,0.0,cfg)
    @test Matrix(time_eqn.equation.A.parent) ≈ A + Diagonal(mesh.cell_volume)
end
