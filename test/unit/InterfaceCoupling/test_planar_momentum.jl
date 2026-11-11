using XCALibre
using Test
using LinearAlgebra
using StaticArrays
using Accessors

if !isdefined(@__MODULE__,:planar_interface_fixture)
    include("test_planar_interface.jl")
end

function planar_momentum_config(mesh,BCs; divergence=Upwind)
    Configuration(solvers=(U=SolverSetup(solver=Bicgstab(),preconditioner=Jacobi(),
        relax=1.0,convergence=1e-10,rtol=1e-12,atol=1e-12),),
        schemes=(U=Schemes(divergence=divergence,laplacian=Linear),),
        runtime=Runtime(iterations=1,write_interval=-1,time_step=1),
        hardware=Hardware(backend=CPU(),workgroup=4),boundaries=BCs)
end

function assemble_momentum!(eqn,U,BCs,cfg,component)
    discretise!(eqn,U,cfg)
    update_equation!(eqn,cfg)
    apply_boundary_conditions!(eqn,BCs.U,component,0.0,cfg)
    Matrix(eqn.equation.A.parent)
end

@testset "Conservative planar vector momentum" begin
    for (I,T) in ((Int64,Float64),(Int32,Float64),(Int32,Float32))
        mesh = planar_interface_fixture(integer=I,float=T,xa=0.5,xb=2.5)
        pair = construct_planar_transport(mesh,:interface_a,:interface_b)
        c = pair[1].value.coupling
        transport = InterfaceTransportFlux(mesh,c)
        q = T.([0.3,-0.2,0.1,-0.4])
        update_interface_flux!(transport,q)
        BCs = assign((U=[pair...,Zerogradient(:outside)],);region=mesh)
        cfg = planar_momentum_config(mesh,BCs)
        U = VectorField(mesh)
        for i in eachindex(U.x.values)
            U[i] = SVector{3,T}(T(0.2)*i,T(-0.3)*i+T(1),T(0.1)*i-T(2))
        end
        zero_source = VectorField(mesh)
        initialise!(zero_source,SVector{3,T}(0,0,0))
        gamma = ConstantScalar(T(0.4))
        scalarBCs = assign((s=[pair...,Zerogradient(:outside)],);region=mesh)
        phi = ScalarField(mesh)
        for S in (Upwind,Linear)
            eqn = (Divergence{S}(transport,U)-Laplacian{Linear}(gamma,U) == Source(zero_source)) → VectorEquation(U,BCs.U)
            scalar = (Divergence{S}(transport,phi)-Laplacian{Linear}(gamma,phi) == Source(ConstantScalar(T(0)))) → ScalarEquation(phi,scalarBCs.s)
            discretise!(scalar,phi,cfg)
            apply_boundary_conditions!(scalar,scalarBCs.s,nothing,0.0,cfg)
            A = Matrix(scalar.equation.A.parent)
            for direction in (XDir(),YDir(),ZDir())
                @test assemble_momentum!(eqn,U,BCs,cfg,direction) ≈ A
            end
            @test eltype(eqn.equation.diag_nz) == I
            # Component switching without a new volume assembly must restore A0.
            for _ in 1:4, direction in (XDir(),YDir(),ZDir())
                update_equation!(eqn,cfg)
                apply_boundary_conditions!(eqn,BCs.U,direction,0.0,cfg)
                @test Matrix(eqn.equation.A.parent) == A
            end
            out = zeros(SVector{3,T},length(q))
            diff = similar(out)
            interface_transport_flux!(out,pair[1],U,transport;scheme=S())
            interface_diffusion_flux!(diff,pair[1],U,gamma)
            @test interface_transport_flux!(similar(out),pair[2],U,transport;scheme=S()) == out
            @test interface_diffusion_flux!(similar(out),pair[2],U,gamma) == diff
            for (j,component) in enumerate((U.x,U.y,U.z))
                field = ScalarField(component.values,mesh)
                scalarout = zeros(T,length(q))
                interface_transport_flux!(scalarout,pair[1],field,transport;scheme=S())
                @test getindex.(out,j) ≈ scalarout
                interface_diffusion_flux!(scalarout,pair[1],field,gamma)
                @test getindex.(diff,j) ≈ scalarout
            end
            balances = zeros(SVector{3,T},length(mesh.cells))
            interface_cell_flux!(balances,c,out+diff)
            @test norm(sum(balances)) < 100eps(T)
            # With outside insulated and ordinary face fluxes zero, remove interior diffusion.
            baseBCs = assign((U=[Zerogradient(:interface_a),Zerogradient(:interface_b),Zerogradient(:outside)],);region=mesh)
            base = (-Laplacian{Linear}(gamma,U) == Source(zero_source)) → VectorEquation(U,baseBCs.U)
            interior = assemble_momentum!(base,U,baseBCs,planar_momentum_config(mesh,baseBCs),XDir())
            D = A-interior
            for j in 1:3
                @test D*getindex.([U[i] for i in eachindex(U.x.values)],j) ≈ getindex.(balances,j) atol=200eps(T)
            end
            negative = (-Divergence{S}(transport,U)+Laplacian{Linear}(gamma,U) == Source(zero_source)) → VectorEquation(U,BCs.U)
            @test assemble_momentum!(negative,U,BCs,cfg,XDir()) ≈ -A
            update_interface_flux!(transport,-q)
            reversed = assemble_momentum!(eqn,U,BCs,cfg,XDir())
            discretise!(scalar,phi,cfg)
            apply_boundary_conditions!(scalar,scalarBCs.s,nothing,0.0,cfg)
            @test reversed ≈ Matrix(scalar.equation.A.parent)
            update_interface_flux!(transport,q)
            faces = zeros(SVector{3,T},length(mesh.faces))
            interface_face_flux!(faces,c,out)
            @test norm(sum(faces[c.face_adjacency.IDs])) < 100eps(T)
        end
        # Normal-linear fields: every overlap has the same exact interface trace.
        for i in eachindex(U.x.values)
            x = mesh.cell_centre[i][1]
            U[i] = SVector{3,T}(2x+1,-3x+2,x-4)
        end
        exact = SVector{3,T}(3,-1,-3)
        out = zeros(SVector{3,T},length(q))
        interface_transport_flux!(out,pair[1],U,transport;scheme=Linear())
        @test out ≈ q .* Ref(exact)
        Uf = FaceVectorField(mesh)
        correct_boundaries!(Uf,U,BCs.U,0.0,cfg)
        @test [Uf[f] for f in c.face_adjacency.IDs] ≈ fill(exact,length(c.face_adjacency.IDs))
        # Standalone PlanarDiffusion also accepts vectors and geometric interpolation.
        dpair = construct_planar_diffusion(mesh,:interface_a,:interface_b)
        dBCs = assign((U=[dpair...,Zerogradient(:outside)],);region=mesh)
        dcfg = planar_momentum_config(mesh,dBCs)
        deqn = (-Laplacian{Linear}(gamma,U) == Source(zero_source)) → VectorEquation(U,dBCs.U)
        A = assemble_momentum!(deqn,U,dBCs,dcfg,XDir())
        @test A ≈ transpose(A)
        correct_boundaries!(Uf,U,dBCs.U,0.0,dcfg)
        @test [Uf[f] for f in dpair[1].value.coupling.face_adjacency.IDs] ≈ fill(exact,length(c.face_adjacency.IDs))
        interface_diffusion_flux!(out,dpair[1],U,gamma)
        slope = SVector{3,T}(2,-3,1)
        @test out ≈ -T(0.4).*c.measure .* Ref(slope)
        # Zero flow and unused donor NaNs must not contaminate upwind flux.
        initialise!(U,SVector{3,T}(T(NaN),T(NaN),T(NaN)))
        update_interface_flux!(transport,zeros(T,length(q)))
        @test interface_transport_flux!(out,pair[1],U,transport;scheme=Linear()) == zeros(SVector{3,T},length(q))
        update_interface_flux!(transport,abs.(q))
        for i in unique(c.cell_a)
            U[i] = exact
        end
        @test interface_transport_flux!(out,pair[1],U,transport) ≈ abs.(q).*Ref(exact)
        @test_throws ArgumentError interface_transport_flux!(out,pair[1],U,transport;scheme=LUST())
        @test_throws DimensionMismatch interface_transport_flux!(out[1:1],pair[1],U,transport)
        @test_throws DimensionMismatch interface_diffusion_flux!(out[1:1],pair[1],U,gamma)
        @test_throws ArgumentError interface_transport_flux!(view(U.x.values,1:length(q)),pair[1],U,transport)
        @test_throws ArgumentError interface_diffusion_flux!(view(U.z.values,1:length(q)),pair[1],U,gamma)
        unsupported = (Divergence{LUST}(transport,U) == Source(zero_source)) → VectorEquation(U,BCs.U)
        @test_throws Exception assemble_momentum!(unsupported,U,BCs,cfg,XDir())
        timed = (Time{Euler}(ConstantScalar(T(1)),U)-Laplacian{Linear}(gamma,U) == Source(zero_source)) → VectorEquation(U,dBCs.U)
        @test assemble_momentum!(timed,U,dBCs,dcfg,XDir()) ≈ A+Diagonal(mesh.cell_volume)
    end
end

@testset "Planar vector convection-diffusion solve" begin
    P,line = XCALibre.Point,XCALibre.line!
    pts = [P(x,y,0.0) for (x,y) in ((0.,0.),(1.,0.),(0.,1.),(1.,1.),(1.,0.),(2.,0.),(1.,1.),(2.,1.))]
    edges = [line(pts,1,2,1),line(pts,3,4,1),line(pts,1,3,2),line(pts,2,4,2),
        line(pts,5,6,1),line(pts,7,8,1),line(pts,5,7,3),line(pts,6,8,3)]
    patches = [Patch(:interface_a,[4]),Patch(:interface_b,[7]),Patch(:inlet,[3]),
        Patch(:outlet,[8]),Patch(:walls,[1,2,5,6])]
    mesh = UNV2.update_mesh_format(generate!(MeshBuilder2D(pts,edges,patches,
        [quad(edges,[1,2,3,4]),quad(edges,[5,6,7,8])])),Int64,Float64)
    pair = construct_planar_transport(mesh,:interface_a,:interface_b)
    c = pair[1].value.coupling
    transport = InterfaceTransportFlux(mesh,c)
    transport.values .= mesh.face_area .* [n[1] for n in mesh.face_normal]
    update_interface_flux!(transport,c.measure)
    exact = SVector(3.0,-2.0,0.0)
    BCs = assign((U=[pair...,Dirichlet(:inlet,collect(exact)),Outlet(:outlet),Zerogradient(:walls)],);region=mesh)
    U = VectorField(mesh)
    for S in (Upwind,Linear)
        cfg = planar_momentum_config(mesh,BCs;divergence=S)
        eqn = (Divergence{S}(transport,U)-Laplacian{Linear}(ConstantScalar(0.4),U)
            == Source(VectorField(mesh))) → VectorEquation(U,BCs.U)
        @reset eqn.preconditioner = set_preconditioner(cfg.solvers.U,eqn)
        @reset eqn.solver = _workspace(cfg.solvers.U,eqn,XDir())
        for _ in 1:3
            initialise!(U,SVector(0.0,0.0,0.0))
            solve_equation!(eqn,U,BCs.U,cfg.solvers.U,XDir(),YDir(),ZDir(),cfg;time=0.0)
            @test U.x.values ≈ fill(exact[1],length(mesh.cells)) rtol=1e-8
            @test U.y.values ≈ fill(exact[2],length(mesh.cells)) rtol=1e-8
            @test U.z.values == zeros(length(mesh.cells)) # Mesh2 solver does not solve z.
        end
        out = zeros(SVector{3,Float64},length(c.measure))
        interface_transport_flux!(out,pair[1],U,transport;scheme=S())
        @test sum(out) ≈ exact
        interface_diffusion_flux!(out,pair[1],U,ConstantScalar(0.4))
        @test maximum(norm,out) < 1e-8
    end
end
