using XCALibre
using Test
using LinearAlgebra
using Accessors

if !isdefined(@__MODULE__,:planar_interface_fixture)
    include("test_planar_interface.jl")
end

function planar_transport_config(mesh,BCs)
    Configuration(solvers=(T=SolverSetup(solver=Bicgstab(),preconditioner=Jacobi(),
        relax=1.0,convergence=1e-10,rtol=1e-12,atol=1e-12),),
        schemes=(T=Schemes(divergence=Upwind,laplacian=Linear),),
        runtime=Runtime(iterations=1,write_interval=-1,time_step=1),
        hardware=Hardware(backend=CPU(),workgroup=4),boundaries=BCs)
end

function assemble_transport!(eqn,phi,BCs,cfg)
    discretise!(eqn,phi,cfg)
    apply_boundary_conditions!(eqn,BCs.T,nothing,0.0,cfg)
    Matrix(eqn.equation.A.parent)
end

@testset "Conservative planar upwind transport" begin
    for (I,T) in ((Int64,Float64),(Int32,Float64),(Int32,Float32))
        mesh = planar_interface_fixture(integer=I,float=T)
        pair = construct_planar_transport(mesh,:interface_a,:interface_b)
        c = pair[1].value.coupling
        transport = InterfaceTransportFlux(mesh,c)
        BCs = assign((T=[pair...,Zerogradient(:outside)],);region=mesh)
        cfg = planar_transport_config(mesh,BCs)
        phi = ScalarField(mesh)
        phi.values .= T.([0.3,-0.7,1.1,-0.2,0.8])
        eqn = (Divergence{Upwind}(transport,phi) == Source(ConstantScalar(T(0)))) → ScalarEquation(phi,BCs.T)
        # Opposing contributions on A's first face cancel as transport flux,
        # but must not cancel as transported scalar flux (different donors).
        q = T.([0.7,-0.7,0.0,0.2])
        update_interface_flux!(transport,q)
        @test transport[c.face_a[1]] == 0
        @test eltype(transport.overlaps) == T
        @test eltype(eqn.equation.diag_nz) == I
        A = assemble_transport!(eqn,phi,BCs,cfg)
        expected = zeros(T,length(mesh.cells),length(mesh.cells))
        for k in eachindex(q)
            a,b = max(q[k],zero(T)),min(q[k],zero(T))
            ca,cb = c.cell_a[k],c.cell_b[k]
            expected[ca,ca] += a; expected[ca,cb] += b
            expected[cb,ca] -= a; expected[cb,cb] -= b
        end
        @test A == expected
        @test vec(sum(A;dims=1)) ≈ zeros(T,length(mesh.cells)) atol=10eps(T)
        out = zeros(T,length(q))
        interface_transport_flux!(out,pair[1],phi,transport)
        @test out ≈ [q[k] == 0 ? zero(T) : q[k]*phi[q[k]>0 ? c.cell_a[k] : c.cell_b[k]] for k in eachindex(q)]
        @test interface_transport_flux!(similar(out),pair[2],phi,transport) == out
        cells = zeros(T,length(mesh.cells))
        interface_cell_flux!(cells,c,out)
        @test A*phi.values ≈ cells
        @test abs(sum(cells)) <= 10eps(T)*sum(abs,out)
        for _ in 1:5
            @test assemble_transport!(eqn,phi,BCs,cfg) == A
        end
        negative = (-Divergence{Upwind}(transport,phi) == Source(ConstantScalar(T(0)))) → ScalarEquation(phi,BCs.T)
        @test assemble_transport!(negative,phi,BCs,cfg) == -A
        # New prescribed values, same equation and sparse graph.
        update_interface_flux!(transport,-q)
        Ar = assemble_transport!(eqn,phi,BCs,cfg)
        # Reversal transposes off-diagonals; diagonals move to the new donors.
        reversed_expected = Matrix(transpose(A))
        for row in axes(A,1)
            reversed_expected[row,row] = -sum(reversed_expected[k,row] for k in axes(A,1) if k != row)
        end
        @test Ar ≈ reversed_expected
        interface_transport_flux!(out,pair[1],phi,transport)
        interface_cell_flux!(cells,c,out)
        @test Ar*phi.values ≈ cells
        update_interface_flux!(transport,zeros(T,length(q)))
        @test assemble_transport!(eqn,phi,BCs,cfg) == zeros(T,size(A))
        @test_throws DimensionMismatch update_interface_flux!(transport,ones(1))
        @test_throws ArgumentError update_interface_flux!(transport,fill(NaN,length(q)))
        @test transport.overlaps == zeros(T,length(q))
        @test_throws ArgumentError interface_transport_flux!(transport.overlaps,pair[1],phi,transport)
        @test_throws DimensionMismatch interface_transport_flux!(zeros(1),pair[1],phi,transport)
        alien = InterfaceTransportFlux(mesh,construct_planar_interface(mesh,:interface_a,:interface_b))
        @test_throws ArgumentError interface_transport_flux!(out,pair[1],phi,alien)
        mismatch = (Divergence{Upwind}(alien,phi) == Source(ConstantScalar(0.0))) → ScalarEquation(phi,BCs.T)
        @test_throws Exception assemble_transport!(mismatch,phi,BCs,cfg)
        @test_throws ArgumentError interface_transport_flux!(out,pair[1],ScalarField(deepcopy(mesh)),transport)
        if T == Float32
            @test_throws ArgumentError update_interface_flux!(transport,fill(floatmax(Float64),length(q)))
            @test transport.overlaps == zeros(T,length(q))
        end
        @test_throws ArgumentError assign((T=[pair[1],Zerogradient(:interface_b),Zerogradient(:outside)],);region=mesh)
        ordinary = FaceScalarField(mesh)
        bad = (Divergence{Upwind}(ordinary,phi) == Source(ConstantScalar(0.0))) → ScalarEquation(phi,BCs.T)
        @test_throws Exception assemble_transport!(bad,phi,BCs,cfg)
        badlinear = (Divergence{Linear}(transport,phi) == Source(ConstantScalar(0.0))) → ScalarEquation(phi,BCs.T)
        @test_throws Exception assemble_transport!(badlinear,phi,BCs,cfg)
    end

    # One owner gathers multiple donors into one sparse row.
    many = planar_interface_fixture(na=1,nb=3)
    many_pair = construct_planar_transport(many,:interface_a,:interface_b)
    many_c = many_pair[1].value.coupling
    many_flux = InterfaceTransportFlux(many,many_c)
    update_interface_flux!(many_flux,[-0.2,0.3,-0.4])
    many_bcs = assign((T=[many_pair...,Zerogradient(:outside)],);region=many)
    many_phi = ScalarField(many)
    many_phi.values .= [0.2,0.7,-0.3,0.8]
    many_eqn = (Divergence{Upwind}(many_flux,many_phi) == Source(ConstantScalar(0.0))) → ScalarEquation(many_phi,many_bcs.T)
    many_A = assemble_transport!(many_eqn,many_phi,many_bcs,planar_transport_config(many,many_bcs))
    many_out = zeros(3)
    interface_transport_flux!(many_out,many_pair[1],many_phi,many_flux)
    many_balance = zeros(4)
    interface_cell_flux!(many_balance,many_c,many_out)
    @test many_A*many_phi.values ≈ many_balance
    @test many_A[1,:] ≈ [0.3,-0.2,0.0,-0.4]
    @test abs(sum(many_balance)) < 1e-14

    # Separate inlet/outlet/wall patches for an actual throughflow solve.
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
    BCs = assign((T=[pair...,Dirichlet(:inlet,3.0),Outlet(:outlet),Zerogradient(:walls)],);region=mesh)
    cfg = planar_transport_config(mesh,BCs)
    phi = ScalarField(mesh)
    eqn = (Divergence{Upwind}(transport,phi) == Source(ConstantScalar(0.0))) → ScalarEquation(phi,BCs.T)
    A = assemble_transport!(eqn,phi,BCs,cfg)
    @test A*fill(3.0,length(mesh.cells)) ≈ eqn.equation.b
    @test A\eqn.equation.b ≈ fill(3.0,length(mesh.cells))
    @reset eqn.preconditioner = set_preconditioner(cfg.solvers.T,eqn)
    @reset eqn.solver = _workspace(cfg.solvers.T,eqn)
    solve_equation!(eqn,phi,BCs.T,cfg.solvers.T,cfg;time=0.0)
    @test phi.values ≈ fill(3.0,length(mesh.cells)) rtol=1e-8
    out = zeros(length(c.measure))
    interface_transport_flux!(out,pair[1],phi,transport)
    @test sum(out) ≈ 3.0
    # Constant-state boundary contributions and normal-distance interpolation.
    phif = FaceScalarField(mesh)
    correct_boundaries!(phif,phi,BCs.T,0.0,cfg)
    @test phif.values[c.face_adjacency.IDs] ≈ fill(3.0,length(c.face_adjacency.IDs))
    gamma = ConstantScalar(0.4)
    combined = (Divergence{Upwind}(transport,phi)-Laplacian{Linear}(gamma,phi)
        == Source(ConstantScalar(0.0))) → ScalarEquation(phi,BCs.T)
    diffusion = (-Laplacian{Linear}(gamma,phi) == Source(ConstantScalar(0.0))) → ScalarEquation(phi,BCs.T)
    D = assemble_transport!(diffusion,phi,BCs,cfg)
    B = assemble_transport!(combined,phi,BCs,cfg)
    @test B ≈ A+D
    @test combined.equation.b ≈ eqn.equation.b+diffusion.equation.b
    @test B\combined.equation.b ≈ fill(3.0,length(mesh.cells))
    interface_diffusion_flux!(out,pair[1],phi,gamma)
    @test out ≈ zeros(length(out)) atol=1e-8
    timed = (Time{Euler}(ConstantScalar(1.0),phi)+Divergence{Upwind}(transport,phi)
        == Source(ConstantScalar(0.0))) → ScalarEquation(phi,BCs.T)
    @test assemble_transport!(timed,phi,BCs,cfg) ≈ A+Diagonal(mesh.cell_volume)
    # Ordinary physical-face writes remain untouched by overlap updates.
    exterior = mesh.boundaries[3].IDs_range
    old = copy(transport.values[exterior])
    update_interface_flux!(transport,-c.measure)
    @test transport.values[exterior] == old
end
