using XCALibre
using Test
using LinearAlgebra
using Accessors

if !isdefined(@__MODULE__,:planar_interface_fixture)
    include("test_planar_interface.jl")
end

function planar_pressure_config(mesh,BCs)
    Configuration(solvers=(p=SolverSetup(solver=Cg(),preconditioner=Jacobi(),
        relax=1.0,convergence=1e-10,rtol=1e-12,atol=1e-12),),
        schemes=(p=Schemes(laplacian=Linear),),
        runtime=Runtime(iterations=1,write_interval=-1,time_step=1),
        hardware=Hardware(backend=CPU(),workgroup=4),boundaries=BCs)
end

function assemble_pressure!(eqn,p,BCs,cfg)
    discretise!(eqn,p,cfg)
    apply_boundary_conditions!(eqn,BCs.p,nothing,0.0,cfg)
    Matrix(eqn.equation.A.parent)
end

@testset "Static planar pressure projection" begin
    for (I,T) in ((Int64,Float64),(Int32,Float64),(Int32,Float32))
        mesh = planar_interface_fixture(integer=I,float=T,xa=0.5,xb=2.5)
        pair = construct_planar_pressure(mesh,:interface_a,:interface_b)
        c = pair[1].value.coupling
        coeff = InterfacePressureCoefficient(mesh,pair[1])
        rD,p = ScalarField(mesh),ScalarField(mesh)
        rD.values .= T.([0.7,1.2,0.4,1.8,0.9])
        update_interface_coefficient!(coeff,rD)
        mobility = [T(0.75)*rD[c.cell_a[k]]+T(0.25)*rD[c.cell_b[k]] for k in eachindex(c.measure)]
        @test coeff.overlaps ≈ mobility
        @test coeff.conductance ≈ mobility .* c.measure
        @test eltype(coeff.conductance) == T
        BCs = assign((p=[pair...,Zerogradient(:outside)],);region=mesh)
        cfg = planar_pressure_config(mesh,BCs)
        source = ScalarField(mesh)
        eqn = (-Laplacian{Linear}(coeff,p) == -Source(source)) → ScalarEquation(p,BCs.p)
        A = assemble_pressure!(eqn,p,BCs,cfg)
        @test A ≈ transpose(A)
        @test A*ones(T,length(mesh.cells)) ≈ zeros(T,length(mesh.cells)) atol=100eps(T)
        @test eigmin(Symmetric(A)) >= -100eps(T)
        @test eltype(eqn.equation.diag_nz) == I
        # Remove ordinary interior faces to isolate the interface block.
        baseBCs = assign((p=[Zerogradient(:interface_a),Zerogradient(:interface_b),Zerogradient(:outside)],);region=mesh)
        base = (-Laplacian{Linear}(coeff,p) == Source(ConstantScalar(T(0)))) → ScalarEquation(p,baseBCs.p)
        D = A-assemble_pressure!(base,p,baseBCs,planar_pressure_config(mesh,baseBCs))
        expected = zeros(T,size(A))
        for k in eachindex(c.measure)
            a,b,G = c.cell_a[k],c.cell_b[k],coeff.conductance[k]
            expected[a,a] += G; expected[a,b] -= G
            expected[b,b] += G; expected[b,a] -= G
        end
        @test D ≈ expected atol=100eps(T)
        p.values .= T.([0.3,-0.7,1.1,-0.2,0.8])
        out = zeros(T,length(c.measure))
        interface_pressure_flux!(out,pair[1],p,coeff)
        balance = zeros(T,length(mesh.cells))
        interface_cell_flux!(balance,c,out)
        @test D*p.values ≈ balance atol=100eps(T)
        @test abs(sum(balance)) <= 100eps(T)*sum(abs,out)
        @test interface_pressure_flux!(similar(out),pair[2],p,coeff) == out
        old = copy(out)
        p.values .+= T(5)
        @test interface_pressure_flux!(out,pair[1],p,coeff) ≈ old atol=100eps(T)
        initialise!(p,T(2))
        @test interface_pressure_flux!(out,pair[1],p,coeff) == zeros(T,length(out))
        for _ in 1:5
            @test assemble_pressure!(eqn,p,BCs,cfg) == A
        end
        rD.values .*= T(2)
        update_interface_coefficient!(coeff,rD)
        @test assemble_pressure!(eqn,p,BCs,cfg) ≈ T(2).*A
        oldcoeff = copy(coeff.conductance)
        rD.values[1] = T(-1)
        @test_throws ArgumentError update_interface_coefficient!(coeff,rD)
        @test coeff.conductance == oldcoeff
        @test_throws ArgumentError update_interface_coefficient!(coeff,ScalarField(deepcopy(mesh)))
        @test_throws DimensionMismatch interface_pressure_flux!(zeros(T,1),pair[1],p,coeff)
        @test_throws ArgumentError interface_pressure_flux!(coeff.conductance,pair[1],p,coeff)
        @test_throws ArgumentError assign((p=[pair[1],Zerogradient(:interface_b),Zerogradient(:outside)],);region=mesh)
        uninitialised = InterfacePressureCoefficient(mesh,pair[1])
        @test_throws ArgumentError interface_pressure_flux!(out,pair[1],p,uninitialised)
        unsupported = (-Laplacian{Linear}(ConstantScalar(T(1)),p) == Source(ConstantScalar(T(0)))) → ScalarEquation(p,BCs.p)
        @test_throws Exception assemble_pressure!(unsupported,p,BCs,cfg)
        unset_eqn = (-Laplacian{Linear}(uninitialised,p) == Source(ConstantScalar(T(0)))) → ScalarEquation(p,BCs.p)
        @test_throws Exception assemble_pressure!(unset_eqn,p,BCs,cfg)
        wrong = InterfacePressureCoefficient(mesh,construct_planar_pressure(mesh,:interface_a,:interface_b)[1])
        update_interface_coefficient!(wrong,ScalarField(fill(T(1),length(mesh.cells)),mesh))
        @test_throws ArgumentError interface_pressure_flux!(out,pair[1],p,wrong)
        wrong_eqn = (-Laplacian{Linear}(wrong,p) == Source(ConstantScalar(T(0)))) → ScalarEquation(p,BCs.p)
        @test_throws Exception assemble_pressure!(wrong_eqn,p,BCs,cfg)
        # Pressure boundary interpolation remains geometric and constant-preserving.
        pf = FaceScalarField(mesh)
        correct_boundaries!(pf,p,BCs.p,0.0,cfg)
        @test pf.values[c.face_adjacency.IDs] ≈ fill(T(2),length(c.face_adjacency.IDs))
        negative = (Laplacian{Linear}(coeff,p) == Source(ConstantScalar(T(0)))) → ScalarEquation(p,BCs.p)
        @test assemble_pressure!(negative,p,BCs,cfg) ≈ -T(2).*A

        # Closed-domain projection: predictor includes interior and overlap fluxes.
        rD.values .= T.([0.7,1.2,0.4,1.8,0.9])
        update_interface_coefficient!(coeff,rD)
        transport = InterfaceTransportFlux(mesh,c)
        predictor = T.([0.2,-0.1,0.3,-0.2])
        update_interface_flux!(transport,predictor)
        nboundary = length(mesh.boundary_cellsID)
        for f in nboundary+1:length(mesh.faces)
            transport.values[f] = T(0.07)*T(f-nboundary)
        end
        saved_faces = copy(transport.values)
        facefield = FaceScalarField(transport.values,mesh)
        div!(source,facefield,cfg)
        @test abs(sum(source.values .* mesh.cell_volume)) < 100eps(T)
        A = assemble_pressure!(eqn,p,BCs,cfg)
        b = copy(eqn.equation.b)
        @test b ≈ -source.values .* mesh.cell_volume
        # Symmetric pressure pin: fix p[1]=0 and solve remaining equations.
        initialise!(p,T(0))
        p.values[2:end] .= A[2:end,2:end]\b[2:end]
        @test A*p.values ≈ b atol=200eps(T)
        if T == Float64
            exact = copy(p.values)
            @reset eqn.preconditioner = set_preconditioner(cfg.solvers.p,eqn)
            @reset eqn.solver = _workspace(cfg.solvers.p,eqn)
            initialise!(p,T(0))
            solve_equation!(eqn,p,BCs.p,cfg.solvers.p,cfg;time=0.0,ref=0.0)
            @test p.values .- p.values[1] ≈ exact atol=1e-9
            @test A*p.values ≈ b atol=1e-9
        end
        for f in nboundary+1:length(mesh.faces)
            a,bcell = mesh.face_ownerCells[f]
            G = coeff.values[f]*mesh.face_gDiff[f]
            transport.values[f] = saved_faces[f]+G*(p[a]-p[bcell])
        end
        correct_interface_pressure_flux!(transport,predictor,pair[1],p,coeff)
        corrected = copy(transport.values)
        correct_interface_pressure_flux!(transport,predictor,pair[2],p,coeff)
        @test transport.values == corrected
        div!(source,facefield,cfg)
        @test maximum(abs,source.values .* mesh.cell_volume) < 300eps(T)
        @test sum(transport.values[mesh.boundaries[1].IDs_range]) ≈ -sum(transport.values[mesh.boundaries[2].IDs_range]) atol=100eps(T)
        @test_throws ArgumentError correct_interface_pressure_flux!(transport,transport.overlaps,pair[1],p,coeff)
        @test_throws DimensionMismatch correct_interface_pressure_flux!(transport,zeros(T,1),pair[1],p,coeff)
        @test_throws ArgumentError correct_interface_pressure_flux!(transport,fill(T(NaN),length(predictor)),pair[1],p,coeff)
        @test transport.values == corrected
    end

    # Split one overlap into two shared pieces with identical owner pair.
    mesh = planar_interface_fixture()
    original = construct_planar_pressure(mesh,:interface_a,:interface_b)
    c = original[1].value.coupling
    ids = [1,1,2,3,4]
    lengths = c.measure[ids]; lengths[1:2] ./= 2
    vectors = c.area_vector[ids]; vectors[1:2] ./= 2
    fa,fb,ca,cb = c.face_a[ids],c.face_b[ids],c.cell_a[ids],c.cell_b[ids]
    adjacency = XCALibre.InterfaceCoupling._interface_adjacency
    split = PlanarInterface(c.patch_a,c.patch_b,fa,fb,ca,cb,lengths,c.centre[ids],vectors,
        adjacency(fa,fb),adjacency(ca,cb))
    V = XCALibre.Discretise.PlanarDiffusionValue
    pair = map(original) do bc
        v = bc.value
        PlanarPressure(bc.ID,V(split,v.distance_a[ids],v.distance_b[ids],v.rows,v.side_a),bc.IDs_range)
    end
    rD,p = ScalarField(mesh),ScalarField(mesh)
    initialise!(rD,1.0); p.values .= [0.2,-0.4,0.8,-0.7,1.2]
    coeff = InterfacePressureCoefficient(mesh,pair[1])
    update_interface_coefficient!(coeff,rD)
    BCs = assign((p=[pair...,Zerogradient(:outside)],);region=mesh)
    eqn = (-Laplacian{Linear}(coeff,p) == Source(ConstantScalar(0.0))) → ScalarEquation(p,BCs.p)
    A = assemble_pressure!(eqn,p,BCs,planar_pressure_config(mesh,BCs))
    @test A[ca[1],cb[1]] ≈ -(coeff.conductance[1]+coeff.conductance[2])
    out = zeros(5)
    interface_pressure_flux!(out,pair[1],p,coeff)
    @test out[1] ≈ coeff.conductance[1]*(p[ca[1]]-p[cb[1]])
    @test out[2] == out[1]
    @test out[1] != -A[ca[1],cb[1]]*(p[ca[1]]-p[cb[1]])
end
