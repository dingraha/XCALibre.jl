using XCALibre
using Test
using LinearAlgebra
using StaticArrays
using Accessors

if !isdefined(@__MODULE__,:planar_interface_fixture)
    include("test_planar_interface.jl")
end

function planar_predictor_config(mesh,BCs)
    setup = SolverSetup(solver=Bicgstab(),preconditioner=Jacobi(),relax=1.0,
        convergence=1e-10,rtol=1e-12,atol=1e-12)
    Configuration(solvers=(U=setup,p=setup),schemes=(U=Schemes(divergence=Upwind,laplacian=Linear),p=Schemes(gradient=Gauss,laplacian=Linear)),
        runtime=Runtime(iterations=1,write_interval=-1,time_step=1),
        hardware=Hardware(backend=CPU(),workgroup=4),boundaries=BCs)
end

@testset "Planar Gauss gradient and predictor" begin
    for (I,T) in ((Int64,Float64),(Int32,Float64),(Int32,Float32))
        mesh = planar_interface_fixture(integer=I,float=T,xa=0.5,xb=2.5)
        pair = construct_planar_transport(mesh,:interface_a,:interface_b)
        pressure = construct_planar_pressure(pair)
        c = pair[1].value.coupling
        @test pressure[1].value === pair[1].value
        @test pressure[2].value === pair[2].value
        @test construct_planar_pressure(reverse(pair))[1].value === pair[2].value
        @test_throws ArgumentError construct_planar_pressure((pair[1],pair[1]))
        other = construct_planar_transport(mesh,:interface_a,:interface_b)
        @test_throws ArgumentError construct_planar_pressure((pair[1],other[2]))
        p,pf,Hv = ScalarField(mesh),FaceScalarField(mesh),VectorField(mesh)
        gradient = Grad{Gauss}(p)
        for offset in (T(0),T(4))
            BCs = assign((p=[pressure...,DirichletFunction(:outside,(x,t,i)->T(2)*x[1]+T(1)+offset)],);region=mesh)
            cfg = planar_predictor_config(mesh,BCs)
            p.values .= [T(2)*x[1]+T(1)+offset for x in mesh.cell_centre]
            grad!(gradient,pf,p,BCs.p,0.0,cfg)
            @test gradient.result.x.values ≈ fill(T(2),length(mesh.cells)) atol=100eps(T)
            @test maximum(abs,gradient.result.y.values) < 100eps(T)
            @test maximum(abs,gradient.result.z.values) < 100eps(T)
        end
        BCs = assign((p=[pressure...,Zerogradient(:outside)],);region=mesh)
        cfg = planar_predictor_config(mesh,BCs)
        initialise!(p,T(3))
        grad!(gradient,pf,p,BCs.p,0.0,cfg)
        @test maximum(norm,[gradient[i] for i in eachindex(p.values)]) < 100eps(T)
        # Arbitrary traces: compare face-averaged pressure area sums to overlap sums.
        p.values .= T.([0.3,-0.8,0.1,1.4,-0.2])
        correct_boundaries!(pf,p,BCs.p,0.0,cfg)
        overlap_terms = [((T(0.75)*p[c.cell_a[k]]+T(0.25)*p[c.cell_b[k]])*c.area_vector[k]) for k in eachindex(c.measure)]
        direct = zeros(SVector{3,T},length(mesh.cells))
        interface_cell_flux!(direct,c,overlap_terms)
        averaged = zeros(SVector{3,T},length(mesh.cells))
        for f in c.face_adjacency.IDs
            averaged[mesh.face_ownerCells[f][1]] += pf[f]*mesh.face_area[f]*mesh.face_normal[f]
        end
        @test direct ≈ averaged atol=100eps(T)
        for i in eachindex(Hv.x.values)
            x = mesh.cell_centre[i][1]
            Hv[i] = SVector{3,T}(2x+1,-3x+2,x-4)
        end
        out = zeros(T,length(c.measure))
        @test interface_predictor_flux!(out,pair[1],Hv) ≈ T(3).*c.measure
        @test eltype(out) == T
        @test interface_predictor_flux!(similar(out),pair[2],Hv) == out
        # One physical A face has two donors with opposing velocities.
        initialise!(Hv,SVector{3,T}(0,0,0))
        for (i,cell) in enumerate(unique(c.cell_b))
            Hv[cell] = SVector{3,T}(i == 2 ? -T(8) : T(8),T(2),T(0))
        end
        interface_predictor_flux!(out,pair[1],Hv)
        @test out[1] > 0 && out[2] < 0
        field = InterfaceTransportFlux(mesh,c)
        field.values .= T(7)
        before = copy(field.values)
        for _ in 1:4
            @test interface_predictor_flux!(similar(out),pair[1],Hv) == out
        end
        @test field.values == before
        update_interface_flux!(field,out)
        @test field.values[c.face_a[1]] ≈ out[1]+out[2]
        @test field.overlaps == out
        @test_throws DimensionMismatch interface_predictor_flux!(out[1:1],pair[1],Hv)
        @test_throws ArgumentError interface_predictor_flux!(view(Hv.x.values,1:length(out)),pair[1],Hv)
        @test_throws ArgumentError interface_predictor_flux!(c.measure,pair[1],Hv)
        @test_throws ArgumentError interface_predictor_flux!(zeros(Int,length(out)),pair[1],Hv)
        incompatible = VectorField(planar_interface_fixture(integer=I,float=T,na=3,nb=2))
        @test_throws ArgumentError interface_predictor_flux!(out,pair[1],incompatible)
        saved = copy(out)
        Hv[c.cell_a[1]] = SVector{3,T}(T(NaN),0,0)
        @test_throws ArgumentError interface_predictor_flux!(out,pair[1],Hv)
        @test out == saved
    end
end

@testset "Planar momentum predictor projection chain" begin
    for (I,T) in ((Int64,Float64),(Int32,Float64),(Int32,Float32))
        mesh = planar_interface_fixture(integer=I,float=T,xa=0.5,xb=2.5)
        pair = construct_planar_transport(mesh,:interface_a,:interface_b)
        pressure = construct_planar_pressure(pair)
        c = pair[1].value.coupling
        BCs = assign((U=[pair...,Dirichlet(:outside,[T(0),T(0),T(0)])],
            p=[pressure...,Zerogradient(:outside)]);region=mesh)
        cfg = planar_predictor_config(mesh,BCs)
        U,Hv,forcing = VectorField(mesh),VectorField(mesh),VectorField(mesh)
        p,rD,divergence = ScalarField(mesh),ScalarField(mesh),ScalarField(mesh)
        pf,Uf = FaceScalarField(mesh),FaceVectorField(mesh)
        gradient = Grad{Gauss}(p)
        field = InterfaceTransportFlux(mesh,c)
        update_interface_flux!(field,T.([0.2,-0.1,0.3,-0.4]))
        for i in eachindex(p.values)
            U[i] = SVector{3,T}(T(0.1)*i,T(-0.2)*i,0)
            forcing[i] = SVector{3,T}(T(0.3)*i,T(0.15)*i,0)
        end
        # Pressure source first, as expected by remove_pressure_source!.
        p.values .= T.([0.2,-0.3,0.8,0.1,-0.4])
        grad!(gradient,pf,p,BCs.p,0.0,cfg)
        momentum = (Divergence{Upwind}(field,U)-Laplacian{Linear}(ConstantScalar(T(0.4)),U)
            == -Source(gradient.result)+Source(forcing)) → VectorEquation(U,BCs.U)
        discretise!(momentum,U,cfg)
        update_equation!(momentum,cfg)
        apply_boundary_conditions!(momentum,BCs.U,XDir(),0.0,cfg)
        A = Matrix(momentum.equation.A.parent)
        original_b = [copy(momentum.equation.bx),copy(momentum.equation.by),copy(momentum.equation.bz)]
        remove_pressure_source!(momentum,gradient,cfg)
        for (j,b) in enumerate((momentum.equation.bx,momentum.equation.by,momentum.equation.bz))
            @test b ≈ original_b[j] .+ getindex.([gradient[i] for i in eachindex(p.values)],j).*mesh.cell_volume atol=200eps(T)
        end
        inverse_diagonal!(rD,momentum,cfg)
        @test rD.values ≈ mesh.cell_volume ./ diag(A)
        H!(Hv,U,momentum,cfg)
        offdiag = A-Diagonal(diag(A))
        for (component,h,b) in zip((U.x,U.y,U.z),(Hv.x,Hv.y,Hv.z),(momentum.equation.bx,momentum.equation.by,momentum.equation.bz))
            @test h.values ≈ (b-offdiag*component.values)./diag(A) atol=200eps(T)
        end
        # Ensure this really exercises interface neighbours rather than just volume edges.
        interface_part = zeros(T,size(A))
        for k in eachindex(c.measure)
            a,b = c.cell_a[k],c.cell_b[k]
            interface_part[a,b] = A[a,b]; interface_part[b,a] = A[b,a]
        end
        @test norm(interface_part*U.x.values) > T(0.01)
        interpolate!(Uf,Hv,cfg)
        correct_boundaries!(Uf,Hv,BCs.U,0.0,cfg)
        facefield = FaceScalarField(field.values,mesh)
        flux!(facefield,Uf,cfg)
        predictor = zeros(T,length(c.measure))
        interface_predictor_flux!(predictor,pair[1],Hv)
        update_interface_flux!(field,predictor)
        predicted_faces = copy(field.values)
        div!(divergence,facefield,cfg)
        @test maximum(abs,divergence.values) > T(0.01)
        @test abs(sum(divergence.values .* mesh.cell_volume)) < 100eps(T)
        coefficient = InterfacePressureCoefficient(mesh,pressure[1])
        update_interface_coefficient!(coefficient,rD)
        projection = (-Laplacian{Linear}(coefficient,p) == -Source(divergence)) → ScalarEquation(p,BCs.p)
        discretise!(projection,p,cfg)
        apply_boundary_conditions!(projection,BCs.p,nothing,0.0,cfg)
        P = Matrix(projection.equation.A.parent)
        b = copy(projection.equation.b)
        initialise!(p,T(0))
        p.values[2:end] .= P[2:end,2:end]\b[2:end]
        @test P*p.values ≈ b atol=200eps(T)
        if T == Float64
            @reset projection.preconditioner = set_preconditioner(cfg.solvers.p,projection)
            @reset projection.solver = _workspace(cfg.solvers.p,projection)
            solve_equation!(projection,p,BCs.p,cfg.solvers.p,cfg;time=0.0,ref=0.0)
            @test P*p.values ≈ b atol=1e-9
        end
        corrected_faces,corrected_velocity = nothing,nothing
        for offset in (T(0),T(4))
            p.values .+= offset
            nboundary = length(mesh.boundary_cellsID)
            field.values .= predicted_faces
            for f in nboundary+1:length(mesh.faces)
                a,bcell = mesh.face_ownerCells[f]
                field.values[f] += coefficient.values[f]*mesh.face_gDiff[f]*(p[a]-p[bcell])
            end
            correct_interface_pressure_flux!(field,predictor,pressure[1],p,coefficient)
            div!(divergence,facefield,cfg)
            @test maximum(abs,divergence.values .* mesh.cell_volume) < 500eps(T)
            @test sum(field.values[mesh.boundaries[c.patch_a].IDs_range]) ≈ -sum(field.values[mesh.boundaries[c.patch_b].IDs_range]) atol=100eps(T)
            grad!(gradient,pf,p,BCs.p,0.0,cfg)
            correct_velocity!(U,Hv,gradient,rD,cfg)
            @test [U[i] for i in eachindex(p.values)] ≈ [Hv[i]-rD[i]*gradient[i] for i in eachindex(p.values)]
            if corrected_faces === nothing
                corrected_faces = copy(field.values)
                corrected_velocity = [U[i] for i in eachindex(p.values)]
            else
                @test field.values ≈ corrected_faces atol=500eps(T)
                @test [U[i] for i in eachindex(p.values)] ≈ corrected_velocity atol=500eps(T)
            end
            saved = copy(field.values)
            correct_interface_pressure_flux!(field,predictor,pressure[2],p,coefficient)
            @test field.values == saved
        end
        # Cell velocity interpolation is not the conservative pressure-corrected flux.
        interpolated = similar(predictor)
        interface_predictor_flux!(interpolated,pair[1],U)
        @test norm(interpolated-field.overlaps) > T(1e-4)
        @test field.values ≈ corrected_faces atol=500eps(T)
    end
end
