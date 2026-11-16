using XCALibre, StaticArrays, LinearAlgebra, Test

# Synthetic ordinary Mesh3: repeated physical incidences share one CSR row/column slot.
function rl_mesh(fractions)
    nf=length(fractions)
    cells=[Cell(SVector(0.,0.,0.),1.,1:0,1:nf),Cell(SVector(1.,0.,0.),1.,1:0,nf+1:2nf)]
    faces=[Face3D(1:0,SVector(1,2),SVector(.5,0.,0.),SVector(1.,0.,0.),SVector(1.,0.,0.),a,1.,.5) for a in fractions]
    nodes=XCALibre.Mesh.NodeArrays(SVector{3,Float64}[],UnitRange{Int}[])
    Mesh3(cells,Int[],vcat(collect(1:nf),collect(1:nf)),vcat(fill(2,nf),fill(1,nf)),vcat(ones(Int,nf),-ones(Int,nf)),faces,Int[],Boundary{Symbol,UnitRange{Int}}[],nodes,Int[],[0.],[0],Int[])
end
function rl_matrix(eqn,vector)
    A=vector ? eqn.equation.A0 : eqn.equation.A
    nz,col,ptr=get_sparse_fields(A)
    M=zeros(2,2)
    for i in 1:2, k in ptr[i]:ptr[i+1]-1;M[i,col[k]]=nz[k];end
    M
end
function rl_assemble(fractions,qs,gs,scheme,vector,workgroup)
    mesh=rl_mesh(fractions)
    phi=vector ? VectorField(mesh) : ScalarField(mesh)
    if vector
        phi.x.values .= [1.3,-.7];phi.y.values .= [-2.,.4];phi.z.values .= [.6,3.]
    else
        phi.values .= [1.3,-.7]
    end
    q=FaceScalarField(mesh);g=FaceScalarField(mesh)
    q.values .= qs;g.values .= gs
    config=Configuration(solvers=nothing,schemes=nothing,boundaries=nothing,
        hardware=Hardware(backend=CPU(),workgroup=workgroup),runtime=Runtime(iterations=1,write_interval=-1,time_step=1))
    src=vector ? VectorField(mesh) : ConstantScalar(0.)
    eqn=(Divergence{scheme}(q,phi)-Laplacian{Linear}(g,phi)==Source(src)) → (vector ? VectorEquation(phi,()) : ScalarEquation(phi,()))
    if length(fractions)>1
        maps=eqn.equation.face_nz
        @test length(unique(maps[1:length(fractions)]))==1
    end
    matrices=Matrix{Float64}[]
    for factor in (1.,-.7,0.)
        q.values .= factor.*qs;g.values .= gs
        A=vector ? eqn.equation.A0 : eqn.equation.A
        first(get_sparse_fields(A)) .= 17. # dirty every stored slot
        discretise!(eqn,phi,config)
        M=rl_matrix(eqn,vector); push!(matrices,copy(M))
        expected=zeros(2,2)
        for (a,f,gamma) in zip(fractions,factor.*qs,gs)
            k=a*gamma
            expected .+= [k -k;-k k]
            if scheme===Linear
                expected .+= [.5f .5f;-.5f -.5f]
            else
                expected .+= [max(f,0.) min(f,0.);-max(f,0.) -min(f,0.)]
            end
        end
        @test M ≈ expected atol=1e-12 rtol=1e-12
        for x in ([1.,1.],[1.3,-.7],[-2.,.4],[.6,3.])
            @test M*x ≈ expected*x atol=1e-12 rtol=1e-12
        end
        if vector
            @test all(iszero,eqn.equation.bx) && all(iszero,eqn.equation.by) && all(iszero,eqn.equation.bz)
        else
            @test all(iszero,eqn.equation.b)
        end
    end
    matrices
end
@testset "Repeated-link native scalar/vector assembly" begin
    for vector in (false,true), scheme in (Upwind,Linear), wg in (1,3,16), q in (-1.2,0.,.8)
        split=rl_assemble([.25,.75],q.*[.25,.75],[1.4,1.4],scheme,vector,wg)
        unsplit=rl_assemble([1.],[q],[1.4],scheme,vector,wg)
        @test all(isapprox(a,b;atol=1e-12,rtol=1e-12) for (a,b) in zip(split,unsplit))
        rl_assemble([.25,.75],[.7,-.9],[.8,1.9],scheme,vector,wg)
    end
end
