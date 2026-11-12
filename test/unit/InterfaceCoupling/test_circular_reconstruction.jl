using XCALibre, Test, LinearAlgebra, StaticArrays
isdefined(@__MODULE__,:circular_interface_fixture) || include("test_circular_interface.jl")

@testset "Circular cell reconstruction and corrected flux" begin
    for (I,T) in ((Int64,Float64),(Int32,Float64),(Int32,Float32)), phase in (0.,0.13,1.1)
        O = SVector(2.,-3.,0.7)
        mesh = circular_interface_fixture(integer=I,float=T,origin=O,phase=phase)
        c = construct_circular_interface(mesh,:interface_a,:interface_b;centre=O,radius=1)
        r = construct_circular_reconstruction(mesh,c)
        f = construct_circular_flux_geometry(r)
        tol = 3000eps(T)
        @test all(x -> length(x)==length(unique(x)),r.neighbours)
        @test all(>(0),f.beta)
        @test all(w -> 0 < w < 1,f.weight_a)
        @test all(norm(f.remainder[k]+f.beta[k]*(f.points[c.cell_b[k]]-f.points[c.cell_a[k]])-c.area_vector[k]) < tol for k in eachindex(c.measure))
        for exact in (SVector{3,T}(0,0,0),SVector{3,T}(1,0,0),SVector{3,T}(0,1,0),SVector{3,T}(0.7,-0.3,0))
            p = [T(2)+dot(exact,x-c.origin) for x in r.points]
            e = [T(2)+dot(exact,x-c.origin) for x in r.exterior_points]
            g = fill(zero(exact),length(p))
            circular_reconstruct!(g,r,p,e)
            @test all(x -> isapprox(x,exact;atol=tol,rtol=tol),g)
            savedg = copy(g)
            circular_reconstruct!(g,r,p .+ T(3),e .+ T(3))
            @test g ≈ savedg atol=tol
            integral = fill(zero(exact),length(c.measure)); reference = similar(integral)
            circular_reconstructed_pressure_area!(integral,f,p,savedg)
            circular_pressure_area!(reference,c,T(2),exact)
            @test integral ≈ reference atol=tol rtol=tol
            baseline = copy(integral)
            circular_reconstructed_pressure_area!(integral,f,p .+ T(3),savedg)
            @test integral ≈ baseline+T(3)*c.area_vector atol=tol
            facep = [T(2)+dot(exact,x-c.origin) for x in mesh.face_centre]
            facep[r.metrics.interface_face] .= NaN
            circular_gauss_gradient!(g,r.metrics,facep,baseline)
            @test all(x -> isapprox(x,exact;atol=tol,rtol=tol),g)
            gamma = T(0.4)
            terms = circular_diffusion_terms(f,savedg,gamma)
            q = zeros(T,length(c.measure))
            circular_diffusion_flux!(q,terms,p)
            @test q ≈ [-gamma*dot(exact,S) for S in c.area_vector] atol=tol
            before = copy(q)
            circular_diffusion_flux!(q,terms,p .+ T(3))
            @test q ≈ before atol=tol
            # Independently assemble coefficient blocks and explicit RHS.
            n = length(p); A=zeros(T,n,n); b=zeros(T,n)
            for k in eachindex(q)
                a,bb = c.cell_a[k],c.cell_b[k]; G=terms.conductance[k]; E=terms.correction[k]
                A[a,a]+=G; A[a,bb]-=G; A[bb,a]-=G; A[bb,bb]+=G
                b[a]-=E; b[bb]+=E
            end
            balance = zeros(T,n)
            interface_cell_flux!(balance,c,q)
            @test balance ≈ A*p-b atol=tol
            @test A ≈ A' atol=tol
            @test norm(A*ones(T,n)) < tol
            @test abs(sum(balance)) < tol
            @test minimum(eigvals(Symmetric(Matrix{Float64}(A)))) > -tol
            # Frozen terms do not see later changes in source gradients/geometry.
            fill!(g,SVector{3,T}(9,9,0))
            circular_diffusion_flux!(q,terms,p)
            @test q ≈ before atol=tol
        end
    end
end

@testset "Circular seam deduplication and primitive validation" begin
    mesh = circular_interface_fixture(na=4,nb=4,angles_a=[0.2,1.6,3.2,4.9],angles_b=[0.2,1.6,3.2,4.9])
    c = circular_coupling(mesh)
    r = construct_circular_reconstruction(mesh,c)
    @test all(length(intersect(row,unique(c.cell_b))) == 1 for row in r.neighbours[unique(c.cell_a)])
    # Cyclic rotation avoids seam splitting yet gives the same LSQ normal matrices.
    rotated = circular_interface_fixture(na=4,nb=4,angles_a=[0.,1.4,3.,4.7],angles_b=[0.,1.4,3.,4.7])
    rr = construct_circular_reconstruction(rotated,circular_coupling(rotated))
    @test length(rr.metrics.coupling.measure) == length(c.measure)-1
    R = @SMatrix [cos(-0.2) -sin(-0.2);sin(-0.2) cos(-0.2)]
    @test all(isapprox(y,R*x*R';atol=1e-12) for (x,y) in zip(r.inverse,rr.inverse))
    # This coarse nonuniform fixture is valid surface geometry but its polygonal
    # outer cell points fail arc-plane bracketing: reject rather than clip weights.
    @test_throws ArgumentError construct_circular_flux_geometry(r)
    mesh = circular_interface_fixture(na=8,nb=11)
    c = circular_coupling(mesh)
    r = construct_circular_reconstruction(mesh,c)
    f = construct_circular_flux_geometry(r)
    p = ones(length(r.points)); e=ones(length(r.exterior_faces)); g=fill(SVector(0.,0.,0.),length(p))
    out = fill(SVector(9.,9.,9.),length(p)); saved=copy(out)
    badp=copy(p);badp[1]=NaN
    @test_throws ArgumentError circular_reconstruct!(out,r,badp,e)
    @test out == saved
    @test_throws DimensionMismatch circular_reconstruct!(out,r,p,e[1:1])
    @test_throws ArgumentError circular_reconstruct!(r.points,r,p,e)
    @test_throws ArgumentError construct_circular_reconstruction(mesh,c;condition_limit=1)
    @test_throws ArgumentError construct_circular_reconstruction(mesh,c;condition_limit=NaN)
    @test_throws ArgumentError XCALibre.InterfaceCoupling._circular_inverse(@SMatrix([1. 0.;0. 0.]),1e6,1)
    @test_throws ArgumentError XCALibre.InterfaceCoupling._circular_inverse(@SMatrix([1. 0.;0. 1e-9]),1e6,1)
    deficient = deepcopy(mesh)
    deficient.cell_centre .= [SVector(x[1],0.,0.) for x in deficient.cell_centre]
    for j in eachindex(deficient.face_centre)
        if !r.metrics.interface_face[j]
            deficient.face_centre[j] = SVector(deficient.face_centre[j][1],0.,0.)
        end
    end
    @test_throws ArgumentError construct_circular_reconstruction(deficient,c)
    unsuitable=deepcopy(r)
    unsuitable.points[c.cell_b[1]]=unsuitable.points[c.cell_a[1]]
    @test_throws ArgumentError construct_circular_flux_geometry(unsuitable)
    backwards=deepcopy(r)
    backwards.points[c.cell_b[1]]=2backwards.points[c.cell_a[1]]-backwards.points[c.cell_b[1]]
    @test_throws ArgumentError construct_circular_flux_geometry(backwards)
    @test_throws ArgumentError construct_circular_flux_geometry(r;min_alignment=0)
    @test_throws ArgumentError construct_circular_flux_geometry(r;min_alignment=1)
    for gamma in (0.,-1.,NaN,Inf)
        @test_throws ArgumentError circular_diffusion_terms(f,g,gamma)
    end
    badg=copy(g);badg[1]=SVector(0.,0.,1.)
    @test_throws ArgumentError circular_diffusion_terms(f,badg,1.)
    area = fill(SVector(9.,9.,9.),length(c.measure)); old=copy(area)
    @test_throws ArgumentError circular_reconstructed_pressure_area!(area,f,badp,g)
    @test area == old
    @test_throws ArgumentError circular_reconstructed_pressure_area!(f.coupling.area_vector,f,p,g)
    terms=circular_diffusion_terms(f,g,1.)
    q=fill(9.,length(c.measure));oldq=copy(q)
    @test_throws ArgumentError circular_diffusion_flux!(q,terms,badp)
    @test q == oldq
    @test_throws ArgumentError circular_diffusion_flux!(terms.conductance,terms,p)
    @test_throws DimensionMismatch circular_diffusion_flux!(q,terms,p[1:1])
    # Source geometry mutation cannot change the frozen assembly/recovery terms.
    frozen = copy(terms.geometry.beta)
    f.beta[1] *= 2
    @test terms.geometry.beta == frozen
    # Output conversion overflow also leaves caller storage untouched.
    huge = fill(1e40,length(p));huge[c.cell_a[1]]=2e40
    small = fill(Float32(9),length(q))
    @test_throws ArgumentError circular_diffusion_flux!(small,terms,huge)
    @test all(==(9),small)
end

@testset "Circular logarithmic radial flux refinement" begin
    errors = Float64[]
    for k in (1,2,4)
        mesh=circular_interface_fixture(na=8k,nb=11k,phase=0.13/k,
            inner_radius=1-0.5/k,outer_radius=1+0.5/k)
        c=circular_coupling(mesh)
        r=construct_circular_reconstruction(mesh,c)
        f=construct_circular_flux_geometry(r)
        p=[log(norm(x)) for x in r.points]
        e=[log(norm(x)) for x in r.exterior_points]
        g=fill(SVector(0.,0.,0.),length(p))
        circular_reconstruct!(g,r,p,e)
        terms=circular_diffusion_terms(f,g,1.)
        q=zeros(length(c.measure));circular_diffusion_flux!(q,terms,p)
        exact=-(c.theta_hi-c.theta_lo) # R=1, outward A
        push!(errors,sum(abs,q-exact)/sum(abs,exact))
        balance=zeros(length(p));interface_cell_flux!(balance,c,q)
        @test abs(sum(balance)) < 1e-12
        @test errors[end] < 0.2
    end
    @info "Radial logarithmic flux errors (angular and radial refinement)" errors
    @test all(diff(errors) .< 0)
    @test errors[end] < errors[1]/4
end
