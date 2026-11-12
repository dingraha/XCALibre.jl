using XCALibre, Test, LinearAlgebra, StaticArrays, SparseArrays
isdefined(@__MODULE__,:circular_interface_fixture) || include("test_circular_interface.jl")

function pressure_fixture(;T=Float64,I=Int64,k=1,phase=0.13)
    mesh=circular_interface_fixture(na=8k,nb=11k,phase=phase/k,float=T,integer=I,
        inner_radius=1-0.5/k,outer_radius=1+0.5/k)
    c=circular_coupling(mesh)
    r=construct_circular_reconstruction(mesh,c)
    mesh,c,r
end

@testset "Complete circular pressure traces and variable terms" begin
    for (T,I) in ((Float64,Int64),(Float64,Int32),(Float32,Int32)),phase in (0.,0.13,0.71)
        mesh,c,r=pressure_fixture(;T,I,phase)
        kinds=fill(:dirichlet,length(r.exterior_faces))
        h=construct_circular_pressure_geometry(mesh,c,kinds)
        tol=3000eps(T)
        @test all(>(0),h.beta)
        @test all(w -> 0<w<1,h.weight_a)
        @test sum(length,h.components)==length(r.points)
        for exact in (SVector{3,T}(0,0,0),SVector{3,T}(1,0,0),SVector{3,T}(0.3,-0.7,0))
            p=[T(2)+dot(exact,x) for x in r.points]
            values=[T(2)+dot(exact,x) for x in r.exterior_points]
            g=fill(zero(exact),length(p))
            circular_pressure_reconstruct!(g,h,p,values)
            @test all(x -> isapprox(x,exact;atol=tol,rtol=tol),g)
            circular_pressure_gradient!(g,h,p,values)
            @test all(x -> isapprox(x,exact;atol=tol,rtol=tol),g)
            mob=[T(0.1+0.8*(i%3)) for i in eachindex(p)]
            terms=circular_pressure_terms(h,fill(exact,length(p)),mob,values)
            flux=circular_pressure_flux(terms,p)
            @test flux.balance ≈ terms.matrix*p+terms.offset atol=tol
            @test flux.divergence .* r.metrics.volume ≈ flux.balance atol=tol
            @test terms.matrix ≈ terms.matrix' atol=tol
            @test minimum(eigvals(Symmetric(Matrix{Float64}(terms.matrix)))) > 0
            for k in eachindex(h.edge_a)
                a,b,w=h.edge_a[k],h.edge_b[k],h.weight_a[k]
                expected=1/((1-w)/mob[a]+w/mob[b])
                @test terms.mobility[k] ≈ expected rtol=tol
                @test min(mob[a],mob[b]) <= terms.mobility[k]+tol
                @test terms.mobility[k] <= max(mob[a],mob[b])+tol
                q=-terms.mobility[k]*dot(exact,h.edge_area[k])
                actual=h.edge_face[k]==0 ? flux.overlaps[h.edge_overlap[k]] : flux.faces[h.edge_face[k]]
                @test actual ≈ q atol=tol
            end
            shifted=circular_pressure_terms(h,fill(exact,length(p)),mob,values .+ T(3))
            shiftedflux=circular_pressure_flux(shifted,p .+ T(3))
            @test shiftedflux.faces ≈ flux.faces atol=tol
            @test shiftedflux.overlaps ≈ flux.overlaps atol=tol
            # Constant-mobility manufactured affine pressure solve from predictor.
            base=circular_pressure_terms(h,fill(exact,length(p)),ones(T,length(p)),values)
            trueflux=circular_pressure_flux(base,p)
            pred=-trueflux.faces; pred[r.metrics.interface_face].=NaN
            result=circular_pressure_project(h,ones(T,length(p)),values,pred,-trueflux.overlaps;
                initial=fill(T(-1),length(p)),tolerance=tol,max_sweeps=200)
            @test result.converged
            @test result.pressure ≈ p atol=10tol rtol=10tol
            @test maximum(abs,result.balance)<tol
            @test result.history[end].refreshed<tol
            recovered=circular_pressure_flux(result.terms,result.pressure)
            @test result.overlaps ≈ -trueflux.overlaps+recovered.overlaps atol=tol
            # No input mutation and no incremental correction accumulation.
            repeat=circular_pressure_project(h,ones(T,length(p)),values,pred,-trueflux.overlaps;
                initial=result.pressure,tolerance=tol,max_sweeps=200)
            @test repeat.converged
            @test repeat.faces ≈ result.faces atol=10tol
        end
    end
end

@testset "Neumann constraints, gauges and frozen recovery" begin
    mesh,c,r=pressure_fixture()
    kinds=fill(:neumann,length(r.exterior_faces))
    h=construct_circular_pressure_geometry(mesh,c,kinds)
    n=length(r.points); values=zeros(length(kinds)); p=fill(3.,n)
    g=fill(SVector(9.,9.,9.),n)
    circular_pressure_reconstruct!(g,h,p,values)
    @test all(iszero,g)
    circular_pressure_gradient!(g,h,p,values)
    @test maximum(norm,g)<1e-12
    # Choose one exterior face with an affine field tangent to that face. Change
    # only this boundary to Neumann; the others retain exact Dirichlet data.
    mixed=fill(:dirichlet,length(kinds)); mixed[1]=:neumann
    hm=construct_circular_pressure_geometry(mesh,c,mixed)
    normal=normalize(r.metrics.face_area_vector[r.exterior_faces[1]])
    exact=SVector(-normal[2],normal[1],0.)
    pp=[2+dot(exact,x) for x in r.points]
    vv=[2+dot(exact,x) for x in r.exterior_points];vv[1]=0
    circular_pressure_reconstruct!(g,hm,pp,vv)
    @test all(x -> isapprox(x,exact;atol=1e-12),g)
    circular_pressure_gradient!(g,hm,pp,vv)
    @test all(x -> isapprox(x,exact;atol=1e-12),g)
    tf=circular_pressure_terms(hm,fill(exact,n),ones(n),vv)
    @test circular_pressure_flux(tf,pp).faces[r.exterior_faces[1]]==0
    # All-Neumann manufactured discrete projection (not a continuum solution):
    # varying pressure, zero boundary flux, predictor cancels frozen full operator.
    pp=[sin(x[1])+0.2x[2] for x in r.points]
    circular_pressure_reconstruct!(g,h,pp,values)
    terms=circular_pressure_terms(h,g,ones(n),values)
    pressureflux=circular_pressure_flux(terms,pp)
    pred=-pressureflux.faces; pred[r.metrics.interface_face].=NaN
    o=-pressureflux.overlaps
    savedpred=copy(pred);savedo=copy(o)
    result=circular_pressure_project(h,ones(n),values,pred,o;reference=Dict(1=>pp[1]),max_sweeps=250,tolerance=1e-10)
    @test result.converged
    @test result.pressure ≈ pp atol=1e-8
    @test maximum(abs,result.balance)<1e-10
    @test result.divergence.*r.metrics.volume ≈ result.balance atol=1e-14
    offset=circular_pressure_project(h,ones(n),values,pred,o;reference=Dict(1=>pp[1]+3),max_sweeps=250,tolerance=1e-10)
    @test offset.converged
    @test offset.pressure ≈ result.pressure .+ 3 atol=1e-8
    @test offset.faces ≈ result.faces atol=1e-9
    @test offset.overlaps ≈ result.overlaps atol=1e-9
    @test isequal(pred,savedpred) && o==savedo
    @test norm(terms.matrix*ones(n))<1e-12
    @test sum(pressureflux.balance) ≈ 0 atol=1e-12
    stopped=circular_pressure_project(h,ones(n),values,pred,o;reference=Dict(1=>pp[1]),max_sweeps=1,tolerance=1e-14)
    @test !stopped.converged
    @test stopped.history[1].continuity<1e-12
    @test stopped.history[1].refreshed>1e-8
    frozen=circular_pressure_flux(terms,pp)
    fill!(g,SVector(99.,99.,0.))
    @test circular_pressure_flux(terms,pp).faces==frozen.faces
    @test_throws ArgumentError circular_pressure_project(h,ones(n),values,pred,o)
    @test_throws ArgumentError circular_pressure_project(h,ones(n),values,pred,o;reference=Dict(1=>0.,2=>0.))
    @test_throws ArgumentError circular_pressure_project(h,ones(n),values,pred,o;reference=Dict(n+1=>0.))
    imbalanced=copy(pred);imbalanced[r.exterior_faces[1]]=1
    @test_throws ArgumentError circular_pressure_project(h,ones(n),values,imbalanced,o;reference=Dict(1=>0.))
    @test_throws ArgumentError circular_pressure_project(hm,ones(n),vv,pred,o;reference=Dict(1=>0.))
    # Component compatibility is local, not a global cancellation test.
    IC=XCALibre.InterfaceCoupling
    @test IC._pressure_components(5,[1,3],[2,4]) == [[1,2],[3,4],[5]]
    components=deepcopy(h.components);empty!(h.components);append!(h.components,[[1],[i for i in 2:n]])
    pb=zeros(n);pb[1]=1;pb[2]=-1
    @test_throws ArgumentError IC._pressure_gauges(h,pb,Dict(1=>0.,2=>0.),2.,1e-12)
    @test length(IC._pressure_gauges(h,zeros(n),Dict(1=>0.,2=>3.),1.,1e-12))==2
    # Large flux scale in a different component must not conceal incompatibility.
    local_scale=fill(1e20,n);local_scale[1]=1
    @test_throws ArgumentError IC._pressure_gauges(h,pb,Dict(1=>0.,2=>0.),local_scale,1e-12)
    empty!(h.components);append!(h.components,components)
    # Independent disconnected sparse solve: two unanchored components and one
    # anchored component. Check original, ungauged equations and gauge changes.
    A=sparse([2. -2. 0. 0. 0.;-2. 2. 0. 0. 0.;0. 0. 3. -3. 0.;0. 0. -3. 3. 0.;0. 0. 0. 0. 4.])
    rhs=[1.,-1.,2.,-2.,12.]
    sol=IC._pressure_solve(A,rhs,Dict(1=>0.,3=>2.))
    @test A*sol ≈ rhs atol=1e-12
    @test sol[[1,3,5]] ≈ [0.,2.,3.]
    shifted=IC._pressure_solve(A,rhs,Dict(1=>4.,3=>-1.))
    @test A*shifted ≈ rhs atol=1e-12
    @test shifted-sol ≈ [4.,4.,-3.,-3.,0.] atol=1e-12
end

@testset "Pressure primitive validation and series resistance" begin
    mesh,c,r=pressure_fixture()
    kinds=fill(:dirichlet,length(r.exterior_faces));h=construct_circular_pressure_geometry(mesh,c,kinds)
    n=length(r.points); p=ones(n);v=ones(length(kinds));g=fill(SVector(0.,0.,0.),n)
    @test_throws ArgumentError construct_circular_pressure_geometry(mesh,c,fill(:bad,length(kinds)))
    @test_throws DimensionMismatch construct_circular_pressure_geometry(mesh,c,kinds[1:1])
    @test_throws ArgumentError construct_circular_pressure_geometry(mesh,c,kinds;condition_limit=1)
    @test_throws ArgumentError construct_circular_pressure_geometry(mesh,c,kinds;min_alignment=0)
    for mobility in (fill(0.,n),fill(-1.,n),fill(NaN,n),fill(Inf,n))
        @test_throws ArgumentError circular_pressure_terms(h,g,mobility,v)
    end
    @test_throws DimensionMismatch circular_pressure_terms(h,g,ones(1),v)
    badg=copy(g);badg[1]=SVector(0.,0.,1.)
    @test_throws ArgumentError circular_pressure_terms(h,badg,ones(n),v)
    badp=copy(p);badp[1]=NaN
    output=fill(SVector(9.,9.,9.),n);saved=copy(output)
    @test_throws ArgumentError circular_pressure_reconstruct!(output,h,badp,v)
    @test output==saved
    @test_throws ArgumentError circular_pressure_gradient!(output,h,badp,v)
    @test output==saved
    @test_throws ArgumentError circular_pressure_gradient!(h.reconstruction.points,h,p,v)
    @test_throws ArgumentError circular_pressure_reconstruct!(h.reconstruction.points,h,p,v)
    hn=construct_circular_pressure_geometry(mesh,c,fill(:neumann,length(kinds)))
    @test_throws ArgumentError circular_pressure_reconstruct!(output,hn,p,v)
    @test_throws ArgumentError circular_pressure_project(h,ones(n),v,zeros(length(mesh.faces)),zeros(length(c.measure));max_sweeps=0)
    @test_throws ArgumentError circular_pressure_project(h,ones(n),v,zeros(length(mesh.faces)),zeros(length(c.measure));tolerance=NaN)
    # Snapshot ownership, boundary/mobility overflow and output conversion.
    mobility=ones(n);terms=circular_pressure_terms(h,g,mobility,v)
    frozen=copy(terms.matrix);frozen_values=copy(terms.boundary_values)
    mobility[1]=99;v[1]=88;h.beta[1]*=2
    @test terms.matrix==frozen && terms.boundary_values==frozen_values
    @test terms.geometry.beta[1]!=h.beta[1]
    @test_throws ArgumentError circular_pressure_flux(terms,badp)
    @test_throws DimensionMismatch circular_pressure_flux(terms,p[1:1])
    @test_throws DimensionMismatch circular_pressure_project(h,ones(n),v,zeros(1),zeros(length(c.measure)))
    @test_throws ArgumentError circular_pressure_project(h,ones(n),v,fill(NaN,length(mesh.faces)),zeros(length(c.measure)))
    mesh32,c32,r32=pressure_fixture(T=Float32)
    h32=construct_circular_pressure_geometry(mesh32,c32,fill(:dirichlet,length(r32.exterior_faces)))
    g32=fill(SVector{3,Float32}(0,0,0),length(r32.points))
    @test_throws ArgumentError circular_pressure_terms(h32,g32,fill(1e100,length(r32.points)),ones(length(r32.exterior_faces)))
    @test_throws ArgumentError circular_pressure_terms(h32,g32,ones(length(r32.points)),fill(1e100,length(r32.exterior_faces)))
    hugep=[1e100*x[1] for x in r.points];hugev=[1e100*x[1] for x in r.exterior_points]
    small=fill(SVector{3,Float32}(9,9,9),n)
    @test_throws ArgumentError circular_pressure_reconstruct!(small,h,hugep,hugev)
    @test all(==(SVector{3,Float32}(9,9,9)),small)
    @test_throws ArgumentError circular_pressure_gradient!(small,h,hugep,hugev)
    @test all(==(SVector{3,Float32}(9,9,9)),small)
    # Orthogonal series resistance, including huge mobility ratios. This is a
    # local constitutive check, not general curved discontinuous-field accuracy.
    for (da,db,ma,mb) in ((0.2,0.8,1.,100.),(0.8,0.2,100.,1.),(1.,2.,1e-200,1e200))
        w=db/(da+db)
        mf=XCALibre.InterfaceCoupling._pressure_harmonic(ma,mb,w)
        G=mf/(da+db)
        @test G ≈ 1/(da/ma+db/mb) rtol=1e-14
        delta=2.
        q=G*delta
        # Avoid catastrophic cancellation in the high-contrast interface trace.
        interfacep=q*db/mb
        if max(ma,mb)/min(ma,mb)<1e6
            @test q ≈ ma*(delta-interfacep)/da rtol=1e-12
            @test q ≈ mb*interfacep/db rtol=1e-12
        else
            @test isfinite(q) && q>0
            @test q*da/ma ≈ delta rtol=1e-14
        end
    end
end

@testset "Variable-mobility manufactured discrete projection" begin
    mesh,c,r=pressure_fixture()
    h=construct_circular_pressure_geometry(mesh,c,fill(:dirichlet,length(r.exterior_faces)))
    n=length(r.points)
    p=[sin(x[1])+0.3x[2]^2 for x in r.points]
    v=[sin(x[1])+0.3x[2]^2 for x in r.exterior_points]
    for mob in ([0.3+norm(x)^2 for x in r.points],
                [norm(x)<1 ? 1. : 10. for x in r.points])
        g=fill(SVector(0.,0.,0.),n)
        circular_pressure_reconstruct!(g,h,p,v)
        terms=circular_pressure_terms(h,g,mob,v)
        target=circular_pressure_flux(terms,p)
        result=circular_pressure_project(h,mob,v,-target.faces,-target.overlaps;
            max_sweeps=250,tolerance=1e-10)
        @test result.converged
        @test result.pressure ≈ p atol=1e-8
        @test maximum(abs,result.balance)<1e-10
        @test result.history[end].refreshed<1e-10
        @test minimum(result.terms.mobility)>0
        # A discrete manufactured RHS proves operator consistency and recovery,
        # not physical gradient-jump accuracy of the cross-interface LSQ stencil.
    end
end

@testset "Standalone radial pressure solve refinement" begin
    errors=Float64[]
    for k in (1,2,4)
        mesh,c,r=pressure_fixture(;k)
        h=construct_circular_pressure_geometry(mesh,c,fill(:dirichlet,length(r.exterior_faces)))
        values=log.(norm.(r.exterior_points))
        result=circular_pressure_project(h,ones(length(r.points)),values,zeros(length(mesh.faces)),zeros(length(c.measure));
            max_sweeps=300,tolerance=1e-11)
        @test result.converged
        @test maximum(abs,result.balance)<1e-11
        push!(errors,maximum(abs,result.pressure-log.(norm.(r.points))))
        # Continuum unit-radius flux is -dtheta; report discretisation error.
        @test sum(abs,result.overlaps+(c.theta_hi-c.theta_lo))/2pi < 0.15
    end
    @info "Standalone radial pressure max errors" errors
    @test all(diff(errors).<0)
    @test errors[end]<errors[1]/4
end
