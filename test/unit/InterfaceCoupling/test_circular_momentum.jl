using XCALibre, Test, LinearAlgebra, StaticArrays, SparseArrays
isdefined(@__MODULE__,:circular_interface_fixture) || include("test_circular_interface.jl")

function momentum_fixture(;T=Float64,I=Int64,phase=0.13,k=1)
    mesh=circular_interface_fixture(na=8k,nb=11k,phase=phase/k,float=T,integer=I,
        inner_radius=1-0.5/k,outer_radius=1+0.5/k)
    c=circular_coupling(mesh);r=construct_circular_reconstruction(mesh,c)
    h=construct_circular_pressure_geometry(mesh,c,fill(:dirichlet,length(r.exterior_faces)))
    m=construct_circular_momentum_geometry(h,fill(:dirichlet,length(r.exterior_faces)))
    mesh,c,r,h,m
end

@testset "Circular vector momentum assembly and conservation" begin
    for (T,I) in ((Float64,Int64),(Float64,Int32),(Float32,Int32)),phase in (0.,0.13,0.71)
        mesh,c,r,h,m=momentum_fixture(;T,I,phase)
        n=length(r.points); tol=4000eps(T)
        exact(x)=SVector{3,T}(0.4+0.2x[1]-0.3x[2],-0.1+0.6x[1]+0.2x[2],0)
        U=exact.(r.points);v=exact.(r.exterior_points)
        p=T[2+0.7x[1]-0.3x[2] for x in r.points]
        pv=T[2+0.7x[1]-0.3x[2] for x in r.exterior_points]
        gradients=circular_velocity_reconstruct(m,U,v)
        @test all(g -> isapprox(g,SVector{3,T}(0.2,-0.3,0);atol=tol),gradients[1])
        @test all(g -> isapprox(g,SVector{3,T}(0.6,0.2,0);atol=tol),gradients[2])
        for convection in (false,true)
            faces=zeros(T,length(mesh.faces));overlaps=zeros(T,length(c.measure))
            if convection
                faces.=T[isodd(j) ? 0.17 : -0.09 for j in eachindex(faces)]
                overlaps.=T[isodd(j) ? 0.3 : -0.1 for j in eachindex(overlaps)]
            end
            faces[r.metrics.interface_face].=NaN
            terms=circular_momentum_terms(m,U,v,p,pv,faces,overlaps,T(0.1))
            flux=circular_momentum_flux(terms,U)
            for j in 1:2
                @test getindex.(flux.balance,j) ≈ terms.matrix*getindex.(U,j)+getindex.(terms.offset,j) atol=tol
            end
            @test terms.pressure_force ≈ [-r.metrics.volume[i]*SVector{3,T}(0.7,-0.3,0) for i in 1:n] atol=tol
            @test flux.residual ≈ flux.balance-terms.pressure_force atol=tol
            @test sum(flux.balance) ≈ sum(flux.faces[r.exterior_faces]) atol=tol
            for k in eachindex(h.edge_a)
                a,b=h.edge_a[k],h.edge_b[k];j=h.edge_face[k]
                S=h.edge_area[k]
                q=iszero(j) ? overlaps[h.edge_overlap[k]] : faces[j]
                expected=-T(0.1)*SVector(dot(gradients[1][a],S),dot(gradients[2][a],S),zero(T)) + q*(q>=0 ? U[a] : U[b])
                actual=iszero(j) ? flux.overlaps[h.edge_overlap[k]] : flux.faces[j]
                @test actual ≈ expected atol=tol
            end
            shifted=circular_momentum_terms(m,U,v,p .+ T(3),pv .+ T(3),faces,overlaps,T(0.1))
            @test shifted.pressure_force ≈ terms.pressure_force atol=tol
            @test shifted.matrix==terms.matrix
            @test shifted.offset==terms.offset
            if !convection
                @test terms.matrix ≈ terms.matrix' atol=tol
                @test minimum(eigvals(Symmetric(Matrix{Float64}(terms.matrix))))>0
                @test maximum(norm,flux.balance)<tol # affine Laplacian zero
            end
        end
        # Constant Cartesian advection: integrated S creates a complete closed
        # divergence-free flux balance across ordinary faces and circular overlaps.
        constant=SVector{3,T}(0.7,-0.2,0)
        uc=fill(constant,n);vc=fill(constant,length(v))
        predictor=circular_momentum_predictor_flux(m,uc,vc)
        qbalance=XCALibre.InterfaceCoupling._pressure_balance(h,predictor.faces,predictor.overlaps)
        @test maximum(abs,qbalance)<tol
        terms=circular_momentum_terms(m,uc,vc,zeros(T,n),zeros(T,length(v)),predictor.faces,predictor.overlaps,T(0.1))
        @test maximum(norm,circular_momentum_flux(terms,uc).balance)<tol
        @test any(>(0),predictor.overlaps) && any(<(0),predictor.overlaps)
        @test abs(sum(predictor.overlaps))<tol
        @test sum(abs,predictor.overlaps)>1
        # Exercise direct solve and recovery for Float32 and Int32 snapshots too.
        solved=circular_momentum_solve(terms,uc;relaxation=0.8)
        @test solved.velocity ≈ uc atol=tol
        @test solved.H0 ≈ uc atol=tol
        @test maximum(norm,solved.refreshed_residual)<tol
    end
end

@testset "Physical circular momentum predictor sparse-row equivalence" begin
    mesh,c,r,h,m=momentum_fixture()
    n=length(r.points); nb=length(r.exterior_faces)
    U=[SVector(0.4+0.1x[2],0.03x[1],0.) for x in r.points]
    v=[SVector(0.4+0.1x[2],0.03x[1],0.) for x in r.exterior_points]
    p=[x[1]+0.2x[2] for x in r.points];pv=[x[1]+0.2x[2] for x in r.exterior_points]
    faces=[isodd(j) ? 0.3 : -0.1 for j in eachindex(mesh.faces)]
    overlaps=[isodd(j) ? 0.3 : -0.1 for j in eachindex(c.measure)]
    terms=circular_momentum_terms(m,U,v,p,pv,faces,overlaps,0.1)
    A=Matrix(terms.matrix);D=diag(A);off=A-Diagonal(D)
    # Independent dense assembly: constant-viscosity symmetric links plus signed
    # Upwind links, including exterior inflow RHS and owner outflow diagonal.
    independent=zeros(n,n)
    for k in eachindex(h.edge_a)
        a,b=h.edge_a[k],h.edge_b[k];j=h.edge_face[k]
        q=iszero(j) ? overlaps[h.edge_overlap[k]] : faces[j]
        G=0.1h.beta[k];ca=max(q,0);cb=min(q,0)
        independent[a,a]+=G+ca;independent[a,b]+=-G+cb
        independent[b,a]+=-G-ca;independent[b,b]+=G-cb
    end
    for (k,j) in enumerate(r.exterior_faces)
        a=r.metrics.owner_a[j]
        independent[a,a]+=0.1h.boundary_beta[k]+max(faces[j],0)
    end
    @test A ≈ independent atol=1e-13
    @test all(A[a,b]!=0 for (a,b) in zip(c.cell_a,c.cell_b))
    @test any(sum(abs,off;dims=2)[:].>D)
    # Identical trial velocity: relaxation-dependent Hv yields the SAME physical
    # H0 when full augmentation and old-velocity source are undone. This is not
    # a comparison of different solved velocities (which need not agree).
    fixed=[SVector(0.2x[1],-0.3x[2],0.) for x in r.points]
    physicalH=[(-getindex.(terms.offset,j)-off*getindex.(fixed,j))./D for j in 1:2]
    for alpha in (0.5,0.8,1.)
        Dt=max.(abs.(D),sum(abs,off;dims=2)[:])./alpha
        for j in 1:2
            pressurefree_rhs=-getindex.(terms.offset,j)+(Dt-D).*getindex.(U,j)
            Hv=(pressurefree_rhs-off*getindex.(fixed,j))./Dt
            @test (Dt.*Hv-(Dt-D).*getindex.(U,j))./D ≈ physicalH[j] atol=1e-13
        end
        saved=copy(U);result=circular_momentum_solve(terms,U;relaxation=alpha)
        Dt=max.(abs.(D),sum(abs,off;dims=2)[:])./alpha
        @test result.physical_diagonal==D
        @test result.relaxed_diagonal ≈ Dt atol=1e-13
        @test any(Dt.>D) # even alpha=1 includes diagonal-dominance augmentation
        @test result.mobility ≈ r.metrics.volume./D atol=1e-14
        @test maximum(abs,result.mobility-r.metrics.polygon_volume./D)>1e-4
        @test maximum(norm,result.frozen_residual)<1e-12
        @test U==saved
        for j in 1:2
            solved=getindex.(result.velocity,j)
            rhs=getindex.(terms.pressure_force-terms.offset,j)+(Dt-D).*getindex.(U,j)
            @test solved ≈ (A+Diagonal(Dt-D))\rhs atol=1e-12
            reference=(-getindex.(terms.offset,j)-off*solved)./D
            @test getindex.(result.H0,j) ≈ reference atol=1e-12
            Hv=(rhs-getindex.(terms.pressure_force,j)-off*solved)./Dt
            @test getindex.(result.H0,j) ≈ (Dt.*Hv-(Dt-D).*getindex.(U,j))./D atol=1e-12
            physical=A*solved+getindex.(terms.offset-terms.pressure_force,j)
            @test getindex.(result.physical_residual,j) ≈ physical atol=1e-12
            @test physical ≈ -(Dt-D).*(solved-getindex.(U,j)) atol=1e-12
        end
        predictor=circular_momentum_predictor_flux(m,result.H0,v)
        for k in eachindex(h.edge_a)
            a,b,w=h.edge_a[k],h.edge_b[k],h.weight_a[k]
            expected=dot(w*result.H0[a]+(1-w)*result.H0[b],h.edge_area[k])
            actual=h.edge_face[k]==0 ? predictor.overlaps[h.edge_overlap[k]] : predictor.faces[h.edge_face[k]]
            @test actual ≈ expected atol=1e-13
        end
        for (k,j) in enumerate(r.exterior_faces)
            @test predictor.faces[j] ≈ dot(v[k],r.metrics.face_area_vector[j]) atol=1e-13
        end
        # Independently refresh with solved velocities; old pressure force remains
        # unchanged, and fresh viscosity corrections need not match frozen ones.
        refreshed=circular_momentum_terms(m,result.velocity,v,p,pv,faces,overlaps,0.1)
        @test result.refreshed_residual ≈ circular_momentum_flux(refreshed,result.velocity).residual atol=1e-12
        @test maximum(norm,result.refreshed_residual-result.physical_residual)>1e-6
    end
end

@testset "Circular manufactured viscous momentum iteration" begin
    mesh,c,r,h,m=momentum_fixture()
    n=length(r.points)
    target(x)=SVector(0.4+0.2x[1]-0.3x[2],-0.1+0.6x[1]+0.2x[2],0.)
    exact=target.(r.points);v=target.(r.exterior_points)
    faces=zeros(length(mesh.faces));overlaps=zeros(length(c.measure))
    # Independent deferred iterations, explicitly NOT an installed SIMPLE loop.
    for alpha in (0.5,0.8,1.)
        U=fill(SVector(0.,0.,0.),n)
        final=nothing
        for sweep in 1:200
            terms=circular_momentum_terms(m,U,v,zeros(n),zeros(length(v)),faces,overlaps,0.1)
            final=circular_momentum_solve(terms,U;relaxation=alpha)
            U=final.velocity
            maximum(norm,final.refreshed_residual)<1e-11 && break
        end
        @test maximum(norm,final.refreshed_residual)<1e-11
        @test U ≈ exact atol=1e-8
        @test final.H0 ≈ U atol=1e-8 # no pressure, converged physical system
    end
end

@testset "Velocity boundary independence and rejection" begin
    mesh,c,r,h,m=momentum_fixture()
    n=length(r.points); nb=length(r.exterior_faces)
    kinds=fill(:dirichlet,nb);kinds[1]=:neumann
    mixed=construct_circular_momentum_geometry(h,kinds)
    @test mixed.pressure.boundary_kind==fill(:dirichlet,nb)
    normal=normalize(r.metrics.face_area_vector[r.exterior_faces[1]])
    tangent=SVector(-normal[2],normal[1],0.)
    exact(x)=SVector(2+dot(tangent,x),0.3+0.7dot(tangent,x),0.)
    U=exact.(r.points);values=exact.(r.exterior_points);values[1]=SVector(0.,0.,0.)
    gradients=circular_velocity_reconstruct(mixed,U,values)
    @test all(g -> isapprox(g,tangent;atol=1e-12),gradients[1])
    @test all(g -> isapprox(g,0.7tangent;atol=1e-12),gradients[2])
    f=zeros(length(mesh.faces));o=zeros(length(c.measure));j=r.exterior_faces[1]
    f[j]=-0.01
    @test_throws ArgumentError circular_momentum_terms(mixed,U,values,zeros(n),zeros(nb),f,o,0.1)
    f[j]=0.1
    terms=circular_momentum_terms(mixed,U,values,zeros(n),zeros(nb),f,o,0.1)
    flux=circular_momentum_flux(terms,U)
    @test flux.faces[j] ≈ 0.1U[r.metrics.owner_a[j]] atol=1e-12
    predictorU=fill(normal,n)
    @test circular_momentum_predictor_flux(mixed,predictorU,values).faces[j]>0
    @test_throws ArgumentError circular_momentum_predictor_flux(mixed,fill(-normal,n),values)
    alln=construct_circular_momentum_geometry(h,fill(:neumann,nb))
    # It is a legitimate frozen operator, but a singular momentum solve is refused.
    tn=circular_momentum_terms(alln,fill(SVector(0.,0.,0.),n),fill(SVector(0.,0.,0.),nb),zeros(n),zeros(nb),zeros(length(f)),o,0.1)
    @test_throws ArgumentError circular_momentum_solve(tn,fill(SVector(0.,0.,0.),n))
    @test_throws ArgumentError circular_velocity_reconstruct(alln,U,exact.(r.exterior_points))
    @test_throws ArgumentError construct_circular_momentum_geometry(h,fill(:bad,nb))
    @test_throws DimensionMismatch construct_circular_momentum_geometry(h,kinds[1:1])
    @test_throws ArgumentError construct_circular_momentum_geometry(h,kinds;condition_limit=1)
    @test_throws ArgumentError construct_circular_momentum_geometry(h,kinds;condition_limit=NaN)
    # Failed calls leave snapshot and caller fields intact; no continuation state.
    saved=deepcopy((mixed.velocity.boundary_kind,mixed.velocity.inverse,U,values))
    @test_throws ArgumentError circular_momentum_predictor_flux(mixed,fill(-normal,n),values)
    @test (mixed.velocity.boundary_kind,mixed.velocity.inverse,U,values)==saved
end

@testset "Circular momentum validation and snapshot ownership" begin
    mesh,c,r,h,m=momentum_fixture()
    n=length(r.points);nb=length(r.exterior_faces)
    U=fill(SVector(1.,0.,0.),n);v=fill(SVector(1.,0.,0.),nb)
    p=zeros(n);pv=zeros(nb);f=zeros(length(mesh.faces));o=zeros(length(c.measure))
    for nu in (0.,-1.,NaN,Inf)
        @test_throws ArgumentError circular_momentum_terms(m,U,v,p,pv,f,o,nu)
    end
    for alpha in (0.,1.1,NaN,Inf)
        t=circular_momentum_terms(m,U,v,p,pv,f,o,0.1)
        @test_throws ArgumentError circular_momentum_solve(t,U;relaxation=alpha)
    end
    badU=copy(U);badU[1]=SVector(0.,0.,1.)
    @test_throws ArgumentError circular_velocity_reconstruct(m,badU,v)
    @test_throws ArgumentError circular_momentum_terms(m,badU,v,p,pv,f,o,0.1)
    @test_throws ArgumentError circular_momentum_predictor_flux(m,badU,v)
    @test_throws DimensionMismatch circular_velocity_reconstruct(m,U,v[1:1])
    @test_throws DimensionMismatch circular_momentum_terms(m,U,v,p,pv,f,o[1:1],0.1)
    badf=copy(f);badf[r.exterior_faces[1]]=NaN
    @test_throws ArgumentError circular_momentum_terms(m,U,v,p,pv,badf,o,0.1)
    terms=circular_momentum_terms(m,U,v,p,pv,f,o,0.1)
    base=circular_momentum_flux(terms,U);snapshot=deepcopy((U,v,p,pv,f,o))
    result=circular_momentum_solve(terms,U)
    @test (U,v,p,pv,f,o)==snapshot
    @test_throws ArgumentError circular_momentum_flux(terms,badU)
    @test_throws DimensionMismatch circular_momentum_solve(terms,U[1:1])
    m.velocity.beta[1]*=2;h.beta[1]*=2
    v[1]=SVector(99.,0.,0.);f[1]=88;o[1]=77
    @test terms.geometry.velocity.beta[1]!=m.velocity.beta[1]
    @test terms.geometry.pressure.beta[1]!=h.beta[1]
    @test circular_momentum_flux(terms,U).faces==base.faces
    @test circular_momentum_solve(terms,U).H0==result.H0
    mesh32,c32,r32,h32,m32=momentum_fixture(T=Float32)
    n32=length(r32.points);nb32=length(r32.exterior_faces)
    u32=fill(SVector{3,Float32}(0,0,0),n32);v32=fill(SVector{3,Float32}(0,0,0),nb32)
    @test_throws ArgumentError circular_momentum_terms(m32,u32,v32,zeros(n32),zeros(nb32),zeros(length(mesh32.faces)),zeros(length(c32.measure)),1e100)
    huge=fill(SVector(1e100,0.,0.),nb32)
    @test_throws ArgumentError circular_momentum_predictor_flux(m32,u32,huge)
    badp=copy(p);badp[1]=NaN
    @test_throws ArgumentError circular_momentum_terms(m,U,v,badp,pv,zeros(length(f)),zeros(length(o)),0.1)
    @test_throws DimensionMismatch circular_momentum_terms(m,U,v,p,pv[1:1],zeros(length(f)),zeros(length(o)),0.1)
end
