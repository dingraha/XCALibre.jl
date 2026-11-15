module PeriodicCircularQuadraticTests
using Test, XCALibre, LinearAlgebra, StaticArrays, SparseArrays
include(pkgdir(XCALibre,"examples","interface_verification","circular_periodic_diffusion.jl"))
const D=CircularPeriodicDiffusion
function independent(t,p)
    h=t.geometry;coeff=[SVector{5}(s.recovery*(p[s.donors].-p[i])) for (i,s) in enumerate(t.stencils)]
    x,w=D.quadrature(12);q=zeros(length(h.beta))
    function grad(i,y)
        c=coeff[i];s=t.stencils[i].scale;d=y-h.points[i]
        SVector(c[1]/s+(c[3]*d[1]+c[4]*d[2])/s^2,c[2]/s+(c[4]*d[1]+c[5]*d[2])/s^2,0.)
    end
    for k in eachindex(q)
        a,b=h.edge_a[k],h.edge_b[k];weight=h.weight[k]
        blended(y)=weight*grad(a,y)+(1-weight)*grad(b,y-h.shift[k])
        if !iszero(h.face_a[k])
            # Straight-face affine gradient average is exact at face centre.
            q[k]=-t.coefficient*dot(blended(h.point[k]),h.area[k])
        else
            c=h.metrics.coupling;o=h.overlap[k];lo,hi=c.theta_lo[o],c.theta_hi[o]
            for j in eachindex(x)
                theta=lo+x[j]*(hi-lo);normal=SVector(cos(theta),sin(theta),0.)
                q[k]-=t.coefficient*w[j]*(hi-lo)*c.radius*dot(blended(c.origin+c.radius*normal),normal)
            end
        end
    end
    q
end
@testset "Opt-in frozen periodic circular quadratic scalar" begin
    for phase in (0.,.13,1.2), scale in (1,2)
        q=D.CS.build(na=16scale,nb=20scale,nra=3scale,nrb=3scale,phase=phase)
        h=D.geometry(q);t=construct_periodic_circular_quadratic(h);p=D.field.(h.points);saved=copy(p)
        @test t.geometry!==h && t.geometry.points!==h.points
        @test all(s -> 11<=length(s.donors)<=16 && s.condition<100,t.stencils)
        for s in t.stencils, j in 1:5
            data=[XCALibre.InterfaceCoupling._quadratic_basis(d/s.scale)[j] for d in s.displacements]
            target=zeros(5);target[j]=1
            @test maximum(abs,s.recovery*data-target)<1e-10
        end
        for samples in (p,[sin(1.23i)+cos(.71i) for i in eachindex(p)],ones(length(p)))
            r=periodic_circular_quadratic_flux(t,samples)
            @test maximum(abs,r.edges-independent(t,samples))<1e-11
            @test maximum(abs,t.matrix*samples-r.balance)<1e-11
            @test abs(sum(r.balance))<1e-12
            @test r.divergence==r.balance./h.metrics.volume
            for k in eachindex(h.beta)
                iszero(h.face_b[k]) || @test r.faces[h.face_a[k]] == -r.faces[h.face_b[k]]
            end
        end
        @test p==saved
        @test maximum(abs,t.matrix*ones(length(p)))<1e-11
        @test maximum(abs,vec(sum(t.matrix;dims=1)))<1e-11
        scaled=construct_periodic_circular_quadratic(h;coefficient=2.)
        @test scaled.edge_matrix==2t.edge_matrix
        @test scaled.matrix==2t.matrix
        if scale==1
            S=Matrix((t.matrix+t.matrix')/2);n=length(p);P=I-ones(n,n)/n;e=eigvals(Symmetric(P*S*P))
            @test count(x->abs(x)<1e-10*maximum(abs,e),e)==1
            @test e[2]>1e-3
        end
        # Rejected input and caller mutations cannot change frozen action.
        before=periodic_circular_quadratic_flux(t,p)
        @test_throws DimensionMismatch periodic_circular_quadratic_flux(t,p[1:end-1])
        bad=copy(p);bad[1]=NaN
        @test_throws ArgumentError periodic_circular_quadratic_flux(t,bad)
        for coefficient in (0.,-1.,Inf,NaN)
            @test_throws ArgumentError construct_periodic_circular_quadratic(h;coefficient)
        end
        for limit in (0.,101.,Inf,NaN)
            @test_throws ArgumentError construct_periodic_circular_quadratic(h;condition_limit=limit)
        end
        @test_throws ArgumentError construct_periodic_circular_quadratic(h;condition_limit=1.)
        for rows in (4,5,81)
            @test_throws ArgumentError construct_periodic_circular_quadratic(h;max_rows=rows)
        end
        h.points[1]+=SVector(.01,0.,0.)
        @test periodic_circular_quadratic_flux(t,p).edges==before.edges
        @test t.geometry.points[1]!=h.points[1]
        r=periodic_circular_quadratic_flux(t,p);r.edges[1]=123
        @test periodic_circular_quadratic_flux(t,p).edges==before.edges
    end
    q=D.CS.build(L=2.,radius=.5,core_half=.2)
    @test_throws ArgumentError construct_periodic_circular_quadratic(D.geometry(q))
    q=D.CS.build();h=D.geometry(q)
    deficient=deepcopy(h)
    for row in deficient.rows;empty!(row);end
    @test_throws ArgumentError construct_periodic_circular_quadratic(deficient)
    collinear=deepcopy(h)
    for i in eachindex(collinear.points)
        collinear.points[i]=SVector(i/(length(h.points)+1),.5,h.points[i][3])
    end
    for (i,row) in enumerate(collinear.rows)
        empty!(row)
        for j in 1:7
            j==i || push!(row,(j,collinear.points[j]-collinear.points[i]))
        end
    end
    @test_throws ArgumentError construct_periodic_circular_quadratic(collinear)
    shifted=deepcopy(h);k=findfirst(!iszero,h.face_b)
    shifted.shift[k]=SVector(.5,0.,0.)
    @test_throws ArgumentError construct_periodic_circular_quadratic(shifted)
    malformed=deepcopy(h);malformed.points[1]=SVector(NaN,0.,0.)
    @test_throws ArgumentError construct_periodic_circular_quadratic(malformed)
    @test all(isfinite,h.points[1]) # failure atomicity on caller data
    # Matched refinement and independently integrated solved PDE with common gauge.
    errors=Float64[];circle=Float64[];seam=Float64[]
    for scale in (1,2,4)
        q=D.CS.build(na=16scale,nb=16scale,nra=3scale,nrb=3scale)
        h=D.geometry(q);t=construct_periodic_circular_quadratic(h);V=h.metrics.volume;b=D.volume_source(q);n=length(V)
        K=[t.matrix sparse(reshape(V,n,1));sparse(reshape(V,1,n)) spzeros(1,1)]
        solution=K\vcat(b,0.);p=solution[1:n];exact=D.field.(h.points);exact.-=dot(V,exact)/sum(V)
        @test abs(solution[end])*sum(V)<1e-10
        @test maximum(abs,t.matrix*p-b)<1e-10
        @test maximum(abs,periodic_circular_quadratic_flux(t,p).balance-b)<1e-10
        @test abs(dot(V,p))<1e-12
        parts=D.partitions(q,h)
        push!(errors,D.norms(p-exact,V,parts.all).rms)
        push!(circle,D.norms(p-exact,V,parts.circle).rms)
        push!(seam,D.norms(p-exact,V,parts.seam).rms)
        if scale==4
            @test sqrt(dot(V,(p-exact).^2)/dot(V,exact.^2))<.05
        end
    end
    @test errors[3]<errors[2]<errors[1] && errors[3]/errors[1]<.6
    @test circle[3]<circle[2]<circle[1]
    @test seam[3]<seam[2]<seam[1]
    # No retroactive coarse RMS<.10 assertion: prior rejected threshold preserved
    # in workspace report, not relabelled as passing.
end
end
