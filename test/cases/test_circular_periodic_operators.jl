module CircularPeriodicOperatorTests
using Test, XCALibre, StaticArrays, LinearAlgebra, SparseArrays, Random
include(pkgdir(XCALibre,"examples","interface_verification","circular_square_setup.jl"))
const CS=CircularSquareGeometry
pairs(q) = [(f,g,-bc.value.transform.distance) for bc in q.periodic if bc isa PeriodicParent for (f,g) in zip(bc.IDs_range,bc.value.face_map)]

@testset "Static periodic circular scalar operators" begin
    rng=MersenneTwister(1701)
    errors=Dict{Tuple{Bool,Float64},Vector{Tuple{Float64,Float64}}}()
    for scale in (1,2,4), (matched,phase) in ((true,0.),(false,0.),(false,.13),(false,1.2))
        q=CS.build(na=16scale,nb=(matched ? 16 : 20)*scale,nra=3scale,nrb=3scale,phase=phase)
        h=construct_periodic_circular_geometry(q.mesh,q.coupling,pairs(q);condition_limit=100.)
        n=length(h.points); zero_g=fill(SVector(0.,0.,0.),n); g=copy(zero_g)
        @test all(isfinite,h.beta) && all(>(0),h.beta)
        @test all(w -> 0<w<1,h.weight)
        @test all(row -> length(row)<=12,h.rows) # bounded sampled controls, not universal limit
        @test maximum(cond, h.inverse)<100
        periodic_circular_reconstruct!(g,h,fill(3.,n))
        @test maximum(norm,g)==0
        periodic_circular_gradient!(g,h,fill(3.,n))
        @test maximum(norm,g)<5e-11
        # Independent geometry moment/reproduction check on image coordinates.
        # A global affine scalar is not periodic: feed per-row affine differences,
        # not inconsistent global affine point data across the seam.
        for row_i in eachindex(h.rows)
            target=SVector(1.3,-.7); rhs=SVector(0.,0.)
            for (_,d3) in h.rows[row_i]
                d=SVector(d3[1],d3[2]);rhs+=dot(target,d)*d/dot(d,d)
            end
            @test h.inverse[row_i]*rhs ≈ target atol=1e-13
        end
        p=randn(rng,n); arbitrary_g=[SVector(randn(rng),randn(rng),0.) for _ in 1:n]
        t=periodic_circular_terms(h,arbitrary_g,.03)
        f=periodic_circular_flux(t,p)
        @test maximum(abs,t.matrix*p+t.offset-f.balance)<1e-13
        @test abs(sum(f.balance))<1e-13
        @test maximum(abs,t.matrix*ones(n))<1e-13
        @test maximum(abs,t.matrix-t.matrix')==0
        @test dot(p,t.matrix*p) >= -1e-13
        @test norm(t.matrix*p- (periodic_circular_terms(h,zero_g,.03).matrix*p))==0
        for k in eachindex(h.beta)
            if h.face_b[k]!=0
                @test f.faces[h.face_a[k]] == -f.faces[h.face_b[k]]
                @test norm(h.points[h.edge_b[k]]+h.shift[k]-h.points[h.edge_a[k]]) < .3/scale
            end
        end
        # Independent affine diffusion identity in each edge's image frame.
        # This tests the split, not a globally affine periodic scalar.
        target=SVector(1.3,-.7,0.)
        for k in eachindex(h.beta)
            d=h.points[h.edge_b[k]]+h.shift[k]-h.points[h.edge_a[k]]
            flux=-.03*(h.beta[k]*dot(target,d)+dot(target,h.remainder[k]))
            @test flux ≈ -.03*dot(target,h.area[k]) atol=2e-15
        end
        # Full independent face/overlap gather, never counting interface diagnostics twice.
        balance=zeros(n); interface_cell_flux!(balance,h.metrics.coupling,f.overlaps)
        for j in eachindex(f.faces)
            h.metrics.interface_face[j] && continue
            a,b=h.metrics.owner_a[j],h.metrics.owner_b[j]
            balance[a]+=f.faces[j];a==b || (balance[b]-=f.faces[j])
        end
        @test maximum(abs,balance-f.balance)<1e-13
        fg=periodic_circular_flux(t,p.+11.)
        @test maximum(abs,fg.edges-f.edges)<1e-13
        # Smooth truly periodic oracle crossing both circle and square seams.
        field(x)=sin(2pi*x[1])*cos(2pi*x[2])
        grad(x)=SVector(2pi*cos(2pi*x[1])*cos(2pi*x[2]),-2pi*sin(2pi*x[1])*sin(2pi*x[2]),0.)
        p=field.(h.points); exact=grad.(h.points)
        periodic_circular_reconstruct!(g,h,p)
        lsq=sqrt(sum(h.metrics.volume.*sum.(abs2,g.-exact))/sum(h.metrics.volume))
        periodic_circular_gradient!(g,h,p)
        gauss=sqrt(sum(h.metrics.volume.*sum.(abs2,g.-exact))/sum(h.metrics.volume))
        push!(get!(errors,(matched,phase),Tuple{Float64,Float64}[]),(lsq,gauss))
        @test lsq<2.0 && gauss<2.0 # fixed before execution; refinement gates below
        println("periodic scale=$scale matched=$matched phase=$phase LSQ=$lsq Gauss=$gauss rows=$(maximum(length,h.rows)) condition=$(maximum(cond,h.inverse))")
        # Frozen diffusion exact edge sign/split with analytic gradients.
        t=periodic_circular_terms(h,exact,1.)
        flux=periodic_circular_flux(t,p)
        @test all(isfinite,flux.divergence)
        if scale==1 && matched
            @test length(XCALibre.InterfaceCoupling._pressure_components(n,h.edge_a,h.edge_b))==1
            @test count(x -> abs(x)<1e-10,eigvals(Symmetric(Matrix(t.matrix))))==1
            @test_throws ArgumentError construct_periodic_circular_geometry(q.mesh,q.coupling,pairs(q)[2:end])
            @test_throws ArgumentError construct_periodic_circular_geometry(q.mesh,q.coupling,vcat(pairs(q),pairs(q)[1:1]))
            wrong=copy(pairs(q)); a,b,s=wrong[1];wrong[1]=(a,b,-s)
            @test_throws ArgumentError construct_periodic_circular_geometry(q.mesh,q.coupling,wrong)
            @test_throws ArgumentError construct_periodic_circular_geometry(q.mesh,q.coupling,pairs(q);condition_limit=1.)
            saved=copy(g);bad=copy(p);bad[1]=NaN
            @test_throws ArgumentError periodic_circular_reconstruct!(g,h,bad)
            @test g==saved
            @test_throws ArgumentError periodic_circular_reconstruct!(h.points,h,p)
            @test_throws ArgumentError periodic_circular_terms(h,exact,0.)
            @test_throws DimensionMismatch periodic_circular_reconstruct!(g,h,p[2:end])
            snap=deepcopy(h); periodic_circular_reconstruct!(g,h,p)
            @test h.points==snap.points && h.rows==snap.rows && h.inverse==snap.inverse
            @test t.geometry.points==h.points
            # Frozen terms own their geometry; changed mesh geometry requires rebuild.
            old=copy(q.mesh.cell_centre)
            q.mesh.cell_centre[1]+=SVector(.001,0.,0.)
            rebuilt=construct_periodic_circular_geometry(q.mesh,q.coupling,pairs(q))
            @test h.points==old && t.geometry.points==old
            @test rebuilt.points[1]!=h.points[1]
            q.mesh.cell_centre[1]=old[1]
        end
    end
    for (key,e) in sort!(collect(errors))
        @test e[2][1]<e[1][1] && e[3][1]<e[2][1]
        @test e[2][2]<e[1][2] && e[3][2]<e[2][2]
        @test e[3][1]<.6e[1][1] && e[3][2]<.6e[1][2]
    end
end

@testset "Stitched ordinary no-interface geometry control" begin
    for scale in (1,2,4)
        q=CS.build(na=16scale,nb=16scale,nra=3scale,nrb=3scale,stitched=true)
        m=q.mesh
        @test q.coupling===nothing && q.metrics===nothing
        @test length(m.boundaries)==4
        @test sum(m.cell_volume)≈q.L^2 atol=1e-13
        @test minimum(m.cell_volume)>0
        rows=[Int[] for _ in m.cells]
        for (a,b) in m.face_ownerCells
            if a!=b;push!(rows[a],b);push!(rows[b],a);end
        end
        @test length(XCALibre.InterfaceCoupling._pressure_components(length(m.cells),
            first.(m.face_ownerCells),last.(m.face_ownerCells)))==1
        # Core/outer seam is ordinary chord geometry; matched split exact arcs
        # are NOT identical per-cell volumes. Centres and connectivity agree.
        split=CS.build(na=16scale,nb=16scale,nra=3scale,nrb=3scale)
        @test maximum(norm.(m.cell_centre.-split.mesh.cell_centre))<1e-14
        @test length(m.faces)==length(split.mesh.faces)-q.na
        @test maximum(abs,m.cell_volume-split.metrics.volume)>0
        nodes=Set(m.cell_nodes)
        edges=Set(minmax(m.face_nodes[r]...) for r in m.face_nodes_range)
        @test length(nodes)-length(edges)+length(m.cells)==1
        for bc in q.periodic
            bc isa PeriodicParent || continue
            for (f,g) in zip(bc.IDs_range,bc.value.face_map)
                @test m.face_centre[f]+bc.value.transform.distance≈m.face_centre[g] atol=1e-14
            end
        end
    end
    @test_throws ArgumentError CS.build(stitched=true)
    @test_throws ArgumentError CS.build(na=16,nb=16,phase=.1,stitched=true)
end
end
