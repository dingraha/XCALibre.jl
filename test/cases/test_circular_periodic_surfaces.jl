using Test, XCALibre, StaticArrays, LinearAlgebra
if !isdefined(@__MODULE__, :CircularPeriodicDiffusion)
    include(joinpath(@__DIR__,"../../examples/interface_verification/circular_periodic_diffusion.jl"))
end
let
    CD=CircularPeriodicDiffusion;IC=XCALibre.InterfaceCoupling
    nodes,weights=CD.quadrature(16)
    # Independent surface parameterisations use actual mesh endpoints, not the
    # production normal-area tensors or trig primitive.
    function quadrature(q,h,k,f)
        if !iszero(h.face_a[k])
            m=q.mesh;A,B=m.node_coords[m.face_nodes[m.face_nodes_range[h.face_a[k]]]]
            sum(weights[j]*f((1-nodes[j])*A+nodes[j]*B,h.area[k]) for j in eachindex(nodes))
        else
            c=q.coupling;o=h.overlap[k];lo,hi=c.theta_lo[o],c.theta_hi[o]
            radial=SVector(cos((lo+hi)/2),sin((lo+hi)/2),0.)
            sign=dot(h.area[k],radial)>0 ? 1. : -1.
            sum(eachindex(nodes)) do j
                theta=lo+nodes[j]*(hi-lo);n=SVector(cos(theta),sin(theta),0.)
                weights[j]*f(c.origin+c.radius*n,sign*c.radius*(hi-lo)*n)
            end
        end
    end
    @testset "Periodic circular quadratic surfaces" begin
        for phase in (0.,.13,1.2)
            q=CD.CS.build(;na=16,nb=20,nra=3,nrb=3,phase)
            h=CD.geometry(q);t=construct_periodic_circular_quadratic(h)
            s=construct_periodic_circular_surfaces(t);h=s.quadratic.geometry
            for k in eachindex(h.beta)
                F=h.point[k]
                M=quadrature(q,h,k,(x,n)->(x-F)*n')
                @test norm(M-s.first[k])<1e-12
                for j in 1:3
                    N=quadrature(q,h,k,(x,n)->(x-F)*(x-F)'*n[j])
                    @test norm(N-s.second[k][j])<1e-12
                end
                # LOCAL owner-image polynomial probes. No global affine scalar
                # is asserted periodic: samples are generated in each stencil.
                for (i,is_a) in ((h.edge_a[k],true),(h.edge_b[k],false))
                    st=s.quadratic.stencils[i]
                    centre=h.points[i]+(is_a ? zero(h.shift[k]) : h.shift[k])
                    for basis in 1:5
                        vals=[IC._quadratic_basis(d/st.scale)[basis] for d in st.displacements]
                        recovered=st.recovery*vals
                        @test norm(recovered-SVector{5}(ntuple(j->j==basis ? 1. : 0.,5)))<1e-11
                        for j in 1:3
                            functional=IC._surface_functional(h,k,i,st.scale,is_a,s.first[k],s.second[k],j)
                            oracle=quadrature(q,h,k,(x,n)->IC._quadratic_basis((x-centre)/st.scale)[basis]*n[j])
                            @test abs(dot(functional,recovered)-oracle)<1e-11
                        end
                    end
                end
            end
            p=CD.field.(h.points);trace=periodic_circular_surface_trace(s,p)
            # Independent reconstruction evaluated at quadrature points, includes
            # owners' constants, image shifts and all five recovered coefficients.
            coefficients=[st.recovery*(p[st.donors].-p[i]) for (i,st) in enumerate(s.quadratic.stencils)]
            for k in eachindex(h.beta)
                oracle=quadrature(q,h,k,(x,n)->begin
                    value=0.
                    for (i,w,is_a) in ((h.edge_a[k],h.weight[k],true),(h.edge_b[k],1-h.weight[k],false))
                        st=s.quadratic.stencils[i]
                        centre=h.points[i]+(is_a ? zero(h.shift[k]) : h.shift[k])
                        value+=w*(p[i]+dot(coefficients[i],IC._quadratic_basis((x-centre)/st.scale)))
                    end
                    value*n
                end)
                @test norm(trace[k]-oracle)<1e-11
            end
            g=periodic_circular_surface_gradient(s,p)
            @test norm(sum(g.*h.metrics.volume))<1e-12
            @test maximum(norm,periodic_circular_surface_gradient(s,ones(length(p))))<1e-10
            U=fill(SVector(1.,-.7,0.),length(p));r=periodic_circular_surface_flux(s,U)
            @test maximum(abs,r.divergence)<1e-10
            @test maximum(abs,r.edges-[dot(u,U[1]) for u in h.area])<1e-12
            U=[SVector(CD.field(x),-cos(2pi*x[1])*sin(2pi*x[2]),0.) for x in h.points]
            r=periodic_circular_surface_flux(s,U)
            @test abs(sum(r.balance))<1e-12
            @test r.edges≈[a[1]+b[2] for (a,b) in zip(periodic_circular_surface_trace(s,[u[1] for u in U]),periodic_circular_surface_trace(s,[u[2] for u in U]))] atol=1e-12
            for k in eachindex(r.edges)
                if !iszero(h.face_b[k])
                    @test r.faces[h.face_a[k]] == -r.faces[h.face_b[k]]
                end
            end
            @test_throws DimensionMismatch periodic_circular_surface_trace(s,p[1:end-1])
            @test_throws ArgumentError periodic_circular_surface_trace(s,fill(NaN,length(p)))
            @test_throws ArgumentError periodic_circular_surface_flux(s,fill(SVector(0.,0.,1.),length(p)))
            @test_throws ArgumentError periodic_circular_surface_flux(s,fill(SVector(0.,NaN,0.),length(p)))
            @test_throws DimensionMismatch periodic_circular_surface_flux(s,U[1:end-1])
            # Ownership: neither source snapshot nor returned arrays are aliases.
            old=copy(trace);t.geometry.points[1]+=SVector(.001,0.,0.)
            @test periodic_circular_surface_trace(s,p)==old
            trace[1]=SVector(0.,0.,0.)
            @test periodic_circular_surface_trace(s,p)==old
            @test s.quadratic.geometry!==t.geometry
            other=construct_periodic_circular_surfaces(construct_periodic_circular_quadratic(CD.geometry(q);coefficient=3.))
            @test periodic_circular_surface_trace(other,p)≈old atol=1e-12
            oscillatory=[sin(17i)+cos(11i) for i in eachindex(p)]
            ot=periodic_circular_surface_trace(s,oscillatory)
            oc=[st.recovery*(oscillatory[st.donors].-oscillatory[i]) for (i,st) in enumerate(s.quadratic.stencils)]
            for k in eachindex(h.beta)
                independent=quadrature(q,h,k,(x,n)->begin
                    value=0.
                    for (i,w,is_a) in ((h.edge_a[k],h.weight[k],true),(h.edge_b[k],1-h.weight[k],false))
                        st=s.quadratic.stencils[i]
                        centre=h.points[i]+(is_a ? zero(h.shift[k]) : h.shift[k])
                        value+=w*(oscillatory[i]+dot(oc[i],IC._quadratic_basis((x-centre)/st.scale)))
                    end
                    value*n
                end)
                @test norm(ot[k]-independent)<1e-11
            end
        end
    end
end
