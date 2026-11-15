module CircularPeriodicDiffusion
using XCALibre, StaticArrays, LinearAlgebra, SparseArrays
include("circular_square_setup.jl")
const CS = CircularSquareGeometry
const IC = XCALibre.InterfaceCoupling

field(x) = sin(2pi*x[1])*cos(2pi*x[2])
gradient(x) = SVector(2pi*cos(2pi*x[1])*cos(2pi*x[2]),-2pi*sin(2pi*x[1])*sin(2pi*x[2]),0.)
source(x) = 8pi^2*field(x)
pairs(q) = [(f,g,-bc.value.transform.distance) for bc in q.periodic if bc isa PeriodicParent for (f,g) in zip(bc.IDs_range,bc.value.face_map)]

# Diagnostic-only ordinary chord control. Applies the SAME explicitly stated
# corrected two-point scalar formula, not the stock XCALibre Laplacian assembly.
# No fake CircularInterface/metrics, analytical ghosts or source projection.
function chord_geometry(q;condition_limit=100.,min_alignment=.05)
    m=q.mesh; points=copy(m.cell_centre)
    a,b,fa,fb=Int[],Int[],Int[],Int[]
    shifts,area,point,rem=[SVector{3,Float64}[] for _ in 1:4]
    beta,w=Float64[],Float64[]
    function add(u,v,j,k,s)
        S=m.face_area[j]*m.face_normal[j]; x=m.face_centre[j]
        B,R,W=IC._pressure_split(S,points[v]+s-points[u],points[u],x;min_alignment,radius=q.radius)
        push!(a,u);push!(b,v);push!(fa,j);push!(fb,k);push!(shifts,s)
        push!(area,S);push!(point,x);push!(rem,R);push!(beta,B);push!(w,W)
    end
    for j in eachindex(m.face_ownerCells)
        u,v=m.face_ownerCells[j];u==v || add(u,v,j,0,SVector(0.,0.,0.))
    end
    for (j,k,s) in pairs(q)
        add(m.face_ownerCells[j][1],m.face_ownerCells[k][1],j,k,s)
    end
    rows=[Tuple{Int,SVector{3,Float64}}[] for _ in points]
    for k in eachindex(a)
        push!(rows[a[k]],(b[k],points[b[k]]+shifts[k]-points[a[k]]))
        push!(rows[b[k]],(a[k],points[a[k]]-shifts[k]-points[b[k]]))
    end
    foreach(unique!,rows)
    inverse=SMatrix{2,2,Float64,4}[]
    for (i,row) in enumerate(rows)
        G=zero(SMatrix{2,2,Float64,4})
        for (_,d3) in row
            d=SVector(d3[1],d3[2]);G+=d*d'/dot(d,d)
        end
        push!(inverse,IC._circular_inverse(G,condition_limit,i))
    end
    (;points,edge_a=a,edge_b=b,face_a=fa,face_b=fb,shift=shifts,area,point,beta,remainder=rem,weight=w,rows,inverse,volume=copy(m.cell_volume))
end
geometry(q) = q.stitched ? chord_geometry(q) : construct_periodic_circular_geometry(q.mesh,q.coupling,pairs(q);condition_limit=100.)
volumes(h::PeriodicCircularGeometry)=h.metrics.volume
volumes(h)=h.volume
function reconstruct(h,p)
    g=fill(SVector(0.,0.,0.),length(p))
    if h isa PeriodicCircularGeometry
        periodic_circular_reconstruct!(g,h,p)
    else
        for i in eachindex(p)
            rhs=SVector(0.,0.)
            for (j,d3) in h.rows[i]
                d=SVector(d3[1],d3[2]);rhs+=(p[j]-p[i])*d/dot(d,d)
            end
            v=h.inverse[i]*rhs;g[i]=SVector(v[1],v[2],0.)
        end
    end
    g
end
function control_flux(h,p,g)
    q=[h.beta[k]*(p[h.edge_a[k]]-p[h.edge_b[k]])-dot(h.weight[k]*g[h.edge_a[k]]+(1-h.weight[k])*g[h.edge_b[k]],h.remainder[k]) for k in eachindex(h.beta)]
    (;edges=q,balance=gather(h,q))
end
function gather(h,flux)
    out=zeros(length(h.points))
    for k in eachindex(flux)
        out[h.edge_a[k]]+=flux[k];out[h.edge_b[k]]-=flux[k]
    end
    out
end

# Gauss-Legendre quadrature, independently generated on [0,1].
function quadrature(n)
    e=eigen(SymTridiagonal(zeros(n),[j/sqrt(4j^2-1) for j in 1:n-1]))
    ((e.values.+1)./2, e.vectors[1,:].^2)
end

"""Independent boundary flux oracle: straight ordinary chords plus exact circle
parameterisation. Quadrature is numerical, not claimed symbolic exact integration.
Use the divergence theorem to compare against a separate volume quadrature below.
"""
function oracle_edges(q,h;nquad=12)
    x,w=quadrature(nquad); m=q.mesh
    flux=zeros(length(h.beta))
    for k in eachindex(flux)
        j=h.face_a[k]
        if j!=0
            ns=m.face_nodes[m.face_nodes_range[j]];A,B=m.node_coords[ns]
            flux[k]=-sum(w[t]*dot(gradient((1-x[t])*A+x[t]*B),h.area[k]) for t in eachindex(x))
        else
            c=q.coupling; o=h.overlap[k]; lo,hi=c.theta_lo[o],c.theta_hi[o]
            for t in eachindex(x)
                theta=lo+x[t]*(hi-lo); radial=SVector(cos(theta),sin(theta),0.)
                flux[k]-=w[t]*(hi-lo)*c.radius*dot(gradient(c.origin+c.radius*radial),radial)
            end
        end
    end
    flux
end

"""Independent integrated source: bilinear quad integration on polygons,
plus signed circular-segment integrals replacing each interface chord once.
Source is sampled inside the geometry, not evaluated at the cell centre.
"""
function volume_source(q;nquad=12,integrand=source)
    m=q.mesh; x,w=quadrature(nquad); out=zeros(length(m.cells))
    for i in eachindex(out)
        p=m.node_coords[m.cell_nodes[m.cell_nodes_range[i]]]
        for s in eachindex(x),t in eachindex(x)
            u,v=x[s],x[t]
            y=(1-u)*(1-v)*p[1]+u*(1-v)*p[2]+u*v*p[3]+(1-u)*v*p[4]
            du=(1-v)*(p[2]-p[1])+v*(p[3]-p[4])
            dv=(1-u)*(p[4]-p[1])+u*(p[3]-p[2])
            out[i]+=w[s]*w[t]*cross(du,dv)[3]*integrand(y)
        end
    end
    if !q.stitched
        c=q.coupling
        for (patch,sign) in ((c.patch_a,1.),(c.patch_b,-1.))
            for f in m.boundaries[patch].IDs_range
                ns=m.face_nodes[m.face_nodes_range[f]];A,B=m.node_coords[ns]
                lo=atan(A[2]-c.origin[2],A[1]-c.origin[1]); hi=atan(B[2]-c.origin[2],B[1]-c.origin[1])
                d=mod(hi-lo,2pi)
                if d>pi;lo=hi;d=2pi-d;end
                mid=lo+d/2; i=m.face_ownerCells[f][1]
                for s in eachindex(x),t in eachindex(x)
                    theta=lo+x[s]*d; radial=SVector(cos(theta),sin(theta),0.)
                    r0=c.radius*cos(d/2)/cos(theta-mid)
                    r=r0+x[t]*(c.radius-r0)
                    out[i]+=sign*w[s]*w[t]*d*(c.radius-r0)*r*integrand(c.origin+r*radial)
                end
            end
        end
    end
    out
end

# Disjoint topological partitions. Circle first, seam second, all remaining cells.
# Interface/seam band thickness shrinks under refinement; additionally report
# fixed physical bands so changing partition measure is visible.
function partitions(q,h)
    circle=falses(length(h.points));seam=copy(circle)
    for k in eachindex(h.beta)
        if q.stitched
            oncircle=all(ns -> abs(norm(q.mesh.node_coords[ns]-q.origin)-q.radius)<1e-12,
                q.mesh.face_nodes[q.mesh.face_nodes_range[h.face_a[k]]])
        else
            oncircle=h.overlap[k]!=0
        end
        if oncircle;circle[h.edge_a[k]]=true;circle[h.edge_b[k]]=true;end
        if h.face_b[k]!=0;seam[h.edge_a[k]]=true;seam[h.edge_b[k]]=true;end
    end
    seam .&= .!circle
    bulk=.!circle .& .!seam
    r=[norm(y-q.origin) for y in h.points]
    fixed_circle=abs.(r.-q.radius).<.06
    fixed_seam=[minimum((y[1],1-y[1],y[2],1-y[2]))<.10 for y in h.points]
    (;all=trues(length(circle)),circle,seam,bulk,fixed_circle,fixed_seam)
end
function norms(error,V,mask)
    ids=findall(mask); isempty(ids) && return (;count=0,volume=0.,rms=NaN,max=NaN)
    (;count=length(ids),volume=sum(V[ids]),rms=sqrt(sum(V[ids].*abs2.(error[ids]))/sum(V[ids])),max=maximum(abs,error[ids]))
end

function evaluate(q)
    q.L==1. || throw(ArgumentError("this manufactured oracle is defined for the unit square only"))
    h=geometry(q);V=volumes(h);p=field.(h.points);g=reconstruct(h,p)
    if q.stitched
        flux=control_flux(h,p,g); exactg=control_flux(h,p,gradient.(h.points))
        identity=NaN # no assembled-matrix certificate for diagnostic chord control
    else
        terms=periodic_circular_terms(h,g,1.);flux=periodic_circular_flux(terms,p)
        exactg=periodic_circular_flux(periodic_circular_terms(h,gradient.(h.points),1.),p)
        identity=maximum(abs,terms.matrix*p+terms.offset-flux.balance)
    end
    oracle=oracle_edges(q,h); reference=volume_source(q)
    reference8=volume_source(q;nquad=8); boundary8=oracle_edges(q,h;nquad=8)
    error=(flux.balance.-reference)./V
    oracleerror=(exactg.balance.-reference)./V
    point_error=reference./V.-source.(h.points)
    edge_error=flux.edges.-oracle
    parts=partitions(q,h)
    metrics=[(;region=String(name),norms(error,V,mask)...,
        exact_gradient_rms=norms(oracleerror,V,mask).rms,
        point_source_difference_rms=norms(point_error,V,mask).rms) for (name,mask) in Base.pairs(parts)]
    certificates=(;identity,conservation=abs(sum(flux.balance)),
        volume_boundary=maximum(abs,(gather(h,oracle).-reference)./V),
        volume_quadrature=maximum(abs,(reference-reference8)./V),
        boundary_quadrature=maximum(abs,(gather(h,oracle-boundary8))./V),
        volume_measure=maximum(abs,volume_source(q;integrand=x -> 1.).-V),
        edge_rms=sqrt(sum(abs2,edge_error)/length(edge_error)),edge_max=maximum(abs,edge_error))
    (;h,p,g,metrics,certificates,error,reference,edge_error)
end
end
