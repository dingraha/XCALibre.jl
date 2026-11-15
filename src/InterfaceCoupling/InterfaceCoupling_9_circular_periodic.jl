export PeriodicCircularGeometry, construct_periodic_circular_geometry
export periodic_circular_reconstruct!, periodic_circular_gradient!, periodic_circular_terms
export periodic_circular_flux, PeriodicCircularTerms

"""Static CPU all-periodic scalar operator snapshot. Each edge is counted once.
`face_b` is nonzero only for a periodic edge; its flux is minus `face_a`.
`shift` translates B into A's frame, not the field value. No quasi-periodic jump,
velocity rotation, BC anchoring, momentum, transient integration or ALE support.
Cell data are point samples; all snapshot arrays are read-only by contract.
"""
struct PeriodicCircularGeometry{M,V,A,W,R,G,F}
    metrics::M
    points::V
    edge_a::A
    edge_b::A
    face_a::A
    face_b::A
    overlap::A
    shift::V
    area::V
    point::V
    beta::W
    remainder::V
    weight::W
    rows::R
    inverse::G
    overlap_geometry::F
end

"""
    construct_periodic_circular_geometry(mesh, coupling, pairs; condition_limit=1e6, min_alignment=0.05)

`pairs` contains `(face_a,face_b,shift_b_to_a)` exactly once per translated
periodic face pair. Cover ALL noninterface exterior faces. Validate translated
centres, equal/opposite area vectors, owner bracketing and LSQ conditioning.
Periodic neighbours are edge/image rows (not deduplicated cell IDs): corners and
small periodic domains may contain multiple distinct images of the same cell.
No regularisation, coordinate search, automatic mutation tracking or fallback.
"""
function construct_periodic_circular_geometry(mesh::Mesh2,c::CircularInterface,pairs;
    condition_limit=1e6,min_alignment=0.05)
    isfinite(condition_limit) && condition_limit>=1 || _reconstruction_error("invalid condition limit")
    isfinite(min_alignment) && 0<min_alignment<1 || _reconstruction_error("invalid alignment limit")
    m = construct_circular_metrics(mesh,c)
    T,I = eltype(m.volume),eltype(c.cell_a)
    points = copy(mesh.cell_centre)
    all(x -> all(isfinite,x) && abs(x[3]-c.origin[3])<=1000eps(T)*c.radius,points) ||
        _reconstruction_error("finite coplanar cell points required")
    a,b,fa,fb,ov = I[],I[],I[],I[],I[]
    shifts,area,point,rem = [SVector{3,T}[] for _ in 1:4]
    beta,weight = T[],T[]
    function add(u,v,j,k,o,s,S,x)
        B,R,w = _pressure_split(S,points[v]+s-points[u],points[u],x;
            min_alignment,radius=c.radius)
        push!(a,u);push!(b,v);push!(fa,j);push!(fb,k);push!(ov,o)
        push!(shifts,s);push!(area,S);push!(point,x);push!(beta,B);push!(rem,R);push!(weight,w)
    end
    z = zero(SVector{3,T})
    for j in eachindex(m.owner_a)
        u,v = m.owner_a[j],m.owner_b[j]
        u==v || add(u,v,I(j),zero(I),zero(I),z,m.face_area_vector[j],m.face_centre[j])
    end
    seen = falses(length(m.owner_a))
    for pair in pairs
        length(pair)==3 || _reconstruction_error("periodic pair requires two face IDs and shift")
        j,k,s0 = pair
        all(f -> f isa Integer && 1<=f<=length(seen), (j,k)) || _reconstruction_error("invalid periodic face ID")
        j!=k && !seen[j] && !seen[k] || _reconstruction_error("duplicate periodic face")
        all(f -> !m.interface_face[f] && m.owner_a[f]==m.owner_b[f],(j,k)) ||
            _reconstruction_error("periodic pairs must be ordinary exterior faces")
        length(s0)==3 && all(isfinite,s0) || _reconstruction_error("finite xy translation required")
        s = SVector{3,T}(s0)
        iszero(s[3]) && norm(s)>0 || _reconstruction_error("nonzero xy translation required")
        scale = max(c.radius,norm(s))
        norm(m.face_centre[k]+s-m.face_centre[j])<=1000eps(T)*scale || _reconstruction_error("periodic centres do not translate")
        norm(m.face_area_vector[k]+m.face_area_vector[j])<=1000eps(T)*scale || _reconstruction_error("periodic areas not opposite")
        seen[j]=seen[k]=true
        add(m.owner_a[j],m.owner_a[k],I(j),I(k),zero(I),s,m.face_area_vector[j],m.face_centre[j])
    end
    all(j -> m.owner_a[j]!=m.owner_b[j] || m.interface_face[j] || seen[j],eachindex(seen)) ||
        _reconstruction_error("incomplete periodic exterior coverage")
    for k in eachindex(c.measure)
        add(c.cell_a[k],c.cell_b[k],zero(I),zero(I),I(k),z,c.area_vector[k],c.midpoint[k])
    end
    # Ordinary and overlap donors deduplicate by cell ID; periodic images remain distinct.
    rows = [Tuple{I,SVector{3,T}}[] for _ in points]
    for k in eachindex(a)
        push!(rows[a[k]],(b[k],points[b[k]]+shifts[k]-points[a[k]]))
        push!(rows[b[k]],(a[k],points[a[k]]-shifts[k]-points[b[k]]))
    end
    foreach(unique!,rows)
    inverse = SMatrix{2,2,T,4}[]
    for (i,row) in enumerate(rows)
        G = zero(SMatrix{2,2,T,4})
        for (_,d3) in row
            d = SVector{2,T}(d3[1:2]); d2=dot(d,d)
            isfinite(d2) && d2>0 || _reconstruction_error("coincident stencil samples")
            G += d*d'/d2
        end
        push!(inverse,_circular_inverse(G,condition_limit,i))
    end
    ids = findall(!iszero,ov)
    f = CircularFluxGeometry(deepcopy(c),copy(points),beta[ids],rem[ids],weight[ids])
    PeriodicCircularGeometry(m,points,a,b,fa,fb,ov,shifts,area,point,beta,rem,weight,rows,inverse,f)
end

"""Failure-atomic LSQ on periodic image rows. Scalar values must be periodic;
a global affine function is NOT periodic and is not a valid seam test."""
function periodic_circular_reconstruct!(out::AbstractVector,h::PeriodicCircularGeometry,p::AbstractVector)
    _circular_input(p)
    length(p)==length(h.points) || throw(DimensionMismatch("incorrect periodic sample length"))
    _circular_output(out,length(p),true,h,p)
    all(isfinite,p) || _reconstruction_error("nonfinite periodic samples")
    T=eltype(h.beta); proposed=similar(out)
    for i in eachindex(p)
        rhs=zero(SVector{2,T})
        for (j,d3) in h.rows[i]
            d=SVector{2,T}(d3[1:2]); rhs+=(p[j]-p[i])*d/dot(d,d)
        end
        g=h.inverse[i]*rhs; proposed[i]=SVector(g[1],g[2],zero(T))
    end
    all(x -> all(isfinite,x),proposed) || _reconstruction_error("nonfinite periodic reconstruction")
    copyto!(out,proposed)
end

"""Reconstructed shared traces and exact arc integrals, divided by corrected
volumes. Periodic B extrapolation uses translated B point in A's frame."""
function periodic_circular_gradient!(out::AbstractVector,h::PeriodicCircularGeometry,p::AbstractVector)
    _circular_output(out,length(h.points),true,h,p)
    T=eltype(h.beta); g=fill(zero(SVector{3,T}),length(h.points))
    periodic_circular_reconstruct!(g,h,p)
    traces=fill(T(NaN),length(h.metrics.owner_a))
    for k in eachindex(h.beta)
        j=h.face_a[k]; iszero(j) && continue
        a,b,x,w=h.edge_a[k],h.edge_b[k],h.point[k],h.weight[k]
        trace=w*(p[a]+dot(g[a],x-h.points[a]))+(1-w)*(p[b]+dot(g[b],x-h.points[b]-h.shift[k]))
        traces[j]=trace
        iszero(h.face_b[k]) || (traces[h.face_b[k]]=trace)
    end
    integral=fill(zero(SVector{3,T}),length(h.metrics.coupling.measure))
    circular_reconstructed_pressure_area!(integral,h.overlap_geometry,p,g)
    circular_gauss_gradient!(out,h.metrics,traces,integral)
end

"""Frozen scalar negative-Laplacian terms. Read-only by contract; assembly and
recovery must use the same snapshot."""
struct PeriodicCircularTerms{H,V,A}
    geometry::H
    conductance::V
    correction::V
    matrix::A
    offset::V
end

"""Frozen constant-positive-coefficient scalar negative-Laplacian terms.
Sparse integrated matrix and explicit offset share the same single edge flux.
No solve/deferred iteration, pressure gauge modification or momentum integration.
"""
function periodic_circular_terms(h::PeriodicCircularGeometry,g::AbstractVector,gamma::Real)
    _circular_samples(h.overlap_geometry,zeros(eltype(h.beta),length(h.points)),g)
    isfinite(gamma) && gamma>0 || _reconstruction_error("positive finite coefficient required")
    T=eltype(h.beta); G=T.(gamma.*h.beta); E=similar(G)
    rows,cols=Int[],Int[]; vals=T[]; offset=zeros(T,length(h.points))
    for k in eachindex(G)
        a,b,w=h.edge_a[k],h.edge_b[k],h.weight[k]
        E[k]=-gamma*dot(w*g[a]+(1-w)*g[b],h.remainder[k])
        append!(rows,(a,a,b,b));append!(cols,(a,b,a,b));append!(vals,(G[k],-G[k],-G[k],G[k]))
        offset[a]+=E[k];offset[b]-=E[k]
    end
    all(x -> isfinite(x) && x>0,G) && all(isfinite,E) && all(isfinite,offset) || _reconstruction_error("nonfinite periodic terms")
    A=sparse(rows,cols,vals,length(h.points),length(h.points))
    all(isfinite,nonzeros(A)) || _reconstruction_error("nonfinite periodic assembled matrix")
    PeriodicCircularTerms(deepcopy(h),G,E,A,offset)
end

"""Recover from frozen terms. Edge flux is authoritative; opposite periodic
boundary entries are diagnostics. Gather each edge once, including self images."""
function periodic_circular_flux(t::PeriodicCircularTerms,p::AbstractVector)
    h=t.geometry; _circular_input(p)
    length(p)==length(h.points) || throw(DimensionMismatch("incorrect periodic sample length"))
    all(isfinite,p) || _reconstruction_error("nonfinite periodic samples")
    q=t.conductance.*(p[h.edge_a].-p[h.edge_b]).+t.correction
    T=eltype(h.beta); balance=zeros(T,length(p)); faces=zeros(T,length(h.metrics.owner_a)); overlaps=zeros(T,length(h.metrics.coupling.measure))
    for k in eachindex(q)
        balance[h.edge_a[k]]+=q[k];balance[h.edge_b[k]]-=q[k]
        if iszero(h.face_a[k])
            overlaps[h.overlap[k]]=q[k]
        else
            faces[h.face_a[k]]=q[k]
            iszero(h.face_b[k]) || (faces[h.face_b[k]]=-q[k])
        end
    end
    interface_face_flux!(faces,h.metrics.coupling,overlaps)
    divergence=balance./h.metrics.volume
    all(isfinite,q) && all(isfinite,divergence) || _reconstruction_error("nonfinite periodic recovered flux")
    (;edges=q,faces,overlaps,balance,divergence)
end
