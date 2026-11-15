export PeriodicCircularSurfaces, construct_periodic_circular_surfaces
export periodic_circular_surface_trace, periodic_circular_surface_gradient
export periodic_circular_surface_flux

"""Frozen quadratic value-to-normal-area maps. `first[k]` is ∫(x-F)n' ds;
`second[k][j]` is ∫(x-F)(x-F)' n_j ds. `trace_matrix[j]` maps point
scalars to ∫p n_j ds on shared A-outward edges. Arrays read-only by contract.
Static CPU Float64 unit-square split-circle only; no pressure/momentum solver,
projection, variable mobility, motion or stability claim. Rebuild after changes.
"""
struct PeriodicCircularSurfaces{T}
    quadratic::T
    first::Vector{SMatrix{3,3,Float64,9}}
    second::Vector{NTuple{3,SMatrix{3,3,Float64,9}}}
    trace_matrix::NTuple{3,SparseMatrixCSC{Float64,Int}}
end

# Exact finite Fourier integral of cos(theta)^a*sin(theta)^b. Midpoint/sinc
# form avoids subtraction of nearby endpoint antiderivatives on small arcs.
function _surface_trig_integral(a,b,lo,hi)
    result=0.0+0.0im;d=hi-lo;mid=(hi+lo)/2
    for u in 0:a, v in 0:b
        mode=a+b-2u-2v
        coefficient=binomial(a,u)*binomial(b,v)*(-1)^v/(2.0^(a+b)*im^b)
        result+=coefficient*d*cis(mode*mid)*sinc(mode*d/(2pi))
    end
    real(result)
end

function _surface_moments(h,k)
    F,S=h.point[k],h.area[k];Z=zero(SMatrix{3,3,Float64,9})
    if !iszero(h.face_a[k])
        # Ordinary faces are validated coplanar straight chords. Tangent sign
        # is immaterial for a symmetric second moment.
        tangent=SVector(-S[2],S[1],0.)
        return Z,ntuple(j -> S[j]*(tangent*tangent')/12,3)
    end
    c=h.metrics.coupling;o=h.overlap[k];lo,hi=c.theta_lo[o],c.theta_hi[o]
    R=c.radius;offset=c.origin-F
    radial=SVector(cos((lo+hi)/2),sin((lo+hi)/2),0.)
    orientation=dot(S,radial)>0 ? 1.0 : -1.0
    integral(a,b)=orientation*R*_surface_trig_integral(a,b,lo,hi)
    function monomial(indices)
        integral(count(==(1),indices),count(==(2),indices))
    end
    # All z coordinates are coplanar: both centred position and normal z zero.
    first=SMatrix{3,3,Float64,9}(ntuple(9) do l
        i,j=mod1(l,3),cld(l,3)
        i==3 || j==3 ? 0. : offset[i]*monomial((j,))+R*monomial((i,j))
    end)
    second=ntuple(3) do j
        SMatrix{3,3,Float64,9}(ntuple(9) do l
            i,v=mod1(l,3),cld(l,3)
            i==3 || v==3 || j==3 ? 0. :
                offset[i]*offset[v]*monomial((j,)) +
                R*(offset[i]*monomial((v,j))+offset[v]*monomial((i,j))) +
                R^2*monomial((i,v,j))
        end)
    end
    first,second
end

# ∫ five scaled nonconstant basis functions n_component ds, evaluated about
# the owner in the edge's A frame. The constant term is handled separately.
function _surface_functional(h,k,owner,scale,is_a,first,second,component)
    d=h.point[k]-h.points[owner]-(is_a ? zero(h.shift[k]) : h.shift[k])
    S=h.area[k][component];M=first[:,component];N=second[component]
    [(d[1]*S+M[1])/scale,(d[2]*S+M[2])/scale,
     (d[1]^2*S+2d[1]*M[1]+N[1,1])/(2scale^2),
     (d[1]*d[2]*S+d[1]*M[2]+d[2]*M[1]+N[1,2])/scale^2,
     (d[2]^2*S+2d[2]*M[2]+N[2,2])/(2scale^2)]
end

"""Construct independently owned frozen value/normal-area maps from the qualified
quadratic stencil snapshot. Diffusion coefficient has no effect on value traces.
Input snapshot must be unmodified; no live mutation tracking or fallback.
"""
function construct_periodic_circular_surfaces(t::PeriodicCircularQuadratic)
    frozen=deepcopy(t);h=frozen.geometry;ne=length(h.beta);n=length(h.points)
    moments=[_surface_moments(h,k) for k in 1:ne]
    first=[m[1] for m in moments];second=[m[2] for m in moments]
    maps=ntuple(3) do component
        rows=Int[];cols=Int[];values=Float64[]
        for k in 1:ne
            for (i,w,is_a) in ((h.edge_a[k],h.weight[k],true),(h.edge_b[k],1-h.weight[k],false))
                st=frozen.stencils[i]
                functional=_surface_functional(h,k,i,st.scale,is_a,first[k],second[k],component)
                row=w.*(st.recovery'*functional)
                for (j,x) in zip(st.donors,row);push!(rows,k);push!(cols,j);push!(values,x);end
                push!(rows,k);push!(cols,i);push!(values,w*h.area[k][component]-sum(row))
            end
        end
        sparse(rows,cols,values,ne,n)
    end
    all(A -> all(isfinite,nonzeros(A)),maps) || _reconstruction_error("nonfinite surface maps")
    PeriodicCircularSurfaces(frozen,first,second,maps)
end

function _surface_samples(t,p)
    _circular_input(p)
    length(p)==length(t.quadratic.geometry.points) || throw(DimensionMismatch("incorrect surface sample length"))
    all(isfinite,p) || _reconstruction_error("nonfinite surface samples")
end

"""Owned shared A-outward vector integrals ∫p n ds from scalar point samples.
This is an integrated normal-area trace, NOT p evaluated at the face centre.
"""
function periodic_circular_surface_trace(t::PeriodicCircularSurfaces,p::AbstractVector)
    _surface_samples(t,p)
    components=map(A -> A*p,t.trace_matrix)
    edges=[SVector(components[1][k],components[2][k],components[3][k]) for k in eachindex(components[1])]
    all(x -> all(isfinite,x),edges) || _reconstruction_error("nonfinite surface trace")
    edges
end

"""Conservative Gauss pressure-gradient observation: gathered ∫p n ds divided
by corrected volume. Not direct centre polynomial derivatives, nor a projection.
"""
function periodic_circular_surface_gradient(t::PeriodicCircularSurfaces,p::AbstractVector)
    edges=periodic_circular_surface_trace(t,p);h=t.quadratic.geometry
    g=fill(SVector(0.,0.,0.),length(h.points))
    for k in eachindex(edges)
        g[h.edge_a[k]]+=edges[k];g[h.edge_b[k]]-=edges[k]
    end
    gradient=g./h.metrics.volume
    all(x -> all(isfinite,x),gradient) || _reconstruction_error("nonfinite surface gradient")
    gradient
end

"""Owned shared integrated Cartesian velocity flux and diagnostic face/overlap
arrays, balances and divergence. Velocity input is a vector of finite xy vectors
with EXACT zero z (three components). No density/mobility multiplication.
"""
function periodic_circular_surface_flux(t::PeriodicCircularSurfaces,U::AbstractVector)
    _circular_input(U);h=t.quadratic.geometry
    length(U)==length(h.points) || throw(DimensionMismatch("incorrect surface velocity length"))
    all(u -> length(u)==3 && all(isfinite,u) && iszero(u[3]),U) || _reconstruction_error("finite xy velocity with zero z required")
    edges=t.trace_matrix[1]*[u[1] for u in U]+t.trace_matrix[2]*[u[2] for u in U]
    faces=zeros(length(h.metrics.owner_a));overlaps=zeros(length(h.metrics.coupling.measure));balance=zeros(length(U))
    for k in eachindex(edges)
        balance[h.edge_a[k]]+=edges[k];balance[h.edge_b[k]]-=edges[k]
        if iszero(h.face_a[k]);overlaps[h.overlap[k]]=edges[k]
        else
            faces[h.face_a[k]]=edges[k]
            iszero(h.face_b[k]) || (faces[h.face_b[k]]=-edges[k])
        end
    end
    interface_face_flux!(faces,h.metrics.coupling,overlaps)
    divergence=balance./h.metrics.volume
    all(isfinite,edges) && all(isfinite,divergence) || _reconstruction_error("nonfinite surface flux")
    (;edges,faces,overlaps,balance,divergence)
end
