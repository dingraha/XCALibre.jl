export CircularReconstruction, construct_circular_reconstruction, circular_reconstruct!
export CircularFluxGeometry, construct_circular_flux_geometry
export circular_reconstructed_pressure_area!, CircularDiffusionTerms, circular_diffusion_terms
export circular_diffusion_flux!

"""Independent CPU point-sample LSQ snapshot. Exterior samples are supplied in
`exterior_faces` order at `exterior_points`. Interface neighbours are distinct
cell IDs, not overlap entries. Arrays are read-only by contract; rebuild after
geometry/ID changes. No boundary conditions or cell-average interpretation."""
struct CircularReconstruction{M,V,R,E,G}
    metrics::M
    points::V
    neighbours::R
    exterior_faces::E
    exterior_points::V
    exterior_rows::R
    inverse::G
end

_reconstruction_error(s) = throw(ArgumentError("Circular reconstruction: $s"))

function _circular_input(arrays...)
    all(x -> get_backend(x) isa CPU,arrays) || _reconstruction_error("CPU arrays required")
    Base.require_one_based_indexing(arrays...)
end

# Include every array recursively, so even snapshot geometry cannot be output.
function _circular_arrays(x)
    x isa AbstractArray && return (x,)
    x isa Number && return ()
    x isa Symbol && return ()
    isstructtype(typeof(x)) || return ()
    Tuple(a for name in fieldnames(typeof(x)) for a in _circular_arrays(getproperty(x,name)))
end
function _circular_output(out,n,vector,inputs...)
    _circular_input(out)
    length(out) == n || throw(DimensionMismatch("incorrect circular output length"))
    (vector ? eltype(out) <: SVector{3,<:AbstractFloat} : eltype(out) <: AbstractFloat) ||
        _reconstruction_error("incorrect output element type")
    any(a -> Base.mightalias(out,a),_circular_arrays(inputs)) && _reconstruction_error("output aliases input/snapshot")
end

function _circular_inverse(G,condition_limit,cell)
    e = eigvals(Symmetric(Matrix(G)))
    e[1] > 0 && isfinite(e[2]/e[1]) && e[2]/e[1] <= condition_limit ||
        _reconstruction_error("deficient or ill-conditioned stencil at cell $cell")
    inverse = inv(G)
    all(isfinite,inverse) || _reconstruction_error("nonfinite inverse at cell $cell")
    inverse
end

"""
    construct_circular_reconstruction(mesh, coupling; condition_limit=1e6)

Build inverse-distance-squared 2D least squares from ordinary internal neighbours,
distinct interface donor cells and ALL noninterface exterior face-centre samples.
Uses existing mesh cell centres as point locations; no curved-centroid correction.
Reject deficient/ill-conditioned stencils without regularisation. Construction
validates circular metrics and copies all geometry. `condition_limit` bounds the
2x2 normal-matrix spectral condition number, must be finite and >=1.
"""
function construct_circular_reconstruction(mesh::Mesh2,c::CircularInterface;condition_limit=1e6)
    isfinite(condition_limit) && condition_limit >= 1 || _reconstruction_error("invalid condition limit")
    m = construct_circular_metrics(mesh,c)
    T,I = _get_float(mesh),_get_int(mesh)
    points = copy(mesh.cell_centre)
    all(x -> all(isfinite,x) && abs(x[3]-c.origin[3]) <= 1000eps(T)*c.radius,points) ||
        _reconstruction_error("finite coplanar cell points required")
    n = length(points)
    neighbours = [I[] for _ in 1:n]
    exterior_rows = [I[] for _ in 1:n]
    exterior_faces = I[]
    for f in eachindex(m.owner_a)
        a,b = m.owner_a[f],m.owner_b[f]
        if a != b
            push!(neighbours[a],b); push!(neighbours[b],a)
        elseif !m.interface_face[f]
            push!(exterior_faces,I(f))
            push!(exterior_rows[a],I(length(exterior_faces)))
        end
    end
    for (a,b) in zip(c.cell_a,c.cell_b)
        push!(neighbours[a],b); push!(neighbours[b],a)
    end
    foreach(x -> sort!(unique!(x)),neighbours)
    ep = m.face_centre[exterior_faces]
    inverse = SMatrix{2,2,T,4}[]
    for i in 1:n
        G = zero(SMatrix{2,2,T,4})
        samples = [points[j] for j in neighbours[i]]
        append!(samples,ep[exterior_rows[i]])
        for x in samples
            d = SVector{2,T}(x[1]-points[i][1],x[2]-points[i][2])
            d2 = dot(d,d)
            isfinite(d2) && d2 > 0 || _reconstruction_error("coincident/nonfinite stencil samples at cell $i")
            G += d*d'/d2
        end
        push!(inverse,_circular_inverse(G,condition_limit,i))
    end
    CircularReconstruction(m,points,neighbours,exterior_faces,ep,exterior_rows,inverse)
end

"""
    circular_reconstruct!(gradient, reconstruction, cell_pressure, exterior_pressure)

Failure-atomic LSQ gradients (SVector{3}, zero z) from finite scalar point samples.
Exterior samples follow reconstruction.exterior_faces, not global face indexing.
No interface face samples enter the stencil; donor cell pairs are deduplicated.
"""
function circular_reconstruct!(out::AbstractVector,r::CircularReconstruction,p::AbstractVector,e::AbstractVector)
    _circular_input(p,e)
    n = length(r.points)
    length(p) == n && length(e) == length(r.exterior_faces) || throw(DimensionMismatch("incorrect reconstruction sample lengths"))
    _circular_output(out,n,true,r,p,e)
    all(isfinite,p) && all(isfinite,e) || _reconstruction_error("nonfinite pressure samples")
    proposed = similar(out)
    T = eltype(r.metrics.volume)
    for i in 1:n
        b = zero(SVector{2,T})
        for j in r.neighbours[i]
            d = SVector{2,T}(r.points[j][1]-r.points[i][1],r.points[j][2]-r.points[i][2])
            b += (p[j]-p[i])*d/dot(d,d)
        end
        for j in r.exterior_rows[i]
            d = SVector{2,T}(r.exterior_points[j][1]-r.points[i][1],r.exterior_points[j][2]-r.points[i][2])
            b += (e[j]-p[i])*d/dot(d,d)
        end
        g = r.inverse[i]*b
        proposed[i] = SVector(g[1],g[2],zero(T))
    end
    all(x -> all(isfinite,x),proposed) || _reconstruction_error("nonfinite reconstructed gradient")
    copyto!(out,proposed)
    out
end

"""Independent overlap split snapshot. `beta=dot(S,S)/dot(S,d)` and
`remainder=S-beta*d`; `weight_a` uses positive owner distances to the plane
through the arc MIDPOINT normal to S. Arc length is never a unit-normal divisor.
Read-only storage; no solver integration or variable coefficients."""
struct CircularFluxGeometry{C,V,F}
    coupling::C
    points::V
    beta::F
    remainder::V
    weight_a::F
end

"""
    construct_circular_flux_geometry(reconstruction; min_alignment=0.05)

Require dot(S,d)/(norm(S)*norm(d)) > min_alignment and positive resolved
owner distances on opposite sides of the overlap-midpoint plane. No clipping.
min_alignment is a dimensionless bound strictly between 0 and 1.
"""
function construct_circular_flux_geometry(r::CircularReconstruction;min_alignment=0.05)
    isfinite(min_alignment) && 0 < min_alignment < 1 || _reconstruction_error("invalid alignment limit")
    c = r.metrics.coupling
    T = eltype(c.measure)
    beta,weights = T[],T[]
    remainder = SVector{3,T}[]
    for k in eachindex(c.measure)
        a,b = c.cell_a[k],c.cell_b[k]
        d,S = r.points[b]-r.points[a],c.area_vector[k]
        ns,nd = norm(S),norm(d)
        dot(S,d) > min_alignment*ns*nd && ns > 0 && nd > 0 ||
            _reconstruction_error("unsuitable owner alignment at overlap $k")
        n = S/ns
        da = dot(c.midpoint[k]-r.points[a],n)
        db = dot(r.points[b]-c.midpoint[k],n)
        min(da,db) > 100eps(T)*max(c.radius,nd) || _reconstruction_error("owner points do not bracket overlap $k")
        push!(beta,dot(S,S)/dot(S,d))
        push!(remainder,S-beta[end]*d)
        push!(weights,db/(da+db))
        isfinite(beta[end]) && beta[end] > 0 && all(isfinite,remainder[end]) || _reconstruction_error("invalid flux split")
    end
    CircularFluxGeometry(deepcopy(c),copy(r.points),beta,remainder,weights)
end

function _circular_samples(f,p,g)
    _circular_input(p,g)
    length(p) == length(f.points) && length(g) == length(p) || throw(DimensionMismatch("incorrect cell samples"))
    eltype(g) <: SVector{3} && all(isfinite,p) && all(x -> all(isfinite,x) && iszero(x[3]),g) ||
        _reconstruction_error("finite xy gradients and pressures required")
end

"""
    circular_reconstructed_pressure_area!(out, geometry, cell_pressure, gradients)

Blend both owner's affine polynomials at each arc midpoint using geometry weights,
then integrate the single shared polynomial with exact arc moments. Failure-atomic;
constant/gauge/affine preserving for exact owner reconstructions. No limiter.
"""
function circular_reconstructed_pressure_area!(out::AbstractVector,f::CircularFluxGeometry,p::AbstractVector,g::AbstractVector)
    _circular_samples(f,p,g)
    c = f.coupling
    _circular_output(out,length(c.measure),true,f,p,g)
    proposed = similar(out)
    for k in eachindex(proposed)
        a,b = c.cell_a[k],c.cell_b[k]
        w,x = f.weight_a[k],c.midpoint[k]
        ga = w*g[a]+(1-w)*g[b]
        trace = w*(p[a]+dot(g[a],x-f.points[a]))+(1-w)*(p[b]+dot(g[b],x-f.points[b]))
        proposed[k] = trace*c.area_vector[k]+(c.area_moment[k]-x*c.area_vector[k]')'*ga
    end
    all(x -> all(isfinite,x),proposed) || _reconstruction_error("nonfinite reconstructed pressure integral")
    copyto!(out,proposed)
    out
end

"""Frozen constant-coefficient terms. Per overlap, implicit block is
conductance*[1 -1; -1 1]; explicit A-outward flux is `correction`, opposite on B.
For -Laplacian assembly move explicit flux to RHS with minus sign on A, plus on B.
Use SAME frozen terms for assembly and recovery; rebuild to refresh gradients.
This object provides coefficients, not an assembled equation or deferred solve."""
struct CircularDiffusionTerms{F,V}
    geometry::F
    conductance::V
    correction::V
end

"""
    circular_diffusion_terms(geometry, gradients, gamma)

Snapshot positive finite constant scalar diffusivity/mobility gamma and shared
weighted gradients. Store G=gamma*beta and correction=-gamma*dot(g_shared,T).
No variable coefficient, operator sign, reconstruction refresh or solver support.
"""
function circular_diffusion_terms(f::CircularFluxGeometry,g::AbstractVector,gamma::Real)
    _circular_samples(f,zeros(eltype(f.beta),length(f.points)),g)
    isfinite(gamma) && gamma > 0 || _reconstruction_error("positive finite constant coefficient required")
    G = similar(f.beta); correction = similar(G)
    for k in eachindex(G)
        c,w = f.coupling,f.weight_a[k]
        shared = w*g[c.cell_a[k]]+(1-w)*g[c.cell_b[k]]
        G[k] = gamma*f.beta[k]
        correction[k] = -gamma*dot(shared,f.remainder[k])
    end
    all(x -> isfinite(x) && x > 0,G) && all(isfinite,correction) || _reconstruction_error("invalid coefficient conversion")
    CircularDiffusionTerms(deepcopy(f),G,correction)
end

"""
    circular_diffusion_flux!(out, terms, cell_values)

Failure-atomic A-outward integrated flux G*(phiA-phiB)+frozen_correction.
For affine phi and exact gradients this equals -gamma*dot(gradient,S).
Gather with opposite signs using terms.geometry.coupling, never recompute the
explicit term independently on each side. Scalar point samples, CPU only.
"""
function circular_diffusion_flux!(out::AbstractVector,t::CircularDiffusionTerms,p::AbstractVector)
    _circular_input(p)
    f = t.geometry; c = f.coupling
    length(p) == length(f.points) || throw(DimensionMismatch("incorrect cell value length"))
    _circular_output(out,length(c.measure),false,t,p)
    all(isfinite,p) && all(x -> isfinite(x) && x > 0,t.conductance) && all(isfinite,t.correction) ||
        _reconstruction_error("invalid frozen flux inputs")
    proposed = similar(out)
    for k in eachindex(out)
        proposed[k] = t.conductance[k]*(p[c.cell_a[k]]-p[c.cell_b[k]])+t.correction[k]
    end
    all(isfinite,proposed) || _reconstruction_error("nonfinite recovered flux")
    copyto!(out,proposed)
    out
end
