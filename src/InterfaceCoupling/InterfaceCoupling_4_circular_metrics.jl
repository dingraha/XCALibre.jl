export CircularMetrics, construct_circular_metrics
export circular_pressure_area!, circular_gauss_gradient!

"""
    CircularMetrics

CPU geometry snapshot for a circular interface: `volume` contains chord-to-arc
corrected cell volumes, `volume_correction` their signed changes, and
`moment_correction` the corresponding surface-moment changes. `coupling` is an
independent copy. Mesh coordinates, polygonal volumes and solver state are not
modified. Rebuild after any geometry/ID change; do not mutate snapshot arrays.
No corrected cell centroids, diffusion, pressure interpolation or solver support
is provided. Ordinary (noninterface) faces remain straight chords.
"""
struct CircularMetrics{C,V,VM,VV,VI,A}
    coupling::C
    volume::V
    polygon_volume::V
    volume_correction::V
    moment_correction::VM
    face_area_vector::VV
    face_centre::VV
    owner_a::VI
    owner_b::VI
    interface_face::Vector{Bool}
    cell_faces::A
end

_metrics_error(s) = throw(ArgumentError("Circular metrics: $s"))

# One ordered row per cell, with boundary faces appearing exactly once.
function _circular_cell_faces(a,b,ncells,::Type{I}) where I
    rows = [Tuple{I,Int8}[] for _ in 1:ncells]
    for f in eachindex(a)
        push!(rows[a[f]],(I(f),Int8(1)))
        a[f] == b[f] || push!(rows[b[f]],(I(f),Int8(-1)))
    end
    ranges = UnitRange{I}[]
    faces,signs = I[],Int8[]
    for row in rows
        first = length(faces)+1
        append!(faces,(x[1] for x in row))
        append!(signs,(x[2] for x in row))
        Mesh._check_index_capacity(I,length(faces)+1,"circular cell-face adjacency")
        push!(ranges,I(first):I(length(faces)))
    end
    InterfaceAdjacency(I.(1:ncells),ranges,faces,signs)
end

"""
    construct_circular_metrics(mesh::Mesh2, coupling::CircularInterface;
                               atol=0, rtol=1000eps(T))

Create independent curved metrics without changing `mesh`. Revalidate coupling
against current mesh geometry, then replace interface chord moments by exact arc
moments. For a complete original face of angular span d, the signed cell-volume
change is s*R^2*(d-sin(d))/2; compute this once per FACE, not once per overlap.
Validate closed area-vector sums and both polygonal and curved affine moment
identities. Reject stale coupling, nonpositive volumes and inconsistent geometry.
`atol` is a length tolerance, scaled by a characteristic length for moment checks.
"""
function construct_circular_metrics(mesh::Mesh2,c::CircularInterface;
    atol=0,rtol=1000eps(_get_float(mesh)))
    get_backend(mesh.node_coords) isa CPU || _metrics_error("CPU Mesh2 required")
    T,I = _get_float(mesh),_get_int(mesh)
    eltype(c.measure) == T && eltype(c.face_a) == I || _metrics_error("mesh and coupling numeric types differ")
    isfinite(atol) && atol >= 0 && isfinite(rtol) && 0 <= rtol < 1 || _metrics_error("invalid tolerances")
    1 <= c.patch_a <= length(mesh.boundaries) && 1 <= c.patch_b <= length(mesh.boundaries) || _metrics_error("invalid patch IDs")
    fresh = construct_circular_interface(mesh,mesh.boundaries[c.patch_a].name,mesh.boundaries[c.patch_b].name;
        centre=c.origin,radius=c.radius,atol=c.validation_atol,rtol=c.validation_rtol)
    for name in fieldnames(typeof(c))
        x,y = getproperty(c,name),getproperty(fresh,name)
        # Construction is deterministic; numeric geometry must match exactly,
        # even if a caller asks for looser validation tolerances.
        if x isa InterfaceAdjacency
            all(getproperty(x,k) == getproperty(y,k) for k in (:IDs,:ranges,:overlaps,:signs)) || _metrics_error("stale coupling adjacency")
        else
            x == y || _metrics_error("stale or modified coupling: $name")
        end
    end
    nc,nf = length(mesh.cells),length(mesh.faces)
    a,b = I[],I[]
    S,C = SVector{3,T}[],SVector{3,T}[]
    for f in 1:nf
        u,v = mesh.face_ownerCells[f]
        1 <= u <= nc && 1 <= v <= nc || _metrics_error("invalid face owner")
        push!(a,u); push!(b,v)
        push!(S,mesh.face_area[f]*mesh.face_normal[f]); push!(C,mesh.face_centre[f])
        isfinite(mesh.face_area[f]) && mesh.face_area[f] > 0 && all(isfinite,S[end]) && all(isfinite,C[end]) ||
            _metrics_error("nonfinite/degenerate ordinary face geometry")
        nr = mesh.face_nodes_range[f]
        length(nr) == 2 || _metrics_error("ordinary faces must be straight chords")
        p,q = (mesh.node_coords[mesh.face_nodes[k]] for k in nr)
        tangent = q-p
        bound = T(atol)+T(rtol)*max(c.radius,norm(tangent))
        all(isfinite,p) && all(isfinite,q) && abs(p[3]-c.origin[3]) <= bound &&
            abs(q[3]-c.origin[3]) <= bound && norm(C[end]-(p+q)/2) <= bound &&
            abs(norm(tangent)-mesh.face_area[f]) <= bound &&
            abs(norm(S[end])-mesh.face_area[f]) <= bound &&
            abs(dot(S[end],tangent)) <= bound*norm(tangent) || _metrics_error("stale ordinary face geometry")
    end
    V = copy(mesh.cell_volume)
    all(x -> isfinite(x) && x > 0,V) || _metrics_error("polygonal volumes must be positive and finite")
    adj = _circular_cell_faces(a,b,nc,I)
    P = SMatrix{3,3,T}(1,0,0,0,1,0,0,0,0)
    delta = zeros(T,nc)
    dM = fill(zero(P),nc)
    marked = fill(false,nf)
    # Work relative to the circle origin to avoid cancellation under translation.
    for row in eachindex(c.face_adjacency.IDs)
        f = c.face_adjacency.IDs[row]
        marked[f] = true
        d = zero(T)
        moment = zero(P)
        for entry in c.face_adjacency.ranges[row]
            k = c.face_adjacency.overlaps[entry]
            sign = c.face_adjacency.signs[entry]
            d += c.theta_hi[k]-c.theta_lo[k]
            moment += sign*(c.area_moment[k]-c.origin*c.area_vector[k]')
        end
        radial = C[f]-c.origin
        sign = dot(S[f],radial) > 0 ? one(T) : -one(T)
        dv = sign*c.radius^2*(d-sin(d))/2
        dm = moment-(C[f]-c.origin)*S[f]'
        scale = max(c.radius^2,norm(dm),abs(dv))
        norm(dm-dv*P) <= T(atol)*c.radius + T(rtol)*scale || _metrics_error("arc/chord moment mismatch")
        delta[a[f]] += dv
        dM[a[f]] += dm
    end
    curved = V+delta
    all(x -> isfinite(x) && x > 0,curved) || _metrics_error("corrected volumes must be positive and finite")
    for cell in 1:nc
        sumS = zero(SVector{3,T})
        M = zero(P)
        area_scale,moment_scale = zero(T),zero(T)
        for entry in adj.ranges[cell]
            f,sign = adj.overlaps[entry],adj.signs[entry]
            sumS += sign*S[f]
            piece = (C[f]-c.origin)*(sign*S[f])'
            M += piece
            area_scale += norm(S[f])
            moment_scale += norm(piece)
        end
        norm(sumS) <= T(atol)+T(rtol)*area_scale || _metrics_error("cell $cell does not close")
        bound = T(atol)*max(c.radius,sqrt(V[cell])) + T(rtol)*max(moment_scale,V[cell])
        norm(M-V[cell]*P) <= bound || _metrics_error("polygonal moment/volume mismatch in cell $cell")
        norm(M+dM[cell]-curved[cell]*P) <= 2bound || _metrics_error("curved moment/volume mismatch in cell $cell")
    end
    CircularMetrics(deepcopy(c),curved,V,delta,dM,S,C,a,b,marked,adj)
end

"""
    circular_pressure_area!(out, coupling, pressure, gradient;
                            reference=coupling.origin)

Overwrite shared A-oriented overlap pressure-area vectors for the AFFINE field
p(x)=pressure + dot(gradient,x-reference), using exact surface moments:
pressure*S + (M-reference*S')'*gradient. CPU arrays only; output is failure-atomic.
This analytical/reconstructed-field primitive does not interpolate cell pressure.
`gradient` and `reference` have three components, with zero gradient.z required.
"""
function circular_pressure_area!(out::AbstractVector,c::CircularInterface,pressure,gradient;reference=c.origin)
    get_backend(out) isa CPU && get_backend(c.measure) isa CPU || _metrics_error("CPU arrays required")
    Base.require_one_based_indexing(out)
    length(out) == length(c.measure) || throw(DimensionMismatch("incorrect overlap pressure-area length"))
    eltype(out) <: SVector{3} || _metrics_error("output must store SVector{3}")
    length(gradient) == 3 && length(reference) == 3 && all(isfinite,gradient) && all(isfinite,reference) &&
        isfinite(pressure) && iszero(gradient[3]) || _metrics_error("finite xy affine pressure required")
    any(x -> Base.mightalias(out,x),(c.area_vector,c.area_moment,c.centre,c.midpoint,c.measure,c.theta_lo,c.theta_hi)) &&
        _metrics_error("output aliases geometry")
    g,r = SVector{3}(gradient),SVector{3}(reference)
    proposed = similar(out)
    for k in eachindex(proposed)
        proposed[k] = pressure*c.area_vector[k]+(c.area_moment[k]-r*c.area_vector[k]')'*g
        all(isfinite,proposed[k]) || _metrics_error("nonfinite pressure-area integral")
    end
    copyto!(out,proposed)
    out
end

"""
    circular_gauss_gradient!(out, metrics, face_pressure, overlap_pressure_area)

Overwrite all cell gradients (SVector{3} output). Sum pressure*area_vector on
noninterface straight faces and shared integrated pressure-area vectors on
circular overlaps, then divide by corrected volume. Interface entries of
`face_pressure` are deliberately ignored (may be NaN); they are REPLACED, never
counted twice. No cell interpolation or boundary-condition application occurs.
All inputs are CPU arrays. Validation/conversion is failure-atomic. Metrics are
an independent snapshot, not a live view of mesh geometry or solver state.
"""
function circular_gauss_gradient!(out::AbstractVector,m::CircularMetrics,
    face_pressure::AbstractVector,overlap_pressure_area::AbstractVector)
    c = m.coupling
    all(x -> get_backend(x) isa CPU,(out,face_pressure,overlap_pressure_area,m.volume)) || _metrics_error("CPU arrays required")
    Base.require_one_based_indexing(out,face_pressure,overlap_pressure_area)
    length(out) == length(m.volume) && length(face_pressure) == length(m.owner_a) &&
        length(overlap_pressure_area) == length(c.measure) || throw(DimensionMismatch("incorrect gradient input/output length"))
    eltype(out) <: SVector{3} && eltype(overlap_pressure_area) <: SVector{3} || _metrics_error("gradient and pressure-area storage must use SVector{3}")
    any(x -> Base.mightalias(out,x),(face_pressure,overlap_pressure_area,m.face_area_vector,m.face_centre,
        m.moment_correction,c.area_vector,c.area_moment,c.centre,c.midpoint,m.volume,m.polygon_volume,m.volume_correction)) &&
        _metrics_error("gradient output aliases inputs/geometry")
    all(x -> isfinite(x) && x > 0,m.volume) || _metrics_error("invalid corrected volume")
    all(k -> m.interface_face[k] || isfinite(face_pressure[k]),eachindex(face_pressure)) &&
        all(x -> all(isfinite,x),overlap_pressure_area) || _metrics_error("nonfinite pressure inputs")
    # Gather each overlap once per owner, in deterministic adjacency order.
    T = eltype(m.volume)
    interface = fill(zero(SVector{3,T}),length(m.volume))
    interface_cell_flux!(interface,c,overlap_pressure_area)
    proposed = similar(out)
    adj = m.cell_faces
    Threads.@threads for cell in eachindex(m.volume)
        total = interface[cell]
        for entry in adj.ranges[cell]
            f = adj.overlaps[entry]
            m.interface_face[f] && continue
            total += face_pressure[f]*(adj.signs[entry]*m.face_area_vector[f])
        end
        proposed[cell] = total/m.volume[cell]
    end
    all(x -> all(isfinite,x),proposed) || _metrics_error("nonfinite gradient conversion")
    copyto!(out,proposed)
    out
end
