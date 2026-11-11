export construct_planar_interface

_interface_error(message) = throw(ArgumentError("Planar interface: $message"))

"""
    construct_planar_interface(mesh::Mesh2, patch_a, patch_b; atol=0, rtol=100eps(T))

Construct shared overlaps between two nonempty, coincident straight 2D patches on
CPU. Checks opposite normals, positive lengths, duplicate coverage, and coverage
and area-vector reconstruction on both sides. No weights are renormalised.
`atol` is a length tolerance; `rtol` is dimensionless. Endpoint-only contacts do not
produce overlaps. Curved interfaces, Mesh3, and non-CPU storage are unsupported.
"""
function construct_planar_interface(mesh::Mesh2, patch_a::Symbol, patch_b::Symbol;
    atol=0, rtol=100eps(_get_float(mesh)))
    get_backend(mesh.node_coords) isa CPU || _interface_error("only CPU meshes are supported")
    T, I = _get_float(mesh), _get_int(mesh)
    isfinite(atol) && atol >= 0 || _interface_error("atol must be finite and nonnegative")
    isfinite(rtol) && 0 <= rtol < 1 || _interface_error("rtol must be finite and in [0,1)")
    pa = boundary_index(mesh.boundaries, patch_a)
    pb = boundary_index(mesh.boundaries, patch_b)
    pa === nothing && _interface_error("unknown patch $patch_a")
    pb === nothing && _interface_error("unknown patch $patch_b")
    pa != pb || _interface_error("patches must be distinct")
    fa = collect(mesh.boundaries[pa].IDs_range)
    fb = collect(mesh.boundaries[pb].IDs_range)
    (isempty(fa) || isempty(fb)) && _interface_error("patches must be nonempty")
    allfaces = [fa; fb]
    endpoints = map(allfaces) do f
        nr = mesh.face_nodes_range[f]
        length(nr) == 2 || _interface_error("face $f must have two nodes")
        is_boundary(mesh.face_ownerCells[f]) || _interface_error("face $f is not a boundary face")
        p, q = (mesh.node_coords[mesh.face_nodes[k]] for k in nr)
        all(isfinite, p) && all(isfinite, q) || _interface_error("nonfinite endpoint at face $f")
        (p, q)
    end
    origin, endpoint = first(endpoints)
    d = endpoint - origin
    norm(d) > zero(T) || _interface_error("degenerate first edge")
    tangent = d/norm(d)
    # Canonical tangent makes endpoint reversal leave overlap ordering unchanged.
    axis = argmax(abs.(tangent))
    tangent[axis] < 0 && (tangent = -tangent)
    scale = maximum(norm(p - origin) for pair in endpoints for p in pair)
    tol = T(atol) + T(rtol)*scale
    ntol = T(rtol) + T(atol)/scale
    normal = mesh.face_normal[first(fa)]
    abs(norm(normal) - one(T)) <= ntol || _interface_error("invalid reference normal")
    abs(dot(normal, tangent)) <= ntol || _interface_error("normal is not perpendicular to interface")
    intervals = map(zip(allfaces, endpoints)) do (f, (p, q))
        a, b = dot(p - origin, tangent), dot(q - origin, tangent)
        norm((p - origin) - a*tangent) <= tol &&
            norm((q - origin) - b*tangent) <= tol || _interface_error("face $f is not on the common line")
        lo, hi = minmax(a, b)
        hi - lo > tol || _interface_error("degenerate or unresolved edge at face $f")
        isfinite(mesh.face_area[f]) && abs(mesh.face_area[f] - (hi - lo)) <= tol ||
            _interface_error("inconsistent length at face $f")
        (lo=lo, hi=hi, face=f)
    end
    ia = sort(intervals[1:length(fa)]; by=x -> (x.lo, x.hi, x.face))
    ib = sort(intervals[length(fa)+1:end]; by=x -> (x.lo, x.hi, x.face))
    for (side, expected) in ((ia, normal), (ib, -normal))
        for (j, edge) in enumerate(side)
            norm(mesh.face_normal[edge.face] - expected) <= ntol ||
                _interface_error("incompatible outward normal at face $(edge.face)")
            j == 1 || edge.lo >= side[j-1].hi - tol ||
                _interface_error("duplicate coverage at face $(edge.face)")
        end
    end
    face_a, face_b, cell_a, cell_b = (I[] for _ in 1:4)
    measure = T[]
    centre, area_vector = SVector{3,T}[], SVector{3,T}[]
    i, j = 1, 1
    while i <= length(ia) && j <= length(ib)
        a, b = ia[i], ib[j]
        lo, hi = max(a.lo, b.lo), min(a.hi, b.hi)
        if hi - lo > tol
            push!(face_a, a.face); push!(face_b, b.face)
            push!(cell_a, mesh.face_ownerCells[a.face][1])
            push!(cell_b, mesh.face_ownerCells[b.face][1])
            push!(measure, hi - lo)
            push!(centre, origin + ((lo + hi)/2)*tangent)
            push!(area_vector, (hi - lo)*normal)
        end
        # Advance exact endpoints; tolerance is used for geometry checks, not topology.
        if a.hi < b.hi
            i += 1
        elseif b.hi < a.hi
            j += 1
        else
            i += 1; j += 1
        end
    end
    adj = _interface_adjacency(face_a, face_b)
    covered = zeros(T, length(mesh.faces))
    vectors = fill(zero(SVector{3,T}), length(mesh.faces))
    for row in eachindex(adj.IDs)
        f = adj.IDs[row]
        for k in adj.ranges[row]
            o = adj.overlaps[k]
            covered[f] += measure[o]
            vectors[f] += adj.signs[k]*area_vector[o]
        end
    end
    for f in allfaces
        abs(covered[f] - mesh.face_area[f]) <= tol ||
            _interface_error("incomplete coverage at face $f: covered $(covered[f]), expected $(mesh.face_area[f])")
        norm(vectors[f] - mesh.face_area[f]*mesh.face_normal[f]) <= tol + ntol*mesh.face_area[f] ||
            _interface_error("area-vector reconstruction failed at face $f")
    end
    PlanarInterface(I(pa), I(pb), face_a, face_b, cell_a, cell_b, measure,
        centre, area_vector, adj, _interface_adjacency(cell_a, cell_b))
end

construct_planar_interface(mesh::AbstractMesh, a::Symbol, b::Symbol; kwargs...) =
    _interface_error("only straight Mesh2 interfaces are supported")
