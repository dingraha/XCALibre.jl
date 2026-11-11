export CircularInterface, construct_circular_interface

"""
    CircularInterface

Geometry-only shared overlaps on a complete circle in the Mesh2 xy plane.
`measure` is arc length, NOT chord length or norm(area_vector). `centre` is
its arc-length centroid (inside the circle), `midpoint` lies on the circle,
and `area_moment` is the exact integral of x*n' ds, oriented out of A.
Angular intervals lie in [0,2pi]; the fixed seam may split a face pair twice.
Gather integrated overlap fluxes with interface_face_flux!/interface_cell_flux!.
No diffusion, pressure, solver, motion, or ALE support is implied. Construct
new geometry after any coordinate/ID change; never transfer old fluxes by index.
"""
struct CircularInterface{I,VI,VF,VV,VM,A,T,V}
    patch_a::I
    patch_b::I
    face_a::VI
    face_b::VI
    cell_a::VI
    cell_b::VI
    measure::VF
    centre::VV
    midpoint::VV
    area_vector::VV
    area_moment::VM
    theta_lo::VF
    theta_hi::VF
    face_adjacency::A
    cell_adjacency::A
    radius::T
    origin::V
end
Adapt.@adapt_structure CircularInterface

_circular_error(s) = throw(ArgumentError("Circular interface: $s"))

"""
    construct_circular_interface(mesh, patch_a, patch_b; centre, radius,
                                 atol=0, rtol=100eps(T))

Validate complete, singly covered circular patches with opposing radial chord
normals. Endpoints must lie on the specified circle; each face spans strictly
less than pi radians. Chord areas/centres/normals must match mesh geometry.
No projection, coverage renormalisation, mesh mutation, or curved-cell volume
replacement is performed. atol is a length tolerance. CPU Mesh2 only.
"""
function construct_circular_interface(mesh::Mesh2,a::Symbol,b::Symbol;
    centre, radius, atol=0, rtol=100eps(_get_float(mesh)))
    get_backend(mesh.node_coords) isa CPU || _circular_error("CPU meshes required")
    T,I = _get_float(mesh),_get_int(mesh)
    length(centre) == 3 && all(isfinite,centre) || _circular_error("centre must have three finite coordinates")
    O = SVector{3,T}(centre)
    all(isfinite,O) || _circular_error("centre conversion is not finite")
    R = T(radius)
    isfinite(R) && R > 0 || _circular_error("radius must be positive and finite")
    isfinite(atol) && atol >= 0 && isfinite(rtol) && 0 <= rtol < 1 ||
        _circular_error("invalid tolerances")
    tol = T(atol)+T(rtol)*R
    isfinite(tol) && tol < R || _circular_error("tolerance must be smaller than radius")
    at = tol/R
    tau = T(2pi)
    pa,pb = boundary_index(mesh.boundaries,a),boundary_index(mesh.boundaries,b)
    pa !== nothing && pb !== nothing && pa != pb || _circular_error("unknown or identical patches")
    fa,fb = collect(mesh.boundaries[pa].IDs_range),collect(mesh.boundaries[pb].IDs_range)
    !isempty(fa) && !isempty(fb) || _circular_error("empty patch")
    orientation = 0
    Interval = NamedTuple{(:lo,:hi,:face),Tuple{T,T,I}}
    sides = Vector{Interval}[]
    expected_length = Dict{I,T}()
    for (side,faces) in enumerate((fa,fb))
        intervals = Interval[]
        for f in faces
            nr = mesh.face_nodes_range[f]
            length(nr) == 2 && is_boundary(mesh.face_ownerCells[f]) || _circular_error("face $f is not a boundary chord")
            p,q = (mesh.node_coords[mesh.face_nodes[k]] for k in nr)
            all(isfinite,p) && all(isfinite,q) || _circular_error("nonfinite endpoint")
            for x in (p,q)
                abs(x[3]-O[3]) <= tol && abs(norm(x-O)-R) <= tol || _circular_error("endpoint off circle at face $f")
            end
            t0,t1 = mod(atan(p[2]-O[2],p[1]-O[1]),tau),mod(atan(q[2]-O[2],q[1]-O[1]),tau)
            d = mod(t1-t0,tau)
            abs(d-T(pi)) > at || _circular_error("ambiguous semicircle at face $f")
            if d > T(pi)
                t0,t1 = t1,t0
                d = tau-d
            end
            d > at || _circular_error("degenerate or unresolved chord at face $f")
            radial = SVector(cos(t0+d/2),sin(t0+d/2),zero(T))
            n = mesh.face_normal[f]
            s = dot(n,radial) >= 0 ? 1 : -1
            norm(n-s*radial) <= 4at || _circular_error("invalid chord normal at face $f")
            abs(mesh.face_area[f]-norm(q-p)) <= 4tol &&
                norm(mesh.face_centre[f]-(p+q)/2) <= 4tol || _circular_error("inconsistent chord metrics at face $f")
            if orientation == 0
                orientation = s
            end
            s == (side == 1 ? orientation : -orientation) || _circular_error("inconsistent/opposing orientation at face $f")
            expected_length[I(f)] = R*d
            hi = t0+d
            if hi <= tau
                push!(intervals,(lo=t0,hi=hi,face=I(f)))
            else
                push!(intervals,(lo=t0,hi=tau,face=I(f)))
                push!(intervals,(lo=zero(T),hi=hi-tau,face=I(f)))
            end
        end
        sort!(intervals;by=x -> (x.lo,x.hi,x.face))
        abs(first(intervals).lo) <= at && abs(last(intervals).hi-tau) <= at || _circular_error("incomplete seam coverage")
        for j in 2:length(intervals)
            gap = intervals[j].lo-intervals[j-1].hi
            abs(gap) <= at || _circular_error(gap < 0 ? "duplicate coverage" : "gap in coverage")
        end
        abs(sum(x.hi-x.lo for x in intervals)-tau) <= 4at || _circular_error("total coverage is not one circle")
        push!(sides,intervals)
    end
    aa,bb,ca,cb = (I[] for _ in 1:4)
    lengths,los,his = (T[] for _ in 1:3)
    centroids,mids,vectors = (SVector{3,T}[] for _ in 1:3)
    moments = SMatrix{3,3,T,9}[]
    i,j = 1,1
    ia,ib = sides
    while i <= length(ia) && j <= length(ib)
        x,y = ia[i],ib[j]
        lo,hi = max(x.lo,y.lo),min(x.hi,y.hi)
        d = hi-lo
        if d > at
            mid = (lo+hi)/2
            radial = SVector(cos(mid),sin(mid),zero(T))
            S = orientation*(2R*sin(d/2))*radial
            C = O + (2R*sin(d/2)/d)*radial
            cc = d/2 + cos(2mid)*sin(d)/2
            ss = d-cc
            cs = sin(2mid)*sin(d)/2
            M = O*S' + orientation*R^2*SMatrix{3,3,T}(cc,cs,0,cs,ss,0,0,0,0)
            all(isfinite,S) && all(isfinite,C) && all(isfinite,M) &&
                all(isfinite,O+R*radial) && isfinite(R*d) || _circular_error("nonfinite overlap metrics")
            push!(aa,x.face); push!(bb,y.face)
            push!(ca,mesh.face_ownerCells[x.face][1]); push!(cb,mesh.face_ownerCells[y.face][1])
            ca[end] != cb[end] || _circular_error("interface owners must be distinct")
            push!(lengths,R*d); push!(los,lo); push!(his,hi)
            push!(centroids,C); push!(mids,O+R*radial); push!(vectors,S); push!(moments,M)
        end
        if x.hi < y.hi
            i += 1
        elseif y.hi < x.hi
            j += 1
        else
            i += 1; j += 1
        end
    end
    adj = _interface_adjacency(aa,bb)
    for row in eachindex(adj.IDs)
        f = adj.IDs[row]
        S = zero(SVector{3,T})
        covered = zero(T)
        for k in adj.ranges[row]
            o = adj.overlaps[k]
            S += adj.signs[k]*vectors[o]
            covered += lengths[o]
        end
        abs(covered-expected_length[f]) <= 16tol || _circular_error("incomplete arc coverage at face $f")
        norm(S-mesh.face_area[f]*mesh.face_normal[f]) <= 16tol || _circular_error("area-vector reconstruction failed at face $f")
    end
    length(adj.IDs) == length(fa)+length(fb) || _circular_error("uncovered face")
    CircularInterface(I(pa),I(pb),aa,bb,ca,cb,lengths,centroids,mids,vectors,moments,
        los,his,adj,_interface_adjacency(ca,cb),R,O)
end

construct_circular_interface(mesh::AbstractMesh,a::Symbol,b::Symbol;kwargs...) =
    _circular_error("only CPU Mesh2 supported")

interface_cell_flux!(out::AbstractVector,c::CircularInterface,flux::AbstractVector) =
    _gather_interface_flux!(out,c.cell_adjacency,flux,length(c.measure))
interface_face_flux!(out::AbstractVector,c::CircularInterface,flux::AbstractVector) =
    _gather_interface_flux!(out,c.face_adjacency,flux,length(c.measure))
