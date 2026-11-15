module CircularSquareGeometry

using XCALibre, StaticArrays, LinearAlgebra
using Accessors: @set

"""Filled circular inner zone in a periodic square, geometry only.

na/nb must be multiples of four. Inner core is a rotated tensor-grid square;
four-sided O-grids connect it to the circle and the circle to the outer square.
Interface chords are independent; exact curved metrics fill the chord gaps.
phase rotates ONLY the inner zone. Ordinary exterior square pairs conform.
No transient solver, periodic circular reconstruction, motion or ALE implied.
"""
function build(; na=16, nb=20, nra=3, nrb=3, phase=0.0,
    L=1.0, radius=0.25, core_half=0.10, stitched=false)
    all(n -> n isa Integer && n >= 8 && n%4 == 0, (na,nb)) ||
        throw(ArgumentError("na/nb must be multiples of four >= 8"))
    all(n -> n isa Integer && n > 0, (nra,nrb)) || throw(ArgumentError("positive layer counts required"))
    all(isfinite,(phase,L,radius,core_half)) && 0 < sqrt(2)*core_half < radius < L/2 ||
        throw(ArgumentError("require 0 < sqrt(2)*core_half < radius < L/2"))
    stitched && (na != nb || phase != 0) && throw(ArgumentError("stitched control requires na==nb and phase==0"))
    P = XCALibre.Point
    points = typeof(P(0.,0.,0.))[]
    edges = typeof(line!([P(0.,0.,0.),P(1.,0.,0.)],1,2,1))[]
    blocks = XCALibre.Block{Int64}[]
    edge_map = Dict{Tuple{Int,Int},Int}()
    origin = SVector(L/2,L/2,0.)
    function node(x,y,angle=0.)
        c,s = cos(angle),sin(angle)
        push!(points,P(L/2+c*x-s*y,L/2+s*x+c*y,0.))
        length(points)
    end
    function edge(a,b)
        key = minmax(a,b)
        get!(edge_map,key) do
            push!(edges,line!(points,a,b,1)); length(edges)
        end
    end
    # quad uses endpoint ordering; each small block has no interior builder points.
    function cell(a,b,c,d)
        block = quad(edges,[edge(a,b),edge(d,c),edge(a,d),edge(b,c)])
        # A shared edge may have been created in the opposite traversal direction.
        # Explicit corner ordering avoids quad's endpoint overwrite in that case.
        block.nodesID[:,1] .= (a,b)
        block.nodesID[:,2] .= (d,c)
        push!(blocks,block)
    end
    # Core tensor grid, with boundary nodes matching equal-angle O-grid rays.
    m = na÷4
    v = [core_half*tan(-pi/4+j*pi/(2m)) for j in 0:m]
    ids = [node(x,y,phase) for x in v, y in v]
    for j in 1:m, i in 1:m
        cell(ids[i,j],ids[i+1,j],ids[i+1,j+1],ids[i,j+1])
    end
    core = vcat([ids[end,j] for j in 1:m], [ids[i,end] for i in m+1:-1:2],
        [ids[1,j] for j in m+1:-1:2], [ids[i,1] for i in 1:m])
    inner_interface = Int[]
    previous = core
    for layer in 1:nra
        alpha = layer/nra
        ring = Int[]
        for j in 0:na-1
            theta = -pi/4+2pi*j/na
            r0 = core_half/max(abs(cos(theta)),abs(sin(theta)))
            r = (1-alpha)*r0+alpha*radius
            push!(ring,node(r*cos(theta),r*sin(theta),phase))
        end
        for j in 1:na
            k = mod1(j+1,na)
            cell(previous[j],ring[j],ring[k],previous[k])
            layer==nra && push!(inner_interface,edge(ring[j],ring[k]))
        end
        previous = ring
    end
    # Outer grid uses fixed angles and square corners at quadrant boundaries.
    outer_interface = Int[]
    exterior = [Int[] for _ in 1:4] # right, top, left, bottom
    previous = stitched ? previous : [node(radius*cos(-pi/4+2pi*j/nb),radius*sin(-pi/4+2pi*j/nb)) for j in 0:nb-1]
    if !stitched
        for j in 1:nb
            push!(outer_interface,edge(previous[j],previous[mod1(j+1,nb)]))
        end
    end
    for layer in 1:nrb
        alpha = layer/nrb
        ring = Int[]
        for j in 0:nb-1
            theta = -pi/4+2pi*j/nb
            rsquare = (L/2)/max(abs(cos(theta)),abs(sin(theta)))
            r = (1-alpha)*radius+alpha*rsquare
            push!(ring,node(r*cos(theta),r*sin(theta)))
        end
        for j in 1:nb
            k = mod1(j+1,nb)
            cell(previous[j],ring[j],ring[k],previous[k])
            layer==nrb && push!(exterior[(j-1)÷(nb÷4)+1],edge(ring[j],ring[k]))
        end
        previous = ring
    end
    patches = [Patch(:interface_a,inner_interface),Patch(:interface_b,outer_interface),
        Patch(:right,exterior[1]),Patch(:top,exterior[2]),Patch(:left,exterior[3]),Patch(:bottom,exterior[4])]
    stitched && (patches = patches[3:end])
    builder = MeshBuilder2D(points,edges,patches,blocks)
    # The stock multiblock connector handles opposite local edge slots only.
    # Core/O-grid joins also meet at perpendicular slots: build owners directly
    # from exact node IDs, never from proximity or coordinate merging.
    BM = XCALibre.BlockMesher2D
    polygon = BM.build!(builder)
    incidence = Dict{Tuple{Int,Int},Vector{Int}}()
    for (i,c) in enumerate(polygon.cells)
        ns = c.nodesID
        for j in eachindex(ns)
            push!(get!(incidence,minmax(ns[j],ns[mod1(j+1,length(ns))]),Int[]),i)
        end
    end
    for (f,face) in enumerate(polygon.faces)
        owners = incidence[minmax(face.nodesID...)]
        length(owners) in (1,2) || error("nonmanifold edge")
        polygon.faces[f] = @set face.ownerCells = SVector(first(owners),last(owners))
    end
    BM.assign_cellsID_to_boundaries!(polygon,builder)
    BM.assign_nodesID_to_boundaries!(polygon,builder)
    BM.assign_facesID_to_cells!(polygon,builder)
    BM.assign_neighbours_to_cells!(polygon,builder)
    BM.geometry!(polygon)
    mesh = XCALibre.UNV2.update_mesh_format(polygon,Int64,Float64)
    coupling = stitched ? nothing : construct_circular_interface(mesh,:interface_a,:interface_b;centre=origin,radius)
    metrics = stitched ? nothing : construct_circular_metrics(mesh,coupling)
    raw_periodic = (construct_periodic(mesh,CPU(),:left,:right)...,
        construct_periodic(mesh,CPU(),:bottom,:top)...)
    # Resolve placeholder IDs/ranges without assigning fake BCs to the interface.
    periodic = map(raw_periodic) do bc
        id = findfirst(b -> b.name==bc.ID,mesh.boundaries)
        ctor = bc isa PeriodicParent ? PeriodicParent : Periodic
        ctor(id,bc.value,mesh.boundaries[id].IDs_range)
    end
    inner_cells = 1:m^2+na*nra
    outer_cells = last(inner_cells)+1:length(mesh.cells)
    (;mesh,coupling,metrics,periodic,inner_cells,outer_cells,origin,L,radius,phase,na,nb,nra,nrb,stitched)
end

end
