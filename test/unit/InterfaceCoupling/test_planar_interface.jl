using XCALibre
using Test
using LinearAlgebra
using StaticArrays

# Duplicate interface coordinates but distinct node/edge IDs keep the two blocks disconnected.
function planar_interface_fixture(; integer=Int64, float=Float64, na=2, nb=3)
    P, line = XCALibre.Point, XCALibre.line!
    pts = [P(x,y,0.0) for (x,y) in ((0.0,0.0), (1.0,0.0), (0.0,1.0), (1.0,1.0),
        (1.0,0.0), (2.0,0.0), (1.0,1.0), (2.0,1.0))]
    edges = [line(pts,1,2,1), line(pts,3,4,1), line(pts,1,3,na), line(pts,2,4,na),
        line(pts,5,6,1), line(pts,7,8,1), line(pts,5,7,nb), line(pts,6,8,nb)]
    patches = [Patch(:interface_a,[4]), Patch(:interface_b,[7]),
        Patch(:outside,[1,2,3,5,6,8])]
    blocks = [quad(edges,[1,2,3,4]), quad(edges,[5,6,7,8])]
    UNV2.update_mesh_format(generate!(MeshBuilder2D(pts,edges,patches,blocks)), integer, float)
end

@testset "Static planar interface" begin
    mesh = planar_interface_fixture()
    a, b = mesh.boundaries[1].IDs_range, mesh.boundaries[2].IDs_range
    @test length(a) == 2 && length(b) == 3
    left = Set(mesh.face_ownerCells[f][1] for f in a)
    right = Set(mesh.face_ownerCells[f][1] for f in b)
    @test isempty(intersect(left, right))
    @test all(zip(mesh.cell_faces, mesh.cell_neighbours)) do (f, neighbour)
        c1, c2 = mesh.face_ownerCells[f]
        (c1 in left && c2 in left) || (c1 in right && c2 in right)
    end
    @test isempty(intersect(Set(mesh.face_nodes[k] for f in a for k in mesh.face_nodes_range[f]),
        Set(mesh.face_nodes[k] for f in b for k in mesh.face_nodes_range[f])))

    c = construct_planar_interface(mesh, :interface_a, :interface_b)
    @test c.face_a == [a[1],a[1],a[2],a[2]]
    @test c.face_b == [b[1],b[2],b[2],b[3]]
    @test c.measure ≈ [1/3,1/6,1/6,1/3]
    @test [x[2] for x in c.centre] ≈ [1/6,5/12,7/12,5/6]
    @test sum(c.measure) ≈ 1
    @test all(v ≈ SVector(l,0.0,0.0) for (v,l) in zip(c.area_vector,c.measure))

    flux = [0.3,-0.7,1.1,-0.2]
    balances = zeros(length(mesh.cells))
    faces = fill(99.0,length(mesh.faces))
    interface_cell_flux!(balances,c,flux)
    interface_face_flux!(faces,c,flux)
    @test abs(sum(balances)) < 10eps(Float64)*sum(abs,flux)
    @test sum(faces[a]) ≈ sum(flux)
    @test sum(faces[b]) ≈ -sum(flux)
    reference = zeros(length(balances))
    for f in [collect(a);collect(b)]
        reference[mesh.face_ownerCells[f][1]] += faces[f]
    end
    @test reference ≈ balances
    @test all(faces[f] == 99 for f in setdiff(eachindex(faces),[collect(a);collect(b)]))
    for _ in 1:10
        @test interface_cell_flux!(zeros(length(balances)),c,flux) == balances
    end
    @test_throws DimensionMismatch interface_cell_flux!(balances,c,[1.0])
    @test_throws BoundsError interface_face_flux!(zeros(1),c,flux)
    @test_throws ArgumentError interface_cell_flux!(flux,c,flux)

    interface_face_flux!(faces,c,2 .* c.measure)
    @test faces[a] ≈ 2 .* mesh.face_area[a]
    @test faces[b] ≈ -2 .* mesh.face_area[b]
    for (side, sign) in ((a,1),(b,-1)), f in side
        ids = findall(==(f), sign == 1 ? c.face_a : c.face_b)
        @test sum(c.measure[ids]) ≈ mesh.face_area[f]
        @test sign*sum(c.area_vector[ids]) ≈ mesh.face_area[f]*mesh.face_normal[f]
    end

    # Reversing endpoint order and face order must not affect geometric overlaps.
    reversed = deepcopy(mesh)
    for f in [collect(a);collect(b)]
        reverse!(@view reversed.face_nodes[reversed.face_nodes_range[f]])
    end
    cr = construct_planar_interface(reversed,:interface_a,:interface_b)
    @test cr.face_a == c.face_a && cr.face_b == c.face_b
    @test cr.centre ≈ c.centre && cr.measure ≈ c.measure
    shuffled = deepcopy(mesh)
    for storage in (shuffled.face_ownerCells, shuffled.face_centre, shuffled.face_normal,
        shuffled.face_e, shuffled.face_area, shuffled.face_delta, shuffled.face_weight,
        shuffled.face_nodes_range, shuffled.boundary_cellsID)
        storage[a] .= reverse(storage[a])
        storage[b] .= reverse(storage[b])
    end
    cs = construct_planar_interface(shuffled,:interface_a,:interface_b)
    @test cs.centre ≈ c.centre && cs.measure ≈ c.measure
    swapped = construct_planar_interface(mesh,:interface_b,:interface_a)
    @test swapped.measure ≈ c.measure
    @test swapped.area_vector ≈ -c.area_vector
    # The common line need not be axis-aligned or pass through the origin.
    rotated = deepcopy(mesh)
    theta = 0.37
    R = @SMatrix [cos(theta) -sin(theta) 0.0; sin(theta) cos(theta) 0.0; 0.0 0.0 1.0]
    shift = SVector(3.0,-2.0,0.0)
    rotated.node_coords .= [R*p + shift for p in mesh.node_coords]
    rotated.face_normal .= [R*n for n in mesh.face_normal]
    ct = construct_planar_interface(rotated,:interface_a,:interface_b)
    @test ct.measure ≈ c.measure
    @test ct.centre ≈ [R*p + shift for p in c.centre]
    @test ct.area_vector ≈ [R*v for v in c.area_vector]
    extra = fill(17.0,length(mesh.cells)+1)
    interface_cell_flux!(extra,c,flux)
    @test extra[end] == 17.0
    @test extra[1:end-1] == balances

    # Matched subdivisions share endpoints, which must not add zero-length elements.
    matched = planar_interface_fixture(na=2,nb=2)
    cm = construct_planar_interface(matched,:interface_a,:interface_b)
    @test cm.measure == [0.5,0.5]
    @test length(cm.face_a) == 2
    @test all(>(0),cm.measure)
    # Consistent individual edges can still contain duplicate coverage.
    duplicate = deepcopy(mesh)
    duplicate.face_nodes_range[b[2]] = duplicate.face_nodes_range[b[1]]
    duplicate.face_area[b[2]] = duplicate.face_area[b[1]]
    err = try
        construct_planar_interface(duplicate,:interface_a,:interface_b)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("duplicate coverage",sprint(showerror,err))
    # Shorten the top B edge and update its length: individual geometry is valid,
    # but A is no longer fully covered. Do not hide the missing portion.
    uncovered = deepcopy(mesh)
    lastnr = uncovered.face_nodes_range[last(b)]
    uncovered.node_coords[uncovered.face_nodes[last(lastnr)]] -= SVector(0.0,0.1,0.0)
    uncovered.face_area[last(b)] -= 0.1
    err = try
        construct_planar_interface(uncovered,:interface_a,:interface_b)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("incomplete coverage",sprint(showerror,err))

    @test_throws ArgumentError construct_planar_interface(mesh,:missing,:interface_b)
    @test_throws ArgumentError construct_planar_interface(mesh,:interface_a,:interface_a)
    @test_throws ArgumentError construct_planar_interface(mesh,:interface_a,:interface_b; atol=-1)
    @test_throws ArgumentError construct_planar_interface(mesh,:interface_a,:interface_b; rtol=NaN)
    empty = deepcopy(mesh)
    empty.boundaries[1] = Boundary(:interface_a,1:0)
    @test_throws ArgumentError construct_planar_interface(empty,:interface_a,:interface_b)
    wrongnormal = deepcopy(mesh)
    wrongnormal.face_normal[b[1]] *= -1
    @test_throws ArgumentError construct_planar_interface(wrongnormal,:interface_a,:interface_b)
    gap = deepcopy(mesh)
    gap.node_coords[gap.face_nodes[first(gap.face_nodes_range[b[1]])]] += SVector(0.0,0.1,0.0)
    @test_throws ArgumentError construct_planar_interface(gap,:interface_a,:interface_b)
    offset = deepcopy(mesh)
    for n in unique(mesh.face_nodes[k] for f in b for k in mesh.face_nodes_range[f])
        offset.node_coords[n] += SVector(0.1,0.0,0.0)
    end
    @test_throws ArgumentError construct_planar_interface(offset,:interface_a,:interface_b)
    degenerate = deepcopy(mesh)
    nr = degenerate.face_nodes_range[a[1]]
    degenerate.node_coords[degenerate.face_nodes[last(nr)]] = degenerate.node_coords[degenerate.face_nodes[first(nr)]]
    @test_throws ArgumentError construct_planar_interface(degenerate,:interface_a,:interface_b)

    for (I,T) in ((Int32,Float64),(Int32,Float32))
        narrow = planar_interface_fixture(integer=I,float=T)
        cn = construct_planar_interface(narrow,:interface_a,:interface_b)
        @test eltype(cn.face_a) == I
        @test eltype(cn.cell_adjacency.overlaps) == I
        @test eltype(cn.measure) == T
        @test eltype(cn.centre) == SVector{3,T}
        @test cn.measure ≈ T.([1/3,1/6,1/6,1/3])
        @test adapt(CPU(),cn).measure == cn.measure
    end
end
