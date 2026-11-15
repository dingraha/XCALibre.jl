module CircularSquareGeometryTests
using Test, XCALibre, LinearAlgebra, StaticArrays
include(pkgdir(XCALibre,"examples","interface_verification","circular_square_setup.jl"))
const CS = CircularSquareGeometry

function components(adjacency)
    remaining = Set(eachindex(adjacency)); count = 0
    while !isempty(remaining)
        count += 1; queue = [first(remaining)]
        while !isempty(queue)
            i = pop!(queue)
            i in remaining || continue
            delete!(remaining,i)
            append!(queue,adjacency[i])
        end
    end
    count
end

function check(q)
    mesh,c,h = q.mesh,q.coupling,q.metrics
    ncell = length(mesh.cells)
    expected = (q.na÷4)^2+q.na*q.nra+q.nb*q.nrb
    @test ncell == expected
    @test all(x -> isfinite(x) && x>0,mesh.cell_volume)
    @test all(>(0),h.volume)
    @test all(x -> iszero(x[3]),mesh.node_coords)
    @test all(x -> 0 <= x[1] <= q.L+1e-14 && 0 <= x[2] <= q.L+1e-14,mesh.node_coords)
    # Independent signed shoelace + strict convexity, not abs(area) hiding inversion.
    for (i,cell) in enumerate(mesh.cells)
        ns = mesh.cell_nodes[mesh.cell_nodes_range[i]]
        p = mesh.node_coords[ns]
        area = sum(p[j][1]*p[mod1(j+1,4)][2]-p[j][2]*p[mod1(j+1,4)][1] for j in 1:4)/2
        @test area > 0
        @test area ≈ mesh.cell_volume[i] atol=2e-15*q.L^2 rtol=1e-11
        @test all(cross(p[mod1(j+1,4)]-p[j],p[mod1(j+2,4)]-p[mod1(j+1,4)])[3]>0 for j in 1:4)
    end
    @test sum(h.volume[q.inner_cells]) ≈ pi*q.radius^2 atol=2e-13*q.L^2
    @test sum(h.volume[q.outer_cells]) ≈ q.L^2-pi*q.radius^2 atol=2e-13*q.L^2
    @test sum(h.volume) ≈ q.L^2 atol=2e-13*q.L^2
    # Actual centre is filled: it lies in or on at least one core cell.
    contains_origin = false
    for i in 1:(q.na÷4)^2
        p = mesh.node_coords[mesh.cell_nodes[mesh.cell_nodes_range[i]]]
        contains_origin |= all(cross(p[mod1(j+1,4)]-p[j],q.origin-p[j])[3]>=-1e-14 for j in 1:4)
    end
    @test contains_origin
    graph = [Int[] for _ in 1:ncell]
    for (a,b) in mesh.face_ownerCells
        if a != b
            push!(graph[a],b);push!(graph[b],a)
        end
    end
    @test components(graph) == 2 # separate zones before virtual coupling
    for (a,b) in zip(c.cell_a,c.cell_b)
        push!(graph[a],b);push!(graph[b],a)
    end
    @test components(graph) == 1
    # Integer topology certificates: disk core/inner zone, annular outer zone.
    for (cells,euler) in ((q.inner_cells,1),(q.outer_cells,0))
        nodes = Set{Int}(); edgeowners = Dict{Tuple{Int,Int},Int}()
        for i in cells
            ns = mesh.cell_nodes[mesh.cell_nodes_range[i]]
            union!(nodes,ns)
            for j in 1:4
                edge = minmax(ns[j],ns[mod1(j+1,4)])
                edgeowners[edge] = get(edgeowners,edge,0)+1
            end
        end
        @test length(nodes)-length(edgeowners)+length(cells) == euler
        @test all(x -> x in (1,2),values(edgeowners))
    end
    @test sum(c.measure) ≈ 2pi*q.radius atol=2e-13*q.L
    @test all(>(0),c.measure)
    @test norm(sum(c.area_vector)) < 2e-13*q.L
    for bc in q.periodic
        bc isa PeriodicParent || continue
        @test length(unique(bc.value.face_map)) == length(bc.IDs_range)
        for (f,g) in zip(bc.IDs_range,bc.value.face_map)
            @test mesh.face_centre[f]+bc.value.transform.distance ≈ mesh.face_centre[g] atol=2e-14*q.L
            @test mesh.face_area[f]*mesh.face_normal[f] ≈ -mesh.face_area[g]*mesh.face_normal[g] atol=2e-14*q.L
        end
    end
    # Independent surface moments: exact affine arc pressure integrals and chord faces.
    integral = fill(zero(SVector{3,Float64}),length(c.measure))
    gradient = fill(zero(SVector{3,Float64}),ncell)
    for g in (SVector(0.,0.,0.),SVector(1.,0.,0.),SVector(0.,1.,0.))
        pface = [2+dot(g,x-c.origin) for x in mesh.face_centre]
        pface[h.interface_face] .= NaN
        circular_pressure_area!(integral,c,2.,g)
        circular_gauss_gradient!(gradient,h,pface,integral)
        @test maximum(norm.(gradient .- Ref(g))) < 5e-11
    end
    (; cells=ncell,overlaps=length(c.measure),polygon_area=sum(mesh.cell_volume),
        curved_area=sum(h.volume),min_volume=minimum(h.volume))
end

@testset "Filled-circle / periodic-square geometry" begin
    for scale in (1,2,4), (matched,phase) in ((true,0.),(false,0.),(false,0.13),(false,1.2))
        q = CS.build(na=16scale,nb=(matched ? 16 : 20)*scale,nra=3scale,nrb=3scale,phase=phase)
        r = check(q)
        if matched
            @test abs(r.polygon_area-q.L^2) < 2e-13
        end
        println("geometry scale=$scale matched=$matched phase=$phase $r")
    end
    q = CS.build(na=12,nb=16,nra=2,nrb=2,phase=0.23,L=2.,radius=0.5,core_half=0.2)
    check(q)
    for kwargs in ((;na=10),(;nra=0),(;phase=NaN),(;radius=0.51),(;core_half=0.2))
        @test_throws ArgumentError CS.build(;kwargs...)
    end
end
end
