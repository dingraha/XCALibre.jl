using XCALibre
using Test
using LinearAlgebra
using StaticArrays

# Two independently segmented polygonal annuli. Radial edges are shared within
# each annulus; nodes at the common circle remain distinct between zones.
function circular_interface_fixture(; na=5,nb=8,phase=0.13,integer=Int64,float=Float64,
    origin=SVector(0.,0.,0.),radius=1.,angles_a=nothing,angles_b=nothing)
    P,line = XCALibre.Point,XCALibre.line!
    points = typeof(P(0.,0.,0.))[]
    edges = typeof(line([P(0.,0.,0.),P(1.,0.,0.)],1,2,1))[]
    blocks = XCALibre.Block{Int64}[]
    interface_a,interface_b,outside = Int64[],Int64[],Int64[]
    for (side,n,shift,angles) in ((1,na,0.,angles_a),(2,nb,phase,angles_b))
        ts = angles === nothing ? [shift+2pi*k/n for k in 0:n-1] : angles
        n = length(ts)
        offset = length(points)
        for r in (side == 1 ? (0.5radius,radius) : (radius,1.5radius)), t in ts
            x = origin + r*SVector(cos(t),sin(t),0.)
            push!(points,P(x...))
        end
        base = length(edges)
        for layer in 0:1, k in 1:n
            push!(edges,line(points,offset+layer*n+k,offset+layer*n+mod1(k+1,n),1))
        end
        for k in 1:n
            push!(edges,line(points,offset+k,offset+n+k,1))
        end
        for k in 1:n
            push!(blocks,quad(edges,[base+k,base+n+k,base+2n+k,base+2n+mod1(k+1,n)]))
        end
        append!(side == 1 ? interface_a : interface_b,base .+ (side == 1 ? (n+1:2n) : (1:n)))
        append!(outside,base .+ (side == 1 ? (1:n) : (n+1:2n)))
    end
    patches = [Patch(:interface_a,interface_a),Patch(:interface_b,interface_b),Patch(:outside,outside)]
    UNV2.update_mesh_format(generate!(MeshBuilder2D(points,edges,patches,blocks)),integer,float)
end

circular_coupling(mesh;kwargs...) = construct_circular_interface(mesh,:interface_a,:interface_b;
    centre=(0.,0.,0.),radius=1.,kwargs...)

@testset "Circular geometry, metrics and conservative gathers" begin
    for (I,T) in ((Int64,Float64),(Int32,Float64),(Int32,Float32)), phase in (0.,0.13,1.1,2pi-0.03)
        mesh = circular_interface_fixture(integer=I,float=T,phase=phase)
        before = deepcopy(mesh)
        c = circular_coupling(mesh)
        tol = 500eps(T)
        @test c isa CircularInterface
        @test eltype(c.face_a) == I && eltype(c.measure) == T
        @test sum(c.measure) ≈ T(2pi) atol=tol
        @test norm(sum(c.area_vector)) < tol
        @test all(>(0),c.measure)
        @test all(c.theta_hi .> c.theta_lo)
        @test all(norm(S) < l for (S,l) in zip(c.area_vector,c.measure))
        @test all(norm(x) ≈ 1 for x in c.midpoint)
        @test all(norm(x) <= 1+tol for x in c.centre)
        @test any(norm(x) < 0.99 for x in c.centre)
        @test sum(c.area_moment) ≈ T(pi)*SMatrix{3,3,T}(1,0,0,0,1,0,0,0,0) atol=tol
        @test mesh.node_coords == before.node_coords && mesh.face_area == before.face_area
        @test mesh.cell_volume == before.cell_volume && mesh.face_normal == before.face_normal
        for (faces,sign) in ((c.face_a,1),(c.face_b,-1)), f in unique(faces)
            ids = findall(==(f),faces)
            @test sign*sum(c.area_vector[ids]) ≈ mesh.face_area[f]*mesh.face_normal[f] atol=tol
            @test sum(c.measure[ids]) > mesh.face_area[f]
        end
        q = [T(isodd(k) ? 0.3 : -0.7) for k in eachindex(c.measure)]
        faces = fill(T(99),length(mesh.faces))
        cells = zeros(T,length(mesh.cells))
        interface_face_flux!(faces,c,q)
        interface_cell_flux!(cells,c,q)
        @test abs(sum(cells)) < tol*sum(abs,q)
        @test sum(faces[mesh.boundaries[1].IDs_range]) ≈ sum(q) atol=tol
        @test sum(faces[mesh.boundaries[2].IDs_range]) ≈ -sum(q) atol=tol
        @test all(==(99),faces[mesh.boundaries[3].IDs_range])
        ref = zeros(T,length(mesh.cells))
        for f in c.face_adjacency.IDs
            ref[mesh.face_ownerCells[f][1]] += faces[f]
        end
        @test cells ≈ ref atol=tol
        for _ in 1:3
            @test interface_cell_flux!(zeros(T,length(cells)),c,q) == cells
        end
        v = SVector{3,T}(0.7,-0.2,0.)
        physical = [dot(v,mesh.face_area[f]*mesh.face_normal[f]) for f in eachindex(mesh.faces)]
        interface_face_flux!(physical,c,[dot(v,S) for S in c.area_vector])
        balance = zeros(T,length(mesh.cells))
        # Count every physical face once, with opposite internal-face signs.
        for f in eachindex(physical)
            a,b = mesh.face_ownerCells[f]
            balance[a] += physical[f]
            a != b && (balance[b] -= physical[f])
        end
        @test maximum(abs,balance) < tol
        vector_flux = [SVector(x,2x,-x) for x in q]
        vf = fill(zero(SVector{3,T}),length(mesh.faces))
        interface_face_flux!(vf,c,vector_flux)
        @test sum(vf[c.face_adjacency.IDs]) ≈ zero(SVector{3,T}) atol=tol
        swapped = construct_circular_interface(mesh,:interface_b,:interface_a;centre=(0,0,0),radius=1)
        @test swapped.measure ≈ c.measure atol=tol
        @test swapped.area_vector ≈ -c.area_vector atol=tol
        @test swapped.area_moment ≈ -c.area_moment atol=tol
        @test_throws DimensionMismatch interface_cell_flux!(cells,c,[T(1)])
        @test_throws BoundsError interface_face_flux!(zeros(T,1),c,q)
        @test_throws ArgumentError interface_cell_flux!(q,c,q)
        @test_throws MethodError InterfaceTransportFlux(mesh,c)
        @test_throws ArgumentError construct_planar_interface(mesh,:interface_a,:interface_b)
    end
end

@testset "Circular seam, analytic moments and rebuild isolation" begin
    mesh = circular_interface_fixture(na=4,nb=4,phase=0.)
    c = circular_coupling(mesh)
    @test length(c.measure) == 4 # endpoint contacts do not add overlaps
    @test c.measure ≈ fill(pi/2,4)
    @test c.area_vector[1] ≈ SVector(1.,1.,0.)
    @test c.centre[1] ≈ SVector(2/pi,2/pi,0.)
    @test c.midpoint[1] ≈ SVector(sqrt(0.5),sqrt(0.5),0.)
    @test c.area_moment[1] ≈ @SMatrix [pi/4 0.5 0.; 0.5 pi/4 0.; 0. 0. 0.]
    # Arc moments differ from straight chord midpoint*area-vector moments.
    @test norm(c.area_moment[1]-mesh.face_centre[c.face_a[1]]*c.area_vector[1]') > 0.1
    # Fixed seam may split the SAME pair twice; neither is a duplicate overlap.
    wrap = circular_interface_fixture(na=4,nb=4,angles_a=[0.2,1.6,3.2,4.9],angles_b=[0.2,1.6,3.2,4.9])
    cw = circular_coupling(wrap)
    @test length(cw.measure) == 5
    @test (cw.face_a[1],cw.face_b[1]) == (cw.face_a[end],cw.face_b[end])
    nonuniform = circular_interface_fixture(angles_a=[0.,0.3,1.8,3.,4.6],angles_b=[0.12,0.8,2.1,3.3,4.9,5.7])
    @test sum(circular_coupling(nonuniform).measure) ≈ 2pi
    reversed = deepcopy(nonuniform)
    for f in [collect(reversed.boundaries[1].IDs_range);collect(reversed.boundaries[2].IDs_range)]
        reverse!(@view reversed.face_nodes[reversed.face_nodes_range[f]])
    end
    cn,cr = circular_coupling(nonuniform),circular_coupling(reversed)
    @test cr.face_a == cn.face_a && cr.face_b == cn.face_b
    @test cr.measure ≈ cn.measure
    shuffled = deepcopy(nonuniform)
    for r in (shuffled.boundaries[1].IDs_range,shuffled.boundaries[2].IDs_range), storage in
        (shuffled.face_ownerCells,shuffled.face_centre,shuffled.face_normal,shuffled.face_area,shuffled.face_nodes_range)
        storage[r] .= reverse(storage[r])
    end
    cs = circular_coupling(shuffled)
    @test cs.measure ≈ cn.measure
    @test cs.area_vector ≈ cn.area_vector
    O,R = SVector(2.,-3.,0.7),2.3
    translated = circular_interface_fixture(na=4,nb=4,phase=0.,origin=O,radius=R)
    ct = construct_circular_interface(translated,:interface_a,:interface_b;centre=O,radius=R)
    @test ct.measure ≈ R*c.measure
    @test ct.centre ≈ [O+R*x for x in c.centre]
    @test ct.area_vector ≈ R*c.area_vector
    @test ct.area_moment ≈ [O*(R*S)' + R^2*M for (S,M) in zip(c.area_vector,c.area_moment)]
    # Independent quadrature checks each stored moment and centroid.
    for k in eachindex(ct.measure)
        lo,hi = ct.theta_lo[k],ct.theta_hi[k]
        ts = [lo+(j-0.5)*(hi-lo)/2000 for j in 1:2000]
        normals = [SVector(cos(t),sin(t),0.) for t in ts]
        positions = [O+R*n for n in normals]
        @test sum(positions)/length(ts) ≈ ct.centre[k] atol=1e-7
        @test (R*(hi-lo)/length(ts))*sum(x*n' for (x,n) in zip(positions,normals)) ≈ ct.area_moment[k] atol=1e-6
    end
    newmesh = circular_interface_fixture(na=4,nb=4,phase=0.2)
    rebuilt = circular_coupling(newmesh)
    @test length(rebuilt.measure) != length(c.measure)
    @test c.measure ≈ fill(pi/2,4) # old object unchanged
    @test rebuilt.measure !== c.measure && rebuilt.face_adjacency.IDs !== c.face_adjacency.IDs
end

@testset "Circular invalid geometry" begin
    mesh = circular_interface_fixture()
    @test_throws ArgumentError circular_coupling(mesh;atol=-1)
    @test_throws ArgumentError circular_coupling(mesh;rtol=NaN)
    @test_throws ArgumentError circular_coupling(mesh;atol=2)
    for radius in (0.,-1.,NaN,Inf), origin in ((0.,0.,0.),)
        @test_throws ArgumentError construct_circular_interface(mesh,:interface_a,:interface_b;centre=origin,radius)
    end
    @test_throws ArgumentError construct_circular_interface(mesh,:missing,:interface_b;centre=(0,0,0),radius=1)
    @test_throws ArgumentError construct_circular_interface(mesh,:interface_a,:interface_a;centre=(0,0,0),radius=1)
    for origin in ((0.,0.),(NaN,0.,0.),(0.1,0.,0.))
        @test_throws ArgumentError construct_circular_interface(mesh,:interface_a,:interface_b;centre=origin,radius=1)
    end
    f = first(mesh.boundaries[2].IDs_range)
    for mutate! in (
        m -> (m.face_normal[f] *= -1),
        m -> (m.face_area[f] *= 1.1),
        m -> (m.face_centre[f] += SVector(0.1,0.,0.)),
        m -> (m.node_coords[m.face_nodes[first(m.face_nodes_range[f])]] *= 0.9),
        m -> (m.node_coords[m.face_nodes[first(m.face_nodes_range[f])]] += SVector(0.,0.,0.1)),
        m -> (m.node_coords[m.face_nodes[first(m.face_nodes_range[f])]] = SVector(NaN,0.,0.)),
        m -> (m.face_nodes_range[f] = m.face_nodes_range[f+1]),
        m -> (m.boundaries[2] = Boundary(:interface_b,(f+1):last(m.boundaries[2].IDs_range))),
        m -> (m.boundaries[2] = Boundary(:interface_b,1:0)))
        bad = deepcopy(mesh)
        mutate!(bad)
        @test_throws ArgumentError circular_coupling(bad)
    end
    # Duplicate a complete valid face's metrics so rejection is due to coverage.
    bad = deepcopy(mesh)
    for storage in (bad.face_nodes_range,bad.face_area,bad.face_normal,bad.face_centre)
        storage[f+1] = storage[f]
    end
    @test_throws ArgumentError circular_coupling(bad)
    semi = circular_interface_fixture(na=4,nb=4)
    ff = first(semi.boundaries[1].IDs_range)
    nr = semi.face_nodes_range[ff]
    semi.node_coords[semi.face_nodes[last(nr)]] = -semi.node_coords[semi.face_nodes[first(nr)]]
    @test_throws ArgumentError circular_coupling(semi)
end
