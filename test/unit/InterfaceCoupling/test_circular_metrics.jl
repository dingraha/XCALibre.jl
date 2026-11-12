using XCALibre
using Test
using LinearAlgebra
using StaticArrays
isdefined(@__MODULE__,:circular_interface_fixture) || include("test_circular_interface.jl")

@testset "Circular corrected volumes and affine pressure gradients" begin
    for (I,T) in ((Int64,Float64),(Int32,Float64),(Int32,Float32)), phase in (0.,0.13,1.2),
        origin in (SVector(0.,0.,0.),SVector(2.,-3.,0.7))
        R = 1.3
        mesh = circular_interface_fixture(integer=I,float=T,phase=phase,origin=origin,radius=R)
        c = construct_circular_interface(mesh,:interface_a,:interface_b;centre=origin,radius=R)
        before = copy(mesh.cell_volume)
        m = construct_circular_metrics(mesh,c)
        tol = 3000eps(T)
        P = SMatrix{3,3,T}(1,0,0,0,1,0,0,0,0)
        @test m.coupling !== c && m.coupling.measure !== c.measure
        @test mesh.cell_volume == before
        @test m.volume ≈ before+m.volume_correction
        @test all(>(0),m.volume)
        @test all(isapprox(M,dv*P;atol=tol) for (M,dv) in zip(m.moment_correction,m.volume_correction))
        # Annular polygon volume + correction equals a circle at the interface
        # and the retained polygon at the other radial boundary.
        left = unique(c.cell_a); right = unique(c.cell_b)
        @test all(>(0),m.volume_correction[left])
        @test all(<(0),m.volume_correction[right])
        inner = 5*(0.5R)^2*sin(2pi/5)/2
        outer = 8*(1.5R)^2*sin(2pi/8)/2
        @test sum(m.volume[left]) ≈ pi*R^2-inner atol=tol
        @test sum(m.volume[right]) ≈ outer-pi*R^2 atol=tol
        integral = fill(zero(SVector{3,T}),length(c.measure))
        out = fill(SVector{3,T}(99,99,99),length(mesh.cells))
        for g in (SVector{3,T}(0,0,0),SVector{3,T}(1,0,0),SVector{3,T}(0,1,0),SVector{3,T}(0.7,-0.3,0))
            facep = [T(2)+dot(g,x-c.origin) for x in mesh.face_centre]
            facep[m.interface_face] .= NaN # ignored rather than added/subtracted
            circular_pressure_area!(integral,m.coupling,T(2),g)
            circular_gauss_gradient!(out,m,facep,integral)
            @test all(x -> isapprox(x,g;atol=tol,rtol=tol),out)
            baseline = copy(out)
            @test circular_gauss_gradient!(out,m,facep,integral) == baseline
            facep .+= T(3)
            circular_pressure_area!(integral,m.coupling,T(5),g)
            circular_gauss_gradient!(out,m,facep,integral)
            @test out ≈ baseline atol=tol rtol=tol
            sums = fill(zero(SVector{3,T}),length(mesh.cells))
            interface_cell_flux!(sums,m.coupling,integral)
            @test norm(sum(sums)) < tol
            if !iszero(norm(g))
                # The exact numerator divided by polygon volume is NOT affine-exact.
                wrong = [x*v/p for (x,v,p) in zip(out,m.volume,m.polygon_volume)]
                @test maximum(norm(x-g) for x in wrong) > 0.01
            end
        end
        # Using p(arc centroid)*S omits the normal/position correlation.
        g = SVector{3,T}(1,0,0)
        approximate = [dot(g,x-c.origin)*S for (x,S) in zip(c.centre,c.area_vector)]
        circular_pressure_area!(integral,c,T(0),g)
        @test maximum(norm(x-y) for (x,y) in zip(integral,approximate)) > 0.001
        # Mutating original mesh/coupling cannot silently alter the snapshot.
        c.measure[1] *= T(1.1)
        mesh.cell_volume[1] *= T(1.1)
        @test m.polygon_volume == before
        @test m.coupling.measure[1] != c.measure[1]
    end
end

@testset "Circular metric seam and validation atomicity" begin
    mesh = circular_interface_fixture(na=4,nb=4,angles_a=[0.2,1.6,3.2,4.9],angles_b=[0.2,1.6,3.2,4.9])
    c = circular_coupling(mesh)
    m = construct_circular_metrics(mesh,c)
    f = c.face_a[1]
    ids = findall(==(f),c.face_a)
    @test length(ids) == 2
    d = sum(c.theta_hi[ids]-c.theta_lo[ids])
    cell = mesh.face_ownerCells[f][1]
    @test m.volume_correction[cell] ≈ (d-sin(d))/2
    @test abs(m.volume_correction[cell]-sum((c.theta_hi[k]-c.theta_lo[k]-sin(c.theta_hi[k]-c.theta_lo[k]))/2 for k in ids)) > 0.001
    for mutate! in (x -> (x.cell_volume[1]*=1.1),x -> (x.cell_volume[1]=0),
        x -> (x.face_normal[end]*=-1),x -> (x.face_centre[end]+=SVector(0.1,0,0)),
        x -> (x.node_coords[1]+=SVector(0.1,0,0)))
        bad = deepcopy(mesh); mutate!(bad)
        @test_throws ArgumentError construct_circular_metrics(bad,c)
    end
    modified = deepcopy(c); modified.area_moment[1] *= 1.1
    @test_throws ArgumentError construct_circular_metrics(mesh,modified)
    @test_throws ArgumentError construct_circular_metrics(mesh,c;atol=-1)
    @test_throws ArgumentError construct_circular_metrics(mesh,c;rtol=NaN)
    integral = fill(SVector(9.,9.,9.),length(c.measure))
    saved = copy(integral)
    @test_throws ArgumentError circular_pressure_area!(integral,c,NaN,(1,0,0))
    @test integral == saved
    @test_throws ArgumentError circular_pressure_area!(integral,c,1,(1,0,1))
    @test_throws ArgumentError circular_pressure_area!(c.area_vector,c,1,(1,0,0))
    @test_throws DimensionMismatch circular_pressure_area!(integral[1:1],c,1,(1,0,0))
    facep = ones(length(mesh.faces))
    circular_pressure_area!(integral,c,1,(0,0,0))
    out = fill(SVector(9.,9.,9.),length(mesh.cells)); savedout = copy(out)
    facep[end] = NaN
    @test_throws ArgumentError circular_gauss_gradient!(out,m,facep,integral)
    @test out == savedout
    facep[end] = 1
    integral[1] = SVector(Inf,0.,0.)
    @test_throws ArgumentError circular_gauss_gradient!(out,m,facep,integral)
    @test out == savedout
    integral[1] = c.area_vector[1]
    @test_throws ArgumentError circular_gauss_gradient!(@view(m.face_centre[1:length(out)]),m,facep,integral)
    @test_throws DimensionMismatch circular_gauss_gradient!(out[1:1],m,facep,integral)
    # Reference-point changes describe the same affine pressure exactly.
    g,r = SVector(0.7,-0.2,0.),SVector(2.,3.,0.)
    circular_pressure_area!(integral,c,2.,g)
    expected = copy(integral)
    circular_pressure_area!(integral,c,2.0 + dot(g,r-c.origin),g;reference=r)
    @test integral ≈ expected atol=1e-13
    # Nonuniform subdivisions also satisfy full-cell affine identities.
    nonuniform = circular_interface_fixture(angles_a=[0.,0.3,1.8,3.,4.6],angles_b=[0.12,0.8,2.1,3.3,4.9,5.7])
    cn = circular_coupling(nonuniform)
    mn = construct_circular_metrics(nonuniform,cn)
    pn = [2.0 + dot(g,x) for x in nonuniform.face_centre]
    qn = fill(zero(g),length(cn.measure))
    gn = fill(zero(g),length(mn.volume))
    circular_pressure_area!(qn,cn,2.,g)
    circular_gauss_gradient!(gn,mn,pn,qn)
    @test all(x -> isapprox(x,g;atol=1e-12),gn)
    # Matching overlap counts do not make geometry interchangeable.
    other = circular_interface_fixture(na=4,nb=4,angles_a=[0.21,1.61,3.21,4.91],angles_b=[0.21,1.61,3.21,4.91])
    @test_throws ArgumentError construct_circular_metrics(other,c)
    corrupt = deepcopy(m); corrupt.volume[1] = -1
    @test_throws ArgumentError circular_gauss_gradient!(out,corrupt,facep,integral)
    @test out == savedout
end
