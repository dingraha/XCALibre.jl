using XCALibre
using StaticArrays
using LinearAlgebra

# Shared by the runnable examples; does not run a simulation on include.
function channel_case(; na=8, nb=9, nx=4, interface_x=0.8, split=true,
    nu=0.1, relaxU=0.7, relaxp=0.3, initialU=0.7, initialp=0.0,
    iterations=1500, convergence=1e-8)
    all(n -> n isa Integer && n > 0, (na,nb,nx)) || throw(ArgumentError("cell counts must be positive integers"))
    0 < interface_x < 2 || throw(ArgumentError("interface_x must lie inside (0,2)"))
    if split
        points = [Point(x,y,0.) for (x,y) in ((0.,0.),(interface_x,0.),(0.,1.),(interface_x,1.),
            (interface_x,0.),(2.,0.),(interface_x,1.),(2.,1.))]
        edges = [line!(points,1,2,nx),line!(points,3,4,nx),line!(points,1,3,na),line!(points,2,4,na),
            line!(points,5,6,nx),line!(points,7,8,nx),line!(points,5,7,nb),line!(points,6,8,nb)]
        patches = [Patch(:interface_a,[4]),Patch(:interface_b,[7]),Patch(:inlet,[3]),Patch(:outlet,[8]),Patch(:walls,[1,2,5,6])]
        blocks = [quad(edges,[1,2,3,4]),quad(edges,[5,6,7,8])]
    else
        points = [Point(x,y,0.) for (x,y) in ((0.,0.),(2.,0.),(0.,1.),(2.,1.))]
        edges = [line!(points,1,2,2nx),line!(points,3,4,2nx),line!(points,1,3,nb),line!(points,2,4,nb)]
        patches = [Patch(:inlet,[3]),Patch(:outlet,[4]),Patch(:walls,[1,2])]
        blocks = [quad(edges,[1,2,3,4])]
    end
    mesh = XCALibre.UNV2.update_mesh_format(generate!(MeshBuilder2D(points,edges,patches,blocks)),Int64,Float64)
    model = Physics(time=Steady(),fluid=Fluid{Incompressible}(nu=nu),turbulence=RANS{Laminar}(),energy=Energy{Isothermal}(),domain=mesh)
    pair = split ? construct_planar_transport(mesh,:interface_a,:interface_b) : ()
    pressure = split ? construct_planar_pressure(pair) : ()
    profile(x,t,i) = SVector(6*x[2]*(1-x[2]),0.,0.)
    boundaries = assign((U=[pair...,DirichletFunction(:inlet,profile),Zerogradient(:outlet),Dirichlet(:walls,[0.,0.,0.])],
        p=[pressure...,Zerogradient(:inlet),Dirichlet(:outlet,0.),Zerogradient(:walls)]);region=mesh)
    setup(relax) = SolverSetup(solver=Bicgstab(),preconditioner=Jacobi(),relax=relax,convergence=convergence,rtol=1e-11,atol=1e-13)
    config = Configuration(solvers=(U=setup(relaxU),p=setup(relaxp)),schemes=(U=Schemes(divergence=Upwind),p=Schemes()),boundaries=boundaries,
        runtime=Runtime(iterations=iterations,write_interval=-1,time_step=1),hardware=Hardware(backend=CPU(),workgroup=64))
    initialise!(model.momentum.U,[initialU,0.,0.]); initialise!(model.momentum.p,initialp)
    model,config
end

# Cross-section average at the first/last cell-centre columns. These are defined
# interior sampling planes, not extrema or extrapolated boundary pressures.
function channel_metrics(model,config,facevalues; transport=nothing)
    mesh = model.domain
    U,p = model.momentum.U,model.momentum.p
    exact = [6*x[2]*(1-x[2]) for x in mesh.cell_centre]
    V = mesh.cell_volume
    velocity_error = sqrt(sum(V .* ((U.x.values .- exact).^2 .+ U.y.values.^2)) / sum(V .* exact.^2))
    xs = first.(mesh.cell_centre)
    xleft,xright = extrema(xs)
    function section(x)
        ids = findall(y -> isapprox(y,x;atol=100eps(Float64),rtol=100eps(Float64)),xs)
        sum(V[ids] .* p.values[ids])/sum(V[ids])
    end
    pressure_drop = section(xleft)-section(xright)
    exact_drop = 12model.fluid.nu.values*(xright-xleft)
    pressure_error = abs(pressure_drop-exact_drop)/abs(exact_drop)
    div = ScalarField(mesh)
    XCALibre.Calculate.div!(div,FaceScalarField(facevalues,mesh),config)
    continuity = maximum(abs,div.values .* V)
    exterior = reduce(vcat,[collect(b.IDs_range) for b in mesh.boundaries if b.name ∉ (:interface_a,:interface_b)])
    boundary_balance = sum(facevalues[exterior])
    if transport === nothing
        interface_balance,interface_flow,unequal_distances = nothing,nothing,false
    else
        c = transport.coupling
        qa = sum(facevalues[mesh.boundaries[c.patch_a].IDs_range])
        qb = sum(facevalues[mesh.boundaries[c.patch_b].IDs_range])
        interface_balance,interface_flow = qa+qb,qa
        pair = only(filter(b -> b isa PlanarTransport && b.value.side_a,config.boundaries.U))
        unequal_distances = any(pair.value.distance_a .!= pair.value.distance_b)
    end
    (; velocity_error,pressure_error,pressure_drop,exact_drop,sections=(xleft,xright),continuity,
        boundary_balance,interface_balance,interface_flow,unequal_distances)
end
