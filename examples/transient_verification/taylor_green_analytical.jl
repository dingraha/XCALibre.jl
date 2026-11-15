module TaylorGreenAnalytical

using XCALibre
using LinearAlgebra

"""Point-sampled, unadvected Taylor–Green solution; p is kinematic pressure."""
function exact(x, y, t; L=1.0, U0=1.0, nu=0.01)
    k = 2pi/L
    a = U0*exp(-2nu*k^2*t)
    (; ux=a*sin(k*x)*cos(k*y), uy=-a*cos(k*x)*sin(k*y),
        p=a^2/4*(cos(2k*x)+cos(2k*y)))
end

"""One common volume-weighted pressure-error gauge, not a cell-reference gauge."""
function errors(ux, uy, p, target, volumes)
    weight = volumes/sum(volumes)
    du2 = (ux .- getproperty.(target, :ux)).^2 + (uy .- getproperty.(target, :uy)).^2
    u2 = getproperty.(target, :ux).^2 + getproperty.(target, :uy).^2
    dp = p .- getproperty.(target, :p)
    gauge = sum(weight .* dp)
    dp .-= gauge
    pexact = getproperty.(target, :p)
    pexact = pexact .- sum(weight .* pexact)
    u_rms = sqrt(sum(weight .* du2))
    p_rms = sqrt(sum(weight .* dp.^2))
    (; u_rms, u_relative=u_rms/sqrt(sum(weight .* u2)), u_max=sqrt(maximum(du2)),
        p_rms, p_relative=p_rms/sqrt(sum(weight .* pexact.^2)), p_max=maximum(abs, dp), gauge)
end

"""Gather ORIGINAL cell balances from retained flux; periodic faces own separate cells."""
function flux_checks(mesh, flux, periodic)
    balance = zeros(length(mesh.cells))
    throughput = zeros(length(mesh.cells))
    for (f, face) in enumerate(mesh.faces)
        a, b = face.ownerCells
        balance[a] += flux[f]
        throughput[a] += abs(flux[f])
        if a != b
            balance[b] -= flux[f]
            throughput[b] += abs(flux[f])
        end
    end
    pair_max = maximum(abs(flux[f] + flux[g]) for bc in periodic
        if bc isa PeriodicParent for (f, g) in zip(bc.IDs_range, bc.value.face_map))
    volumes = [c.volume for c in mesh.cells]
    (; balance, continuity_max=maximum(abs, balance), divergence_max=maximum(abs, balance./volumes),
        continuity_scaled=maximum(abs, balance)/maximum(throughput), pair_max,
        global_balance=sum(balance), flux_max=maximum(abs, flux))
end

"""Ordinary static conforming periodic mesh: NO sliding interface or ALE.

Euler, Linear convection, componentwise viscosity by default. Solver diagnostics
are not independently refreshed transient momentum certificates. No repeated run!
stepping: each case is one run preserving its actual internal time histories.
"""
function run_case(; n=16, dt=0.001, tend=0.2, L=1.0, U0=1.0, nu=0.01,
    inner_loops=2, time_scheme=Euler, convection=Linear)
    n isa Integer && n >= 4 || throw(ArgumentError("n must be an integer >= 4"))
    all(x -> isfinite(x) && x > 0, (dt, tend, L, U0, nu)) || throw(ArgumentError("positive finite parameters required"))
    inner_loops isa Integer && inner_loops > 0 || throw(ArgumentError("positive corrector count required"))
    steps = round(Int, tend/dt)
    steps > 0 && isapprox(steps*dt, tend; rtol=1e-12, atol=0) ||
        throw(ArgumentError("dt must divide tend"))
    Base.cumulative_compile_timing(true)
    compile0 = Base.cumulative_compile_time_ns()[1]
    setup_seconds = @elapsed begin
        points = [Point(0.,0.,0.), Point(L,0.,0.), Point(0.,L,0.), Point(L,L,0.)]
        edges = [line!(points,1,2,n), line!(points,3,4,n), line!(points,1,3,n), line!(points,2,4,n)]
        patches = [Patch(:left,[3]), Patch(:right,[4]), Patch(:bottom,[1]), Patch(:top,[2])]
        mesh = XCALibre.UNV2.update_mesh_format(
            generate!(MeshBuilder2D(points,edges,patches,[quad(edges,[1,2,3,4])])), Int64, Float64)
        px = construct_periodic(mesh, CPU(), :left, :right)
        py = construct_periodic(mesh, CPU(), :bottom, :top)
        periodic = (px..., py...)
        model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=nu),
            turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=mesh)
        boundaries = assign(region=mesh, (U=[periodic...], p=[periodic...]))
        solver() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
            convergence=1e-12, relax=1.0, rtol=1e-12, atol=1e-15)
        config = Configuration(solvers=(U=solver(), p=solver()),
            schemes=(U=Schemes(time=time_scheme, divergence=convection, gradient=Gauss),
                p=Schemes(time=time_scheme, gradient=Gauss)),
            runtime=Runtime(iterations=steps, write_interval=-1, time_step=dt),
            hardware=Hardware(backend=CPU(), workgroup=1024), boundaries=boundaries)
        U, p = model.momentum.U, model.momentum.p
        initial = [exact(c.centre[1], c.centre[2], 0.; L, U0, nu) for c in mesh.cells]
        U.x.values .= getproperty.(initial, :ux)
        U.y.values .= getproperty.(initial, :uy)
        U.z.values .= 0.0
        p.values .= getproperty.(initial, :p)
    end
    setup_compile_seconds = (Base.cumulative_compile_time_ns()[1]-compile0)/1e9
    compile1 = Base.cumulative_compile_time_ns()[1]
    solve_seconds = @elapsed residuals = run!(model, config; progress=false, pref=0.0,
        inner_loops=inner_loops, transpose_stress=false)
    solve_compile_seconds = (Base.cumulative_compile_time_ns()[1]-compile1)/1e9
    ux, uy, uz, pressure = copy(U.x.values), copy(U.y.values), copy(U.z.values), copy(p.values)
    flux = copy(model.momentum.mdotf.values)
    target = [exact(c.centre[1],c.centre[2],steps*dt; L, U0, nu) for c in mesh.cells]
    volumes = [c.volume for c in mesh.cells]
    error = errors(ux, uy, pressure, target, volumes)
    conservation = flux_checks(mesh, flux, boundaries.U)
    finite = all(isfinite, ux) && all(isfinite, uy) && all(isfinite, uz) &&
        all(isfinite, pressure) && all(isfinite, flux) &&
        all(a -> all(isfinite, a), values(residuals))
    (; n, dt, tend=steps*dt, steps, inner_loops, time_scheme, convection, ux, uy, uz,
        pressure, flux, volumes, error, conservation, finite,
        setup_seconds, solve_seconds, setup_compile_seconds, solve_compile_seconds)
end

function print_result(r)
    e, c = r.error, r.conservation
    println("n=$(r.n) dt=$(r.dt) correctors=$(r.inner_loops) Urel=$(e.u_relative) ",
        "pRMS=$(e.p_rms) pmax=$(e.p_max) divmax=$(c.divergence_max) ",
        "pairmax=$(c.pair_max) setup=$(r.setup_seconds)s solve=$(r.solve_seconds)s ",
        "compile=$(r.setup_compile_seconds+r.solve_compile_seconds)s")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    # Small demonstration; larger studies are deliberately outside routine tests.
    results = TaylorGreenAnalytical.run_case()
    TaylorGreenAnalytical.print_result(results)
end
