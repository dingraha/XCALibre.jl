export PlanarSimpleState, planar_simple!

"""
    PlanarSimpleState(model, config; pref=nothing, continuity=1e-9)

Reusable state for restricted static CPU Mesh2 SIMPLE: steady unit-density
incompressible laminar/isothermal physics, Upwind convection, Linear diffusion,
Gauss gradients without limiters, constant positive viscosity, one PlanarTransport
pair in boundaries.U and a shared PlanarPressure pair in boundaries.p.
Exterior boundaries support Dirichlet/DirichletFunction velocity with homogeneous
pressure Zerogradient, or Zerogradient/Outlet velocity with fixed Dirichlet
pressure. Zero-gradient velocity/pressure pairs are also allowed; these are not
impermeable walls. Use prescribed zero velocity for no-slip walls. No output, restart, SIMPLEC, nonorthogonal or transpose-stress
correction is implemented. Rebuild state after mesh, boundary or configuration
changes. Do not directly mutate state flux/coefficient storage.

`continuity` bounds max(abs(cell volume * corrected divergence)), an integrated
volume-flow imbalance. With all-Neumann pressure, supply pref and a globally
balanced predictor. Changing U or p between calls reinitialises fluxes and iteration
count; otherwise subsequent calls retain individual conservative overlap fluxes.
"""
struct PlanarSimpleState{M,C,W}
    model::M
    config::C
    work::W
end

function _validate_planar_simple(model,config,pref,continuity)
    mesh = model.domain
    mesh isa Mesh2 && _get_backend(mesh) isa CPU && config.hardware.backend isa CPU ||
        throw(ArgumentError("planar SIMPLE requires CPU Mesh2"))
    model.time isa Steady && model.fluid isa Incompressible &&
        model.turbulence isa Laminar && model.energy === nothing ||
        throw(ArgumentError("planar SIMPLE requires steady incompressible laminar isothermal physics"))
    model.fluid.nu isa ConstantScalar && isfinite(model.fluid.nu.values) && model.fluid.nu.values > 0 &&
        model.fluid.rho isa ConstantScalar && model.fluid.rho.values == 1 ||
        throw(ArgumentError("planar SIMPLE requires constant positive viscosity and unit density"))
    model.momentum.sources === nothing || throw(ArgumentError("additional momentum sources are unsupported"))
    all(iszero,model.momentum.U.z.values) || throw(ArgumentError("Mesh2 planar SIMPLE does not solve z velocity"))
    all(a -> all(isfinite,a.values),(model.momentum.U.x,model.momentum.U.y,model.momentum.p)) ||
        throw(ArgumentError("initial velocity and pressure must be finite"))
    for s in (config.schemes.U,config.schemes.p)
        s.time === SteadyState && s.laplacian === Linear && s.gradient === Gauss && s.limiter === nothing ||
            throw(ArgumentError("planar SIMPLE requires steady Linear Laplacians and unlimited Gauss gradients"))
    end
    config.schemes.U.divergence === Upwind || throw(ArgumentError("planar SIMPLE requires Upwind momentum"))
    for setup in (config.solvers.U,config.solvers.p)
        isfinite(setup.relax) && 0 < setup.relax <= 1 ||
            throw(ArgumentError("planar SIMPLE relaxation factors must lie in (0,1]"))
    end
    config.runtime.write_interval < 0 && (config.postprocess === nothing || isempty(config.postprocess)) ||
        throw(ArgumentError("planar SIMPLE output/postprocessing is not supported; use write_interval=-1"))
    isfinite(continuity) && continuity > 0 || throw(ArgumentError("continuity tolerance must be positive and finite"))
    pref === nothing || isfinite(pref) || throw(ArgumentError("pressure reference must be finite"))
    UBCs,pBCs = config.boundaries.U,config.boundaries.p
    pair = Tuple(filter(bc -> bc isa PlanarTransport,UBCs))
    length(pair) == 2 || throw(ArgumentError("planar SIMPLE requires exactly one momentum interface pair"))
    pressure = Tuple(filter(bc -> bc isa PlanarPressure,pBCs))
    length(pressure) == 2 && all(bc -> bc.value.coupling === pair[1].value.coupling,pressure) ||
        throw(ArgumentError("pressure and momentum must share one interface coupling"))
    construct_planar_pressure(pair) # validate pair geometry
    for bc in pressure
        Discretise.validate_boundary_pair(bc,pBCs)
        bc.value.distance_a === pair[1].value.distance_a && bc.value.distance_b === pair[1].value.distance_b ||
            throw(ArgumentError("pressure and momentum must share normal distances"))
    end
    length(UBCs) == length(mesh.boundaries) && length(pBCs) == length(mesh.boundaries) ||
        throw(ArgumentError("each field must specify every mesh patch exactly once"))
    sort([bc.ID for bc in UBCs]) == collect(eachindex(mesh.boundaries)) &&
        sort([bc.ID for bc in pBCs]) == collect(eachindex(mesh.boundaries)) ||
        throw(ArgumentError("boundaries must be assigned to this mesh"))
    fixed_pressure = false
    for ubc in UBCs
        pbc = only(filter(bc -> bc.ID == ubc.ID,pBCs))
        if ubc isa Dirichlet
            length(ubc.value) == 3 && all(isfinite,ubc.value) && iszero(ubc.value[3]) ||
                throw(ArgumentError("prescribed velocity must have finite x/y and zero z"))
        elseif ubc isa DirichletFunction
            # Validate initial values; dynamic callbacks are checked again during prediction.
            for (i,fID) in enumerate(ubc.IDs_range)
                v = ubc.value(mesh.face_centre[fID],0,i)
                length(v) == 3 && all(isfinite,v) && iszero(v[3]) ||
                    throw(ArgumentError("prescribed velocity function must return finite x/y and zero z"))
            end
        end
        pbc isa Dirichlet && !isfinite(pbc.value) && throw(ArgumentError("fixed pressure must be finite"))
        if ubc isa PlanarTransport
            pbc isa PlanarPressure && pbc.value.side_a == ubc.value.side_a ||
                throw(ArgumentError("interface patch sides must match"))
        elseif ubc isa Union{Dirichlet,DirichletFunction}
            pbc isa Zerogradient || throw(ArgumentError("prescribed velocity requires homogeneous pressure Zerogradient"))
        elseif ubc isa Union{Zerogradient,Outlet}
            pbc isa Union{Zerogradient,Dirichlet} || throw(ArgumentError("velocity outlet requires fixed or homogeneous Neumann pressure"))
            fixed_pressure |= pbc isa Dirichlet
        else
            throw(ArgumentError("unsupported exterior velocity boundary $(typeof(ubc))"))
        end
    end
    fixed_pressure && pref !== nothing && throw(ArgumentError("do not combine fixed pressure boundaries with pref"))
    !fixed_pressure && pref === nothing && throw(ArgumentError("all-Neumann pressure requires pref"))
    pair,pressure,fixed_pressure
end

function PlanarSimpleState(model,config; pref=nothing,continuity=1e-9)
    pair,pressure,fixed = _validate_planar_simple(model,config,pref,continuity)
    mesh = model.domain
    U,p = model.momentum.U,model.momentum.p
    transport = InterfaceTransportFlux(mesh,pair[1].value.coupling)
    coefficient = InterfacePressureCoefficient(mesh,pressure[1])
    gradp = Grad{Gauss}(p)
    divergence = ScalarField(mesh)
    Ueqn = (Time{SteadyState}(U)+Divergence{Upwind}(transport,U)-Laplacian{Linear}(model.fluid.nu,U)
        == -Source(gradp.result)) → VectorEquation(U,config.boundaries.U)
    peqn = (-Laplacian{Linear}(coefficient,p) == -Source(divergence)) → ScalarEquation(p,config.boundaries.p)
    @reset Ueqn.preconditioner = set_preconditioner(config.solvers.U,Ueqn)
    @reset Ueqn.solver = _workspace(config.solvers.U,Ueqn,XDir())
    @reset peqn.preconditioner = set_preconditioner(config.solvers.p,peqn)
    @reset peqn.solver = _workspace(config.solvers.p,peqn)
    work = (; pair,pressure,transport,coefficient,gradp,divergence,Ueqn,peqn,
        Hv=VectorField(mesh),rD=ScalarField(mesh),Uf=FaceVectorField(mesh),pf=FaceScalarField(mesh),
        faceflux=FaceScalarField(transport.values,mesh),predictor=similar(transport.overlaps),
        previous=similar(p.values),savedU=VectorField(mesh),savedp=ScalarField(mesh),
        oldU=VectorField(mesh),physical_diagonal=similar(p.values),relaxed_diagonal=similar(p.values),
        iteration=Ref(0),valid=Ref(false),pref,continuity,fixed)
    PlanarSimpleState(model,config,work)
end

# Restricted component solve: retain the actual pre-relaxation diagonal and
# source velocity, including boundary contributions and diagonal-dominance changes.
function _planar_solve_momentum!(w,U,cfg,time)
    eqn = w.Ueqn
    for (a,b) in zip((w.oldU.x,w.oldU.y,w.oldU.z),(U.x,U.y,U.z))
        copyto!(a.values,b.values)
    end
    discretise!(eqn,U,cfg;time)
    nz = _nzval(eqn.equation.A)
    ids = eqn.equation.diag_nz
    residuals = eltype(nz)[]
    for (j,(dir,field)) in enumerate(((XDir(),U.x),(YDir(),U.y)))
        update_equation!(eqn,cfg)
        apply_boundary_conditions!(eqn,cfg.boundaries.U,dir,time,cfg)
        if j == 1
            copyto!(w.physical_diagonal,nz[ids])
            all(x -> isfinite(x) && x > 0,w.physical_diagonal) ||
                throw(ArgumentError("physical momentum diagonals must be positive and finite"))
        else
            nz[ids] == w.physical_diagonal ||
                throw(ArgumentError("planar SIMPLE requires identical x/y physical momentum diagonals"))
        end
        implicit_relaxation_diagdom!(eqn,field.values,cfg.solvers.U.relax,dir,cfg)
        if j == 1
            copyto!(w.relaxed_diagonal,nz[ids])
        else
            nz[ids] == w.relaxed_diagonal ||
                throw(ArgumentError("planar SIMPLE requires identical x/y relaxed momentum diagonals"))
        end
        update_preconditioner!(eqn.preconditioner,U.mesh,cfg)
        push!(residuals,solve_system!(eqn,cfg.solvers.U,field,dir,cfg))
    end
    residuals[1],residuals[2]
end

# H! returns H/relaxed_D after pressure-source removal. Undo the complete
# diagonal augmentation and its old-velocity source, not merely division by alpha.
function _planar_physical_predictor!(w)
    D,Dt = w.physical_diagonal,w.relaxed_diagonal
    for (h,u) in zip((w.Hv.x,w.Hv.y),(w.oldU.x,w.oldU.y))
        @. h.values = (Dt*h.values - (Dt-D)*u.values)/D
    end
    fill!(w.Hv.z.values,0)
    @. w.rD.values = w.rD.mesh.cell_volume/D
    all(a -> all(isfinite,a.values),(w.Hv.x,w.Hv.y,w.rD)) ||
        throw(ArgumentError("nonfinite physical momentum predictor"))
    nothing
end

_planar_state_fresh(w,U,p) = w.valid[] && all(a.values == b.values for (a,b) in
    zip((U.x,U.y,U.z),(w.savedU.x,w.savedU.y,w.savedU.z))) && p.values == w.savedp.values

"""
    planar_simple!(state::PlanarSimpleState; iterations=state.config.runtime.iterations)

Advance a restricted static SIMPLE state. Return residual histories Ux, Uy, p,
continuity, p_absolute, completed iteration count, and converged. The pressure
stopping test accepts either its normalised residual or an absolute initial
pressure-equation L1 residual below the integrated-flow continuity tolerance;
this avoids division by a tiny pressure scale near zero-pressure solutions. Histories include only
executed iterations. Pressure flux correction uses unrelaxed pressure; relaxed
pressure is used for cell velocity correction. Corrected overlap fluxes are kept
for the next momentum solve and subsequent calls. The cell predictor and pressure
mobility use the physical pre-relaxation momentum diagonal: the full diagonal
augmentation and its old-velocity source are removed after the relaxed momentum
solve. Identical positive x/y physical diagonals are required. Ordinary SIMPLE
is unchanged. No background execution.
"""
function planar_simple!(state::PlanarSimpleState; iterations=state.config.runtime.iterations)
    model,cfg,w = state.model,state.config,state.work
    U,p = model.momentum.U,model.momentum.p
    fresh = _planar_state_fresh(w,U,p)
    w.valid[] = false # every failed call, including validation, invalidates continuation
    iterations isa Integer && iterations > 0 || throw(ArgumentError("iterations must be positive"))
    _validate_planar_simple(model,cfg,w.pref,w.continuity)
    if !fresh
        w.iteration[] = 0
        interpolate!(w.Uf,U,cfg)
        correct_boundaries!(w.Uf,U,cfg.boundaries.U,0.0,cfg)
        flux!(w.faceflux,w.Uf,cfg)
        interface_predictor_flux!(w.predictor,w.pair[1],U)
        update_interface_flux!(w.transport,w.predictor)
    end
    grad!(w.gradp,w.pf,p,cfg.boundaries.p,w.iteration[],cfg)
    T = eltype(p.values)
    rx,ry,rp,rc,ra = T[],T[],T[],T[],T[]
    converged = false
    for _ in 1:iterations
        time = w.iteration[]+1
        ux,uy = _planar_solve_momentum!(w,U,cfg,time)
        remove_pressure_source!(w.Ueqn,w.gradp,cfg)
        H!(w.Hv,U,w.Ueqn,cfg)
        _planar_physical_predictor!(w)
        interpolate!(w.Uf,w.Hv,cfg)
        correct_boundaries!(w.Uf,w.Hv,cfg.boundaries.U,time,cfg)
        all(iszero,w.Uf.z.values) || throw(ArgumentError("Mesh2 planar SIMPLE requires zero prescribed z velocity"))
        flux!(w.faceflux,w.Uf,cfg)
        interface_predictor_flux!(w.predictor,w.pair[1],w.Hv)
        update_interface_flux!(w.transport,w.predictor)
        all(isfinite,w.transport.values) || throw(ArgumentError("nonfinite predicted face flux"))
        div!(w.divergence,w.faceflux,cfg)
        if !w.fixed
            balance = sum(w.divergence.values .* model.domain.cell_volume)
            scale = sum(abs,w.transport.values)
            abs(balance) <= 1000eps(T)*max(scale,one(T)) ||
                throw(ArgumentError("all-Neumann pressure requires globally balanced predicted boundary flux"))
        end
        update_interface_coefficient!(w.coefficient,w.rD)
        copyto!(w.previous,p.values)
        pr = solve_equation!(w.peqn,p,cfg.boundaries.p,cfg.solvers.p,cfg;time,ref=w.pref)
        # Evaluate the initial absolute residual on the same assembled system.
        mul!(w.peqn.equation.R,w.peqn.equation.A,w.previous)
        pabsolute = sum(abs,w.peqn.equation.R .- w.peqn.equation.b)
        correct_mass_flux!(w.faceflux,w.peqn,cfg;previous=w.previous,time)
        correct_interface_pressure_flux!(w.transport,w.predictor,w.pressure[1],p,w.coefficient)
        div!(w.divergence,w.faceflux,cfg)
        imbalance = maximum(abs,w.divergence.values .* model.domain.cell_volume)
        explicit_relaxation!(p,w.previous,cfg.solvers.p.relax,cfg)
        grad!(w.gradp,w.pf,p,cfg.boundaries.p,time,cfg)
        correct_velocity!(U,w.Hv,w.gradp,w.rD,cfg)
        all(isfinite,p.values) && all(a -> all(isfinite,a.values),(U.x,U.y,U.z)) && isfinite(imbalance) ||
            throw(ArgumentError("nonfinite planar SIMPLE state"))
        push!(rx,ux); push!(ry,uy); push!(rp,pr); push!(rc,imbalance); push!(ra,pabsolute)
        w.iteration[] = time
        converged = ux <= cfg.solvers.U.convergence && uy <= cfg.solvers.U.convergence &&
            (pr <= cfg.solvers.p.convergence || pabsolute <= w.continuity) && imbalance <= w.continuity
        converged && break
    end
    for (a,b) in zip((w.savedU.x,w.savedU.y,w.savedU.z),(U.x,U.y,U.z))
        copyto!(a.values,b.values)
    end
    copyto!(w.savedp.values,p.values)
    # Retain physical face diagnostics, but ordinary run! cannot continue overlap state.
    copyto!(model.momentum.mdotf.values,w.transport.values)
    # Invalidate ordinary continuation snapshots: they cannot represent overlaps.
    fill!(model.momentum.pmdot.values,NaN)
    w.valid[] = true
    (; Ux=rx,Uy=ry,p=rp,p_absolute=ra,continuity=rc,iterations=length(rx),converged)
end
