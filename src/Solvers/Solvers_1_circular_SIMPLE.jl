export CircularSimpleState, circular_simple!

"""Restricted array-based static CPU circular SIMPLE. Owns copied geometry/data,
cell point values and individual conservative overlap fluxes. No model/run!,
output, motion, sources or transient integration. Arrays in geometry/config are
read-only by contract; construct a new state after geometry/config changes.
Only mutate velocity/pressure between calls: this reinitialises continuation.
Boundary pairs: prescribed velocity/homogeneous or prescribed-gradient pressure, or homogeneous
velocity Neumann/fixed pressure. Backflow is rejected. One velocity anchor per
component and one reference per pressure-unanchored component are required.
Tolerances are absolute integrated momentum/flow and point-field-change units.
`predictor=:low_order` preserves the original path; opt-in `:reconstructed`
reconstructs H0-rD*GaussGradient(p_old), retaining low-order interpolation of
rD*GaussGradient(p_old). Config changes require a new state.
"""
mutable struct CircularSimpleState{M,V,P,C,F}
    geometry::M
    velocity::V
    pressure::P
    config::C
    faces::F
    overlaps::F
    saved_velocity::V
    saved_pressure::P
    iteration::Int
    valid::Bool
end

function CircularSimpleState(m::CircularMomentumGeometry,U::AbstractVector,p::AbstractVector,
    velocity_values::AbstractVector,pressure_values::AbstractVector,viscosity::Real;
    reference=Dict{Int,Float64}(),relaxation_U=0.7,relaxation_p=0.3,
    momentum_tolerance=1e-9,continuity_tolerance=1e-10,change_tolerance=1e-9,
    pressure_tolerance=1e-11,pressure_sweeps=200,predictor=:low_order)
    IC=InterfaceCoupling
    IC._momentum_values(m,U,velocity_values)
    IC._pressure_data(m.pressure,p,pressure_values)
    isfinite(viscosity) && viscosity>0 || throw(ArgumentError("positive constant viscosity required"))
    all(x -> isfinite(x) && 0<x<=1,(relaxation_U,relaxation_p)) || throw(ArgumentError("relaxations must lie in (0,1]"))
    all(x -> isfinite(x) && x>0,(momentum_tolerance,continuity_tolerance,change_tolerance,pressure_tolerance)) || throw(ArgumentError("positive finite tolerances required"))
    pressure_sweeps isa Integer && pressure_sweeps>0 || throw(ArgumentError("positive pressure sweep count required"))
    predictor in (:low_order,:reconstructed) || throw(ArgumentError("unsupported circular predictor"))
    for k in eachindex(m.velocity.boundary_kind)
        u,q=m.velocity.boundary_kind[k],m.pressure.boundary_kind[k]
        (u==:dirichlet && q in (:neumann,:normal_gradient)) || (u==:neumann && q==:dirichlet) ||
            throw(ArgumentError("unsupported coupled boundary pair"))
    end
    h=m.velocity; anchored=falses(length(U))
    for (k,j) in enumerate(h.exterior_faces)
        h.boundary_kind[k]==:dirichlet && (anchored[h.reconstruction.metrics.owner_a[j]]=true)
    end
    all(c -> any(anchored[c]),h.components) || throw(ArgumentError("velocity anchor required per component"))
    pred=circular_momentum_predictor_flux(m,U,velocity_values)
    balance=IC._pressure_balance(m.pressure,pred.faces,pred.overlaps)
    metrics=m.pressure.reconstruction.metrics
    scale=zeros(eltype(m.pressure.beta),length(U))
    for j in eachindex(pred.faces)
        metrics.interface_face[j] && continue
        a,b=metrics.owner_a[j],metrics.owner_b[j]
        scale[a]+=abs(pred.faces[j]);a==b || (scale[b]+=abs(pred.faces[j]))
    end
    for k in eachindex(pred.overlaps)
        scale[metrics.coupling.cell_a[k]]+=abs(pred.overlaps[k])
        scale[metrics.coupling.cell_b[k]]+=abs(pred.overlaps[k])
    end
    IC._pressure_gauges(m.pressure,balance,reference,scale,1000eps(eltype(m.pressure.beta)))
    T=eltype(m.pressure.beta)
    cfg=(;velocity_values=SVector{3,T}.(velocity_values),pressure_values=T.(pressure_values),viscosity=T(viscosity),
        reference=Dict(Int(i)=>T(v) for (i,v) in pairs(reference)),relaxation_U,relaxation_p,
        momentum_tolerance,continuity_tolerance,change_tolerance,pressure_tolerance,pressure_sweeps,predictor)
    all(x -> all(isfinite,x),cfg.velocity_values) && all(isfinite,cfg.pressure_values) &&
        isfinite(cfg.viscosity) && cfg.viscosity>0 && all(isfinite,values(cfg.reference)) || throw(ArgumentError("invalid configuration conversion"))
    u=SVector{3,T}.(U);q=T.(p)
    all(x -> all(isfinite,x),u) && all(isfinite,q) || throw(ArgumentError("nonfinite field conversion"))
    CircularSimpleState(deepcopy(m),u,q,cfg,copy(pred.faces),copy(pred.overlaps),copy(u),copy(q),0,false)
end

"""
    circular_simple!(state; iterations=100)

Advance restricted static SIMPLE. Pressure projection must converge internally;
otherwise throw and invalidate continuation without publishing the failed step.
Unrelaxed pressure corrects flux, relaxed pressure corrects cell velocity using
H0-rD*GaussGradient(p). Refreshed momentum is checked using UPDATED velocity,
pressure and corrected fluxes. Require momentum, pressure refreshed residual,
independent continuity and field/flux fixed-point changes. Iteration exhaustion
returns converged=false. Unchanged fields retain overlap continuation; external
field changes or failed calls reinitialise from current fields. No background work.
"""
function circular_simple!(s::CircularSimpleState;iterations=100)
    IC=InterfaceCoupling
    fresh=s.valid && s.velocity==s.saved_velocity && s.pressure==s.saved_pressure
    s.valid=false
    iterations isa Integer && iterations>0 || throw(ArgumentError("positive iteration count required"))
    m,cfg=s.geometry,s.config
    IC._momentum_values(m,s.velocity,cfg.velocity_values)
    IC._pressure_data(m.pressure,s.pressure,cfg.pressure_values)
    if !fresh
        pred=circular_momentum_predictor_flux(m,s.velocity,cfg.velocity_values)
        s.faces=pred.faces;s.overlaps=pred.overlaps;s.iteration=0
    end
    history=NamedTuple[];converged=false
    for _ in 1:iterations
        U,p=s.velocity,s.pressure
        terms=circular_momentum_terms(m,U,cfg.velocity_values,p,cfg.pressure_values,s.faces,s.overlaps,cfg.viscosity)
        step=circular_momentum_solve(terms,U;relaxation=cfg.relaxation_U)
        if cfg.predictor==:reconstructed
            # Reuse the exact saved integrated force: -force/D = rD*G(p_old).
            pressure_velocity=-terms.pressure_force./step.physical_diagonal
            pred=circular_momentum_reconstructed_predictor_flux(m,step.H0,pressure_velocity,cfg.velocity_values)
        else
            pred=circular_momentum_predictor_flux(m,step.H0,cfg.velocity_values)
        end
        # Add the opposite of the fixed pressure derivative flux so recovery
        # preserves prescribed boundary flow, including impermeable walls.
        for (k,j) in enumerate(m.pressure.exterior_faces)
            m.pressure.boundary_kind[k]==:normal_gradient || continue
            a=m.pressure.reconstruction.metrics.owner_a[j]
            pred.faces[j]+=step.mobility[a]*norm(m.pressure.reconstruction.metrics.face_area_vector[j])*cfg.pressure_values[k]
        end
        projection=circular_pressure_project(m.pressure,step.mobility,cfg.pressure_values,pred.faces,pred.overlaps;
            initial=p,reference=cfg.reference,max_sweeps=cfg.pressure_sweeps,tolerance=cfg.pressure_tolerance)
        projection.converged || throw(ErrorException("circular SIMPLE pressure projection did not converge"))
        relaxed=p+cfg.relaxation_p*(projection.pressure-p)
        # Gauge-shift each unanchored component uniformly, not just the reference cell.
        for cells in m.pressure.components
            ids=[i for i in cells if haskey(cfg.reference,i)]
            if !isempty(ids)
                i=only(ids);relaxed[cells].+=cfg.reference[i]-relaxed[i]
            end
        end
        gradient=fill(zero(eltype(U)),length(U))
        circular_pressure_gradient!(gradient,m.pressure,relaxed,cfg.pressure_values)
        corrected=[step.H0[i]-step.mobility[i]*gradient[i] for i in eachindex(U)]
        IC._momentum_values(m,corrected,cfg.velocity_values)
        refreshed=circular_momentum_terms(m,corrected,cfg.velocity_values,relaxed,cfg.pressure_values,
            projection.faces,projection.overlaps,cfg.viscosity)
        momentum=maximum(norm,circular_momentum_flux(refreshed,corrected).residual)
        continuity=maximum(abs,IC._pressure_balance(m.pressure,projection.faces,projection.overlaps))
        velocity_change=maximum(norm,corrected-U);pressure_change=maximum(abs,relaxed-p)
        flux_change=max(maximum(abs,projection.faces-s.faces),maximum(abs,projection.overlaps-s.overlaps))
        pr=projection.history[end].refreshed
        all(isfinite,(momentum,continuity,velocity_change,pressure_change,flux_change,pr)) || throw(ArgumentError("nonfinite circular SIMPLE diagnostics"))
        # Publish only after validating the entire outer step.
        copyto!(s.velocity,corrected);copyto!(s.pressure,relaxed)
        s.faces=projection.faces;s.overlaps=projection.overlaps;s.iteration+=1
        push!(history,(;momentum,continuity,pressure_residual=pr,velocity_change,pressure_change,flux_change,
            pressure_sweeps=projection.iterations))
        converged=momentum<=cfg.momentum_tolerance && continuity<=cfg.continuity_tolerance &&
            pr<=cfg.pressure_tolerance && max(velocity_change,pressure_change,flux_change)<=cfg.change_tolerance
        converged && break
    end
    copyto!(s.saved_velocity,s.velocity);copyto!(s.saved_pressure,s.pressure);s.valid=true
    (;history,iterations=length(history),total_iterations=s.iteration,converged)
end
