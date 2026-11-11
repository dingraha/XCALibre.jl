# Larger verification study; separate from the fast unit checks.
include("channel_setup.jl")

function refinement_case(; split=true, kwargs...)
    model,config = channel_case(;split,kwargs...)
    if split
        state = PlanarSimpleState(model,config;continuity=1e-9)
        residuals = planar_simple!(state)
        residuals.converged || error("nonconforming refinement case did not converge")
        metrics = channel_metrics(model,config,state.work.transport.values;transport=state.work.transport)
    else
        residuals = run!(model,config;progress=false,transpose_stress=false)
        metrics = channel_metrics(model,config,model.momentum.mdotf.values)
        # Ordinary run! histories are padded after convergence.
        last = findlast(!iszero,residuals.Ux)
        last !== nothing && residuals.Ux[last] <= config.solvers.U.convergence &&
            residuals.Uy[last] <= config.solvers.U.convergence && residuals.p[last] <= config.solvers.p.convergence ||
            error("conforming refinement case did not converge")
    end
    (; metrics, cells=length(model.domain.cells))
end

levels = [1,2,4]
nonconforming = [refinement_case(na=4k,nb=5k,nx=3k,iterations=2500) for k in levels]
conforming = [refinement_case(split=false,nb=5k,nx=3k,iterations=2500) for k in levels]
strong_mismatch = refinement_case(na=6,nb=19,nx=6,interface_x=0.65,iterations=2500)
function orders(cases,key)
    e = [getproperty(c.metrics,key) for c in cases]
    log2.(e[1:end-1] ./ e[2:end])
end
results = (; levels,nonconforming,conforming,strong_mismatch,
    velocity_orders=orders(nonconforming,:velocity_error),pressure_orders=orders(nonconforming,:pressure_error))
println("\nChannel refinement: relative volume-weighted velocity and interior-section pressure-drop errors")
for (k,a,b) in zip(levels,nonconforming,conforming)
    println("level=",k," nonconforming=",(a.metrics.velocity_error,a.metrics.pressure_error),
        " conforming=",(b.metrics.velocity_error,b.metrics.pressure_error)," continuity=",a.metrics.continuity)
end
println("Nonconforming observed velocity orders: ",results.velocity_orders)
println("Nonconforming observed pressure-drop orders: ",results.pressure_orders)
println("Strong mismatch: ",strong_mismatch.metrics)
