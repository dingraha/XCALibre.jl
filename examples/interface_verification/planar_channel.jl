# Run with: julia --threads=4 --project=<environment with XCALibre> planar_channel.jl
include("channel_setup.jl")

function run_channel(; first_iterations=20, kwargs...)
    model,config = channel_case(;kwargs...)
    state = PlanarSimpleState(model,config;continuity=1e-9)
    first = planar_simple!(state;iterations=first_iterations)
    # Same state: preserve corrected individual overlaps, not just face sums.
    continued = planar_simple!(state)
    metrics = channel_metrics(model,config,state.work.transport.values;transport=state.work.transport)
    (; model,config,state,first,continued,metrics)
end

results = run_channel()
println("\nStatic nonconforming planar channel (CPU, unit density)")
println("Converged: ",results.continued.converged,"; cumulative iterations: ",results.state.work.iteration[])
for key in (:continuity,:boundary_balance,:interface_balance,:interface_flow,:velocity_error,:pressure_error,:pressure_drop,:exact_drop,:sections)
    println(key,": ",getproperty(results.metrics,key))
end
results.continued.converged || error("channel did not converge within its iteration budget")
