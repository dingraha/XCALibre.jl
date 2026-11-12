using XCALibre
include("circular_setup.jl")
case=circular_couette_case()
results=circular_simple!(case.state;iterations=500)
results.converged || error("Static circular Couette did not converge")
println("Static circular Couette: ",results.total_iterations," iterations")
println("Errors: ",circular_couette_errors(case))
println("Final residuals: ",results.history[end])
println("Pressure uses homogeneous wall Neumann, not exact radial Couette dp/dr; no pressure-order claim.")
