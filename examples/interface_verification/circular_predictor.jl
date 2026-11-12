using XCALibre
include("circular_setup.jl")
results=NamedTuple[]
for k in (1,2,4), predictor in (:low_order,:reconstructed)
    case=circular_couette_case(na=12k,nb=15k,nra=2k,nrb=2k,phase=.13/k,
        wall_pressure=:analytical,predictor=predictor)
    report=circular_simple!(case.state;iterations=600)
    report.converged || error("Circular predictor comparison did not converge")
    row=(;k,predictor,iterations=report.total_iterations,circular_couette_errors(case)...)
    push!(results,row)
    println(row)
end
println("Reconstructed predictor reduces pressure error in this sequence. Max gradient error is nonmonotonic; no universal order or phase-independent accuracy claim.")
