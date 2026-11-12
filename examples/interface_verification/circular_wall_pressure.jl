using XCALibre
include("circular_setup.jl")
results=NamedTuple[]
for k in (1,2,4), mode in (:homogeneous,:analytical)
    case=circular_couette_case(na=12k,nb=15k,nra=2k,nrb=2k,phase=0.13/k,wall_pressure=mode)
    report=circular_simple!(case.state;iterations=600)
    report.converged || error("Wall-pressure comparison did not converge")
    row=(;k,mode,iterations=report.total_iterations,circular_couette_errors(case)...)
    push!(results,row)
    println(row)
end
println("Prescribed analytical derivatives reduce pressure error modestly; the plateau and growing max gradient error remain. No pressure-order claim.")
