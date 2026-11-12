module InterfaceCoupling

using Adapt
using KernelAbstractions
using LinearAlgebra
using StaticArrays
using XCALibre.Mesh

include("InterfaceCoupling_0_types.jl")
include("InterfaceCoupling_1_planar.jl")
include("InterfaceCoupling_2_flux.jl")
include("InterfaceCoupling_3_circular.jl")
include("InterfaceCoupling_4_circular_metrics.jl")
include("InterfaceCoupling_5_circular_reconstruction.jl")

end
