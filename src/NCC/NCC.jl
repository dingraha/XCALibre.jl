# SPDX-License-Identifier: MPL-2.0
module NCC

using StaticArrays: SVector
using LinearAlgebra: dot, norm
using KernelAbstractions: CPU
import Adapt
import ..Solvers
import ..Distribute
import ..IOFormats
import ..Discretise
import ..ModelFramework
import ..Mesh
import ..Fields
import ..Calculate
import ..Solve
import ..ModelPhysics

export NCCCellInput, NCCFaceKey, NCCFaceInput, NCCPatchInput, NCCPhysicalInput
export NCCGeometryInput, FrozenNCCGeometry, NCCScalarContext
export FoundationUncorrected, FoundationCorrected
export ZeroGradient, FixedScalarValues, NoRuntimeFaces
export ncc_scalar, ncc_mobility, ncc_face_flux, ncc_boundaries, ncc_snapshot
export ncc_equation, ncc_update!, ncc_assemble!, ncc_trace_gradient, ncc_correction
export ncc_complete!, ncc_flux, ncc_pressure_component, ncc_recover
export StationaryNCCBinding, StationaryNCCMesh
export ncc_native_scalar, ncc_native_face_scalar, ncc_native_boundaries, ncc_native_equation
export ncc_native_trace_gradient, ncc_native_correction, ncc_native_complete!
export ncc_native_flux, ncc_native_pressure_component, ncc_native_recover

include("types.jl")
include("geometry.jl")
include("context.jl")
include("guards.jl")
include("assembly.jl")
include("correction.jl")
include("flux.jl")
include("native_types.jl")
include("native_mesh.jl")
include("native_boundaries.jl")
include("native_scalar.jl")
include("native_calculate.jl")
include("native_guards.jl")

end # module NCC
