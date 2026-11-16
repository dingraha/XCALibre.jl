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

export NCCCellInput, NCCFaceKey, NCCFaceInput, NCCPatchInput, NCCPhysicalInput
export NCCGeometryInput, FrozenNCCGeometry, NCCScalarContext
export FoundationUncorrected, FoundationCorrected
export ZeroGradient, FixedScalarValues, NoRuntimeFaces
export ncc_scalar, ncc_mobility, ncc_face_flux, ncc_boundaries, ncc_snapshot

include("types.jl")
include("geometry.jl")
include("context.jl")
include("guards.jl")

end # module NCC
