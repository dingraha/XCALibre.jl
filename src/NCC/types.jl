# SPDX-License-Identifier: MPL-2.0
# Frozen scalar facade only. These handles deliberately are not stock mesh/field types.
const Vec3 = SVector{3,Float64}
abstract type NCCHandle end
abstract type NCCPolicy end
struct FoundationUncorrected <: NCCPolicy end
struct FoundationCorrected <: NCCPolicy end
mutable struct IdentityToken
    _reserved::Nothing
end
new_identity() = IdentityToken(nothing)
require_ncc(ok, message) = ok || throw(ArgumentError("NCC: " * message))

struct NCCCellInput
    id::Int32
    centre::Vec3
    volume::Float64
end
struct NCCFaceKey
    class::Symbol
    patch::Union{Nothing,Int32}
    local_id::Int32
end
struct NCCFaceInput
    key::NCCFaceKey
    owner::Int32
    neighbour::Union{Nothing,Int32}
    Sf::Vec3
    Cf::Vec3
    physical_face::Int32
end
struct NCCPatchInput
    id::Int32
    name::Symbol
    kind::Symbol
    keys::Vector{NCCFaceKey}
    partner::Union{Nothing,Int32}
    translation::Vec3
end
struct NCCPhysicalInput
    points::Vector{Vec3}
    polygons::Vector{Vector{Int32}}
end
struct NCCGeometryInput
    schema::Int
    cells::Vector{NCCCellInput}
    faces::Vector{NCCFaceInput}
    patches::Vector{NCCPatchInput}
    pairs::Vector{Tuple{NCCFaceKey,NCCFaceKey}}
    physical::Union{Nothing,NCCPhysicalInput}
end
NCCGeometryInput(; schema=1, cells, faces, patches, pairs, physical=nothing) =
    NCCGeometryInput(schema, cells, faces, patches, pairs, physical)

# Explicit checked inner constructors prevent accidental positional bypass of facade checks.
struct FrozenNCCGeometry <: NCCHandle
    _data::NamedTuple
    _identity::IdentityToken
    function FrozenNCCGeometry(input::NCCGeometryInput; index_type=Int32, rotation=nothing, moving=false)
        require_ncc(index_type === Int32, "index_type must be Int32")
        require_ncc(rotation === nothing, "rotation unsupported")
        require_ncc(moving === false, "motion unsupported")
        data = validate_geometry(input)
        new(data, new_identity())
    end
end
struct NCCScalarContext <: NCCHandle
    _geometry::FrozenNCCGeometry
    _policy::NCCPolicy
    _factors::Vector{Float64}
    _corrections::Vector{Vec3}
    _identity::IdentityToken
    function NCCScalarContext(g::FrozenNCCGeometry; policy, backend=CPU(), precision=Float64)
        require_ncc(backend isa CPU, "backend must be CPU")
        require_ncc(precision === Float64, "precision must be Float64")
        require_ncc(policy isa Union{FoundationUncorrected,FoundationCorrected}, "unsupported policy")
        factors, corrections = context_coefficients(g, policy)
        new(g, policy, factors, corrections, new_identity())
    end
end
struct ScalarValues <: NCCHandle
    _context::NCCScalarContext
    _values::Vector{Float64}
    ScalarValues(c, v, ::Val{:owned}) = new(c, v)
end
struct MobilityValues <: NCCHandle
    _context::NCCScalarContext
    _values::Vector{Float64}
    MobilityValues(c, v, ::Val{:owned}) = new(c, v)
end
struct FluxValues <: NCCHandle
    _context::NCCScalarContext
    _values::Vector{Float64}
    FluxValues(c, v, ::Val{:owned}) = new(c, v)
end
abstract type ScalarBoundaryInput end
struct ZeroGradient <: ScalarBoundaryInput end
struct NoRuntimeFaces <: ScalarBoundaryInput end
struct FixedScalarValues <: ScalarBoundaryInput
    values::Vector{Float64}
end
struct BoundaryValues <: NCCHandle
    _context::NCCScalarContext
    _physical::Vector{Pair{Symbol,ScalarBoundaryInput}}
    BoundaryValues(c, v, ::Val{:owned}) = new(c, v)
end
