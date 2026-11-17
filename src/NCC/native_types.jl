# SPDX-License-Identifier: MPL-2.0
# Native integration is opt-in; storage is owned/read-only by contract, not a security boundary.
struct StationaryNCCBinding{M} <: NCCHandle
    _context::NCCScalarContext
    _runtime::M
    _native_to_input::Vector{Int32}
    _input_to_native::Vector{Int32}
    _identity::IdentityToken
    StationaryNCCBinding(c,m,a,b,::Val{:owned}) = new{typeof(m)}(c,m,a,b,new_identity())
end
struct StationaryNCCMesh{B<:StationaryNCCBinding} <: Mesh.AbstractMesh
    _binding::B
end
function Base.getproperty(m::StationaryNCCMesh, name::Symbol)
    name === :_binding && return getfield(m,name)
    getproperty(getfield(m,:_binding)._runtime,name)
end
Base.propertynames(m::StationaryNCCMesh, private::Bool=false) =
    private ? (:_binding,propertynames(m._binding._runtime)...) : propertynames(m._binding._runtime)
Base.show(io::IO,m::StationaryNCCMesh) = print(io,"Experimental stationary NCC domain (",length(m.cells)," cells; operator-only topology)")
const NativeScalarField = Fields.ScalarField{V,M} where {V,M<:StationaryNCCMesh}
const NativeFaceScalarField = Fields.FaceScalarField{V,M} where {V,M<:StationaryNCCMesh}
const NativeVectorField = Fields.VectorField{X,Y,Z,M} where {X,Y,Z,M<:StationaryNCCMesh}

abstract type NativeScalarBoundary <: Discretise.AbstractBoundary end
struct NCCCoupledScalar{M} <: NativeScalarBoundary
    ID::Int32
    value::M
    IDs_range::UnitRange{Int32}
end
struct NCCPhysicalScalar{M} <: NativeScalarBoundary
    ID::Int32
    value::M
    IDs_range::UnitRange{Int32}
end
struct NCCNoRuntimeScalar{M} <: NativeScalarBoundary
    ID::Int32
    value::M
    IDs_range::UnitRange{Int32}
end

mutable struct NativeScalarStorage{E,M,O,B} <: ModelFramework.AbstractEquation
    _native::E
    _mesh::M
    _identity::IdentityToken
    _form::Symbol
    _scheme::Any
    _operands::O
    _boundaries::B
    _input::Any
    _hardware::Any
    _compact::Vector{Float64}
    _vs::Vector{Float64}
    _applied::Vector{Float64}
    _epoch::UInt64
    _state::Symbol
    _model::Any
    function NativeScalarStorage(n,m,f,s,o,b,::Val{:owned})
        new{typeof(n),typeof(m),typeof(o),typeof(b)}(n,m,new_identity(),f,s,o,b,nothing,nothing,
            zeros(length(m.cells)),zeros(length(m.cells)),zeros(length(m.faces)),UInt64(0),:unassembled,nothing)
    end
end
function Base.getproperty(s::NativeScalarStorage, name::Symbol)
    name in (:A,:opA,:b,:R,:Fx,:diag_nz,:face_nz) && return getproperty(getfield(s,:_native),name)
    getfield(s,name)
end
const NativeScalarEquation = ModelFramework.ModelEquation{T,M,E,S,P} where {T<:ModelFramework.ScalarModel,M,E<:NativeScalarStorage,S,P}

struct NativeTraceGradient{E} <: NCCHandle
    _equation::E
    _epoch::UInt64
    _trace::Vector{Float64}
    _gradient::Vector{Vec3}
end
struct NativeCorrection{E} <: NCCHandle
    _equation::E
    _epoch::UInt64
    _values::Vector{Float64}
    _divergence::Vector{Float64}
end
struct NativeTransportFlux{E} <: NCCHandle
    _equation::E
    _epoch::UInt64
    _values::Vector{Float64}
    _divergence::Vector{Float64}
end
struct NativePressureComponent{E} <: NCCHandle
    _equation::E
    _epoch::UInt64
    _p::Vector{Float64}
    _increment::Vector{Float64}
end
struct NativeRecoveredFlux{E} <: NCCHandle
    _equation::E
    _epoch::UInt64
    _values::Vector{Float64}
    _increment::Vector{Float64}
    _divergence::Vector{Float64}
end
