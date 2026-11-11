export PlanarPressure, construct_planar_pressure
export interface_pressure_flux!, correct_interface_pressure_flux!

"""Paired CPU scalar pressure-Laplacian boundaries for a static straight Mesh2
interface. This is a unit-density projection primitive, not SIMPLE/PISO support.
No vector coupling, density weighting, or nonorthogonal corrections are provided.
"""
struct PlanarPressure{I,V,R<:UnitRange} <: AbstractBoundary
    ID::I
    value::V
    IDs_range::R
end
Adapt.@adapt_structure PlanarPressure

"""Construct paired static pressure boundaries. Supply an
`InterfacePressureCoefficient(mesh,pair[1])` to `Laplacian{Linear}` and refresh
it from cell mobilities rD=V/aP before assembly. Rebuild after geometry/ID changes.
"""
function construct_planar_pressure(mesh::Mesh2,a::Symbol,b::Symbol; kwargs...)
    map(bc -> PlanarPressure(bc.ID,bc.value,bc.IDs_range),
        construct_planar_diffusion(mesh,a,b;kwargs...))
end

"""
    construct_planar_pressure(momentum_pair::Tuple{PlanarTransport,PlanarTransport})

Reuse a matching momentum boundary pair's coupling and normal distances. Required
when one InterfaceTransportFlux is used for both convection and pressure
correction: independently constructed geometry objects are not interchangeable.
"""
function construct_planar_pressure(pair::Tuple{PlanarTransport,PlanarTransport})
    a,b = pair
    a.value.coupling === b.value.coupling && a.value.side_a != b.value.side_a && a.ID != b.ID ||
        throw(ArgumentError("momentum descriptors must form a matching opposite-side pair"))
    for bc in pair
        expected = bc.value.side_a ? bc.value.coupling.patch_a : bc.value.coupling.patch_b
        # Before assign, IDs are names; afterwards they are numeric patch IDs.
        bc.ID isa Integer && bc.ID != expected &&
            throw(ArgumentError("momentum descriptor assigned to the wrong patch"))
        rowfaces = bc.value.coupling.face_adjacency.IDs[bc.value.rows]
        rowfaces == collect(bc.IDs_range) ||
            throw(ArgumentError("momentum descriptor face range must match its geometry"))
    end
    a.value.distance_a === b.value.distance_a && a.value.distance_b === b.value.distance_b ||
        throw(ArgumentError("momentum pair must share normal distance storage"))
    map(bc -> PlanarPressure(bc.ID,bc.value,bc.IDs_range),pair)
end

@inline _diffusion_boundary(bc::PlanarPressure) = PlanarDiffusion(bc.ID,bc.value,bc.IDs_range)
_extend_matrix(bc::PlanarPressure,mesh,i,j) = _extend_matrix(_diffusion_boundary(bc),mesh,i,j)

function validate_boundary_pair(bc::PlanarPressure,BCs)
    c = bc.value.coupling
    matches = count(BCs) do other
        other isa PlanarPressure && other.value.coupling === c && other.value.side_a != bc.value.side_a
    end
    matches == 1 || throw(ArgumentError("Planar pressure boundaries must be assigned as a matching pair"))
    bc.ID == (bc.value.side_a ? c.patch_a : c.patch_b) ||
        throw(ArgumentError("Planar pressure descriptor assigned to the wrong patch"))
end

@inline function (bc::PlanarPressure)(
    term::Operator{F,P,I,Laplacian{Linear}},colval,rowptr,nzval,
    cellID,zcellID,cells,faces,fID,i,component,time
    ) where {F<:InterfacePressureCoefficient,P<:ScalarField,I}
    coefficient,v = term.flux,bc.value
    coefficient.coupling === v.coupling && coefficient.mesh === term.phi.mesh ||
        throw(ArgumentError("pressure coefficient, boundary, and field must match"))
    diagonal = zero(eltype(nzval))
    adj = v.coupling.face_adjacency
    for entry in adj.ranges[v.rows[i]]
        k = adj.overlaps[entry]
        G = coefficient.conductance[k]
        isfinite(G) && G > 0 || throw(ArgumentError("refresh positive pressure coefficients before assembly"))
        neighbour = v.side_a ? v.coupling.cell_b[k] : v.coupling.cell_a[k]
        index = spindex(rowptr,colval,cellID,neighbour)
        index != 0 || throw(ArgumentError("missing pressure interface matrix connectivity"))
        ap = -term.sign*G
        nzval[index] -= ap
        diagonal += ap
    end
    diagonal,zero(diagonal)
end

@inline function (bc::PlanarPressure)(
    term::Operator{F,P,I,<:Time{T}},colval,rowptr,nzval,cellID,zcellID,
    cells::AbstractVector{<:Cell{TF}},faces,fID,i,component,time
    ) where {F,P,I,T,TF}
    zero(TF),zero(TF)
end

@inline function (bc::PlanarPressure)(term::Operator,args...)
    throw(ArgumentError("PlanarPressure supports scalar Linear Laplacian with InterfacePressureCoefficient and Time only"))
end

@inline boundary_interpolation!(bc::PlanarPressure,phif::FaceScalarField,
    phi::ScalarField,boundary_cellsID,time,fID) =
    boundary_interpolation!(_diffusion_boundary(bc),phif,phi,boundary_cellsID,time,fID)

"""
    interface_pressure_flux!(out, boundary, pressure, coefficient)

Recover pressure correction G*(pA-pB), outward from A regardless of descriptor
side. Uses conductances stored for negative-Laplacian assembly, never summed
matrix entries. No density or operator sign is included. Keep coefficients fixed
between assembly, pressure solve, and correction.
"""
function interface_pressure_flux!(out::AbstractVector,bc::PlanarPressure,
    p::ScalarField,coefficient::InterfacePressureCoefficient)
    c = bc.value.coupling
    coefficient.coupling === c && coefficient.mesh === p.mesh ||
        throw(ArgumentError("pressure coefficient, boundary, and field must match"))
    get_backend(out) isa CPU && get_backend(p.values) isa CPU &&
        get_backend(coefficient.conductance) isa CPU || throw(ArgumentError("pressure correction requires CPU storage"))
    Base.require_one_based_indexing(out)
    length(out) == length(c.measure) || throw(DimensionMismatch("incorrect overlap correction length"))
    any(x -> Base.mightalias(out,x),(p.values,coefficient.values,coefficient.overlaps,coefficient.conductance)) &&
        throw(ArgumentError("pressure correction output must not alias input storage"))
    all(x -> isfinite(x) && x > 0,coefficient.conductance) ||
        throw(ArgumentError("refresh positive pressure coefficients before correction"))
    for k in eachindex(out)
        out[k] = coefficient.conductance[k]*(p[c.cell_a[k]]-p[c.cell_b[k]])
    end
    out
end

"""
    correct_interface_pressure_flux!(transport, predicted, boundary, pressure, coefficient)

Set overlaps to predicted + G*(pA-pB), then refresh signed physical interface face
sums; other faces are untouched. Supply a saved predictor to make repeated calls
idempotent. This corrects only the interface: ordinary interior/exterior fluxes
must be corrected consistently by the caller for a complete pressure projection.
"""
function correct_interface_pressure_flux!(transport::InterfaceTransportFlux,
    predicted::AbstractVector,bc::PlanarPressure,p::ScalarField,coefficient::InterfacePressureCoefficient)
    transport.coupling === bc.value.coupling && transport.mesh === p.mesh ||
        throw(ArgumentError("transport and pressure boundary must match"))
    get_backend(predicted) isa CPU || throw(ArgumentError("predictor requires CPU storage"))
    Base.require_one_based_indexing(predicted)
    length(predicted) == length(transport.overlaps) || throw(DimensionMismatch("incorrect predictor length"))
    # An aliased predictor would be overwritten and cease to be a saved predictor.
    (Base.mightalias(predicted,transport.overlaps) || Base.mightalias(predicted,transport.values)) &&
        throw(ArgumentError("save predictor separately from corrected transport storage"))
    correction = similar(transport.overlaps)
    interface_pressure_flux!(correction,bc,p,coefficient)
    update_interface_flux!(transport,predicted .+ correction)
end
