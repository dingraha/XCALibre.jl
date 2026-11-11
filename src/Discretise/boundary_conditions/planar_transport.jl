export PlanarTransport, construct_planar_transport, interface_transport_flux!

"""Paired CPU scalar upwind transport boundaries, with optional constant Linear
 diffusion and Time terms. Geometric interpolation is not the upwind flux trace.
No pressure correction, vector transport, or moving interfaces are supported.
"""
struct PlanarTransport{I,V,R<:UnitRange} <: AbstractBoundary
    ID::I
    value::V
    IDs_range::R
end
Adapt.@adapt_structure PlanarTransport

"""
    construct_planar_transport(mesh, patch_a, patch_b; kwargs...)

Construct paired static straight-interface boundaries. Create the transport field
as `InterfaceTransportFlux(mesh, pair[1].value.coupling)` and supply it to
`Divergence{Upwind}`. Both sides must use the same coupling object.
"""
function construct_planar_transport(mesh::Mesh2,a::Symbol,b::Symbol; kwargs...)
    pair = construct_planar_diffusion(mesh,a,b; kwargs...)
    map(bc -> PlanarTransport(bc.ID,bc.value,bc.IDs_range),pair)
end

# Share geometry, matrix extension, diffusion, and geometric interpolation with
# PlanarDiffusion, retaining its existing public API and supported scope.
@inline _diffusion_boundary(bc::PlanarTransport) = PlanarDiffusion(bc.ID,bc.value,bc.IDs_range)
_extend_matrix(bc::PlanarTransport,mesh,i,j) = _extend_matrix(_diffusion_boundary(bc),mesh,i,j)

function validate_boundary_pair(bc::PlanarTransport,BCs)
    c = bc.value.coupling
    matches = count(BCs) do other
        other isa PlanarTransport && other.value.coupling === c &&
            other.value.side_a != bc.value.side_a
    end
    matches == 1 || throw(ArgumentError("Planar transport boundaries must be assigned as a matching pair"))
    expected = bc.value.side_a ? c.patch_a : c.patch_b
    bc.ID == expected || throw(ArgumentError("Planar transport descriptor assigned to the wrong patch"))
end

@inline function (bc::PlanarTransport)(
    term::Operator{F,P,I,Divergence{Upwind}}, colval,rowptr,nzval,
    cellID,zcellID,cells,faces,fID,i,component,time
    ) where {F<:InterfaceTransportFlux,P<:ScalarField,I}
    v, flux = bc.value,term.flux
    flux.coupling === v.coupling || throw(ArgumentError("transport flux and boundary coupling must match"))
    flux.mesh === term.phi.mesh || throw(ArgumentError("transport flux and scalar field must use the same mesh"))
    adj = v.coupling.face_adjacency
    diagonal = zero(eltype(nzval))
    for entry in adj.ranges[v.rows[i]]
        k = adj.overlaps[entry]
        q = flux.overlaps[k]
        isfinite(q) || throw(ArgumentError("overlap fluxes must be finite"))
        # Choose the physical donor before applying the algebraic operator sign.
        # This also makes -Divergence exactly the negative of +Divergence.
        a,b = max(q,zero(q)),min(q,zero(q))
        ac,an = v.side_a ? (a,b) : (-b,-a)
        neighbour = v.side_a ? v.coupling.cell_b[k] : v.coupling.cell_a[k]
        index = spindex(rowptr,colval,cellID,neighbour)
        index != 0 || throw(ArgumentError("missing planar interface matrix connectivity"))
        nzval[index] += term.sign*an
        diagonal += term.sign*ac
    end
    diagonal,zero(diagonal)
end

@inline function (bc::PlanarTransport)(
    term::Operator{F,P,I,Laplacian{Linear}},colval,rowptr,nzval,
    cellID,zcellID,cells,faces,fID,i,component,time
    ) where {F<:ConstantScalar,P<:ScalarField,I}
    _diffusion_boundary(bc)(term,colval,rowptr,nzval,cellID,zcellID,cells,faces,fID,i,component,time)
end

@inline function (bc::PlanarTransport)(
    term::Operator{F,P,I,<:Time{T}},colval,rowptr,nzval,cellID,zcellID,
    cells::AbstractVector{<:Cell{TF}},faces,fID,i,component,time
    ) where {F,P,I,T,TF}
    zero(TF),zero(TF)
end

@inline function (bc::PlanarTransport)(term::Operator,args...)
    throw(ArgumentError("PlanarTransport supports scalar Upwind divergence with InterfaceTransportFlux, constant Linear diffusion, and Time only"))
end

@inline boundary_interpolation!(bc::PlanarTransport,phif::FaceScalarField,
    phi::ScalarField,boundary_cellsID,time,fID) =
    boundary_interpolation!(_diffusion_boundary(bc),phif,phi,boundary_cellsID,time,fID)

interface_diffusion_flux!(flux::AbstractVector,bc::PlanarTransport,phi::ScalarField,gamma::ConstantScalar) =
    interface_diffusion_flux!(flux,_diffusion_boundary(bc),phi,gamma)

"""
    interface_transport_flux!(out, boundary, phi, transport)

Recover integrated scalar flux q*phi_upwind per overlap, outward from A regardless
of the descriptor side. Uses the same donor as assembly. Does not include an
operator sign or diffusion. Gather for diagnostics, not as an additional source.
"""
function interface_transport_flux!(out::AbstractVector,bc::PlanarTransport,
    phi::ScalarField,transport::InterfaceTransportFlux)
    c = bc.value.coupling
    transport.coupling === c || throw(ArgumentError("transport flux and boundary coupling must match"))
    transport.mesh === phi.mesh || throw(ArgumentError("transport flux and scalar field must use the same mesh"))
    get_backend(out) isa CPU && _get_backend(phi.mesh) isa CPU &&
        get_backend(transport.overlaps) isa CPU || throw(ArgumentError("interface transport requires CPU storage"))
    Base.require_one_based_indexing(out)
    length(out) == length(c.measure) || throw(DimensionMismatch("incorrect overlap output length"))
    (Base.mightalias(out,phi.values) || Base.mightalias(out,transport.overlaps) ||
        Base.mightalias(out,transport.values)) && throw(ArgumentError("transport output must not alias input storage"))
    all(isfinite,transport.overlaps) || throw(ArgumentError("overlap fluxes must be finite"))
    for k in eachindex(out)
        q = transport.overlaps[k]
        out[k] = q == 0 ? zero(eltype(out)) : q*phi[q > 0 ? c.cell_a[k] : c.cell_b[k]]
    end
    out
end
