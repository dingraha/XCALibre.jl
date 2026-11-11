export InterfaceTransportFlux, update_interface_flux!

"""
    InterfaceTransportFlux(mesh, coupling)

CPU integrated transport fluxes: `values` holds ordinary physical-face fluxes and
`overlaps` holds fluxes outward from side A. Interface assembly reads overlaps,
never their face sums. Fill ordinary faces through indexing, and update overlaps
with `update_interface_flux!`. One coupling per field is supported initially.
"""
struct InterfaceTransportFlux{V,C,M} <: AbstractScalarField
    values::V
    overlaps::V
    coupling::C
    mesh::M
end
Adapt.@adapt_structure InterfaceTransportFlux

function InterfaceTransportFlux(mesh::Mesh2, c::PlanarInterface)
    _get_backend(mesh) isa CPU || throw(ArgumentError("interface transport requires CPU storage"))
    get_backend(c.measure) isa CPU || throw(ArgumentError("interface transport requires CPU coupling storage"))
    for (patch,face,cell) in ((c.patch_a,c.face_a,c.cell_a),(c.patch_b,c.face_b,c.cell_b))
        checkbounds(mesh.boundaries,patch)
        for k in eachindex(face)
            face[k] in mesh.boundaries[patch].IDs_range && mesh.face_ownerCells[face[k]][1] == cell[k] ||
                throw(ArgumentError("coupling does not match mesh topology; rebuild after reordering"))
        end
    end
    T = _get_float(mesh)
    eltype(c.measure) == T || throw(ArgumentError("coupling and mesh floating types must match"))
    InterfaceTransportFlux(zeros(T,length(mesh.faces)),zeros(T,length(c.measure)),c,mesh)
end

"""
    update_interface_flux!(field, integrated_fluxes)

Validate and copy overlap fluxes, then overwrite physical interface faces with
signed sums. Other faces remain untouched. Geometry and sparse connectivity are
unchanged. Call before assembly after every change to the prescribed flux.
"""
function update_interface_flux!(field::InterfaceTransportFlux, flux::AbstractVector)
    get_backend(flux) isa CPU && get_backend(field.values) isa CPU &&
        get_backend(field.overlaps) isa CPU ||
        throw(ArgumentError("interface transport requires CPU storage"))
    Base.require_one_based_indexing(flux)
    length(flux) == length(field.overlaps) || throw(DimensionMismatch("incorrect overlap flux length"))
    all(isfinite,flux) || throw(ArgumentError("overlap fluxes must be finite"))
    converted = eltype(field.overlaps).(flux)
    all(isfinite,converted) || throw(ArgumentError("overlap fluxes overflow storage type"))
    copyto!(field.overlaps,converted)
    interface_face_flux!(field.values,field.coupling,field.overlaps)
    field
end
