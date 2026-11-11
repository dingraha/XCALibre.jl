export interface_cell_flux!, interface_face_flux!

"""
    interface_cell_flux!(balance, coupling, flux)

Overwrite coupled cells with signed sums of integrated overlap fluxes; other
entries are untouched. Positive flux leaves side A and enters side B. This is an
unnormalised balance (no division by cell volume). CPU arrays only. Each cell has
one writer and a fixed summation order. Do not also count the physical interface
faces when adding these contributions to a cell balance.
"""
interface_cell_flux!(balance::AbstractVector, coupling::PlanarInterface, flux::AbstractVector) =
    _gather_interface_flux!(balance, coupling.cell_adjacency, flux, length(coupling.measure))

"""
    interface_face_flux!(face_flux, coupling, flux)

Overwrite physical interface faces with outward integrated flux sums; other faces
are untouched. Uses the same overlap fluxes as `interface_cell_flux!`.
"""
interface_face_flux!(face_flux::AbstractVector, coupling::PlanarInterface, flux::AbstractVector) =
    _gather_interface_flux!(face_flux, coupling.face_adjacency, flux, length(coupling.measure))

function _gather_interface_flux!(out, adj, flux, n)
    get_backend(out) isa CPU && get_backend(flux) isa CPU && get_backend(adj.IDs) isa CPU ||
        _interface_error("flux gathers currently require CPU arrays")
    Base.require_one_based_indexing(out, flux)
    length(flux) == n || throw(DimensionMismatch("expected $n integrated overlap fluxes"))
    Base.mightalias(out, flux) && _interface_error("output must not alias overlap fluxes")
    isempty(adj.IDs) || checkbounds(out, adj.IDs)
    Threads.@threads for row in eachindex(adj.IDs)
        value = zero(eltype(out))
        for k in adj.ranges[row]
            value += adj.signs[k]*flux[adj.overlaps[k]]
        end
        out[adj.IDs[row]] = value
    end
    out
end
