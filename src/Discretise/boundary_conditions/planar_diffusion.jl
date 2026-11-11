export PlanarDiffusion, construct_planar_diffusion, interface_diffusion_flux!

"""Scalar, constant-diffusivity coupling for a static straight Mesh2 interface.
Use both descriptors returned by `construct_planar_diffusion` in `assign`.
Only Time and Laplacian{Linear} terms are supported. No convection, pressure
correction, vector coupling, GPU, or nonorthogonal interface correction is provided.
"""
struct PlanarDiffusion{I,V,R<:UnitRange} <: AbstractBoundary
    ID::I
    value::V
    IDs_range::R
end
Adapt.@adapt_structure PlanarDiffusion

struct PlanarDiffusionValue{C,V,I}
    coupling::C
    distance_a::V
    distance_b::V
    rows::I
    side_a::Bool
end
Adapt.@adapt_structure PlanarDiffusionValue

"""
    construct_planar_diffusion(mesh, patch_a, patch_b; kwargs...)

Return paired scalar diffusion boundaries. Conductance is Γ L/(dA+dB), where
dA and dB are positive normal distances from owner centres to the overlap.
This two-point approximation is exact for fields linear in the normal direction,
but omits tangential/nonorthogonal corrections on skew owner-cell pairs.
Build after mesh reordering and rebuild after geometry or topology changes.
"""
function construct_planar_diffusion(mesh::Mesh2, a::Symbol, b::Symbol; kwargs...)
    c = construct_planar_interface(mesh,a,b; kwargs...)
    T = _get_float(mesh)
    da, db = T[], T[]
    for k in eachindex(c.measure)
        n = c.area_vector[k]/c.measure[k]
        push!(da, dot(c.centre[k] - mesh.cell_centre[c.cell_a[k]], n))
        push!(db, dot(mesh.cell_centre[c.cell_b[k]] - c.centre[k], n))
        isfinite(da[end]) && isfinite(db[end]) && da[end] > 0 && db[end] > 0 ||
            throw(ArgumentError("Planar diffusion requires positive normal owner distances at overlap $k"))
        c.cell_a[k] != c.cell_b[k] || throw(ArgumentError("interface owners must be distinct"))
    end
    function descriptor(name, patch, side)
        range = mesh.boundaries[patch].IDs_range
        rows = _get_int(mesh)[findfirst(==(f),c.face_adjacency.IDs) for f in range]
        PlanarDiffusion(name,PlanarDiffusionValue(c,da,db,rows,side),range)
    end
    (descriptor(a,c.patch_a,true),descriptor(b,c.patch_b,false))
end

function adapt_value(v::PlanarDiffusionValue, mesh)
    _get_backend(mesh) isa CPU || throw(ArgumentError("Planar diffusion requires CPU storage"))
    v
end

function validate_boundary_pair(bc::PlanarDiffusion, BCs)
    v = bc.value
    matches = count(BCs) do other
        other isa PlanarDiffusion && other.value.coupling === v.coupling &&
            other.value.side_a != v.side_a
    end
    matches == 1 || throw(ArgumentError("Planar diffusion boundaries must be assigned as a matching pair"))
    expected = v.side_a ? v.coupling.patch_a : v.coupling.patch_b
    bc.ID == expected || throw(ArgumentError("Planar diffusion descriptor assigned to the wrong patch"))
end

function _extend_matrix(bc::PlanarDiffusion, mesh, i, j)
    _get_backend(mesh) isa CPU || throw(ArgumentError("Planar diffusion requires CPU storage"))
    v, c = bc.value, bc.value.coupling
    # Each side extends only its own rows; duplicate cell pairs are merged by CSR construction.
    owners, neighbours = v.side_a ? (c.cell_a,c.cell_b) : (c.cell_b,c.cell_a)
    for k in eachindex(owners)
        f = v.side_a ? c.face_a[k] : c.face_b[k]
        f in bc.IDs_range && mesh.face_ownerCells[f][1] == owners[k] ||
            throw(ArgumentError("stale planar diffusion geometry; rebuild after mesh reordering"))
    end
    [i;owners], [j;neighbours]
end

# Normal-distance weights of the A and B owner values at an overlap.
@inline function _planar_weights(v, k)
    distance = v.distance_a[k] + v.distance_b[k]
    v.distance_b[k]/distance, v.distance_a[k]/distance
end

@inline _planar_conductance(v, k, gamma) =
    gamma*v.coupling.measure[k]/(v.distance_a[k]+v.distance_b[k])

@inline function (bc::PlanarDiffusion)(
    term::Operator{F,P,I,Laplacian{Linear}}, colval, rowptr, nzval,
    cellID, zcellID, cells, faces, fID, i, component, time
    ) where {F<:ConstantScalar,P<:ScalarField,I}
    gamma = term.flux.values
    isfinite(gamma) && gamma > 0 || throw(ArgumentError("Planar diffusion requires positive finite diffusivity"))
    v, adj = bc.value, bc.value.coupling.face_adjacency
    diagonal = zero(eltype(nzval))
    for entry in adj.ranges[v.rows[i]]
        k = adj.overlaps[entry]
        neighbour = v.side_a ? v.coupling.cell_b[k] : v.coupling.cell_a[k]
        ap = -term.sign*_planar_conductance(v,k,gamma)
        index = spindex(rowptr,colval,cellID,neighbour)
        index != 0 || throw(ArgumentError("missing planar interface matrix connectivity"))
        nzval[index] -= ap
        diagonal += ap
    end
    diagonal, zero(diagonal)
end

# Fail explicitly instead of silently treating unsupported operators as insulated faces.
@inline function (bc::PlanarDiffusion)(term::Operator{F,P,I,O}, args...) where {F,P,I,O}
    throw(ArgumentError("PlanarDiffusion supports only scalar Linear Laplacian with ConstantScalar diffusivity"))
end
@inline function (bc::PlanarDiffusion)(
    term::Operator{F,P,I,<:Time{T}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,T,TF}
    z = zero(TF)
    z,z
end

@inline function boundary_interpolation!(bc::PlanarDiffusion, phif::FaceScalarField,
    phi::ScalarField, boundary_cellsID, time, fID)
    v, c = bc.value, bc.value.coupling
    row = v.rows[fID-first(bc.IDs_range)+1]
    value = zero(eltype(phif.values))
    length_sum = zero(value)
    for entry in c.face_adjacency.ranges[row]
        k = c.face_adjacency.overlaps[entry]
        wa, wb = _planar_weights(v,k)
        trace = wa*phi[c.cell_a[k]] + wb*phi[c.cell_b[k]]
        value += c.measure[k]*trace
        length_sum += c.measure[k]
    end
    phif[fID] = value/length_sum
    nothing
end

"""
    interface_diffusion_flux!(flux, boundary, phi, diffusivity::ConstantScalar)

Overwrite integrated overlap fluxes with Γ L (φA-φB)/(dA+dB), positive outward
from A, independently of which paired descriptor is passed. These are physical
fluxes for -Laplacian; use interface_cell_flux! or interface_face_flux! to gather.
"""
function interface_diffusion_flux!(flux::AbstractVector, bc::PlanarDiffusion,
    phi::ScalarField, gamma::ConstantScalar)
    v, c = bc.value, bc.value.coupling
    get_backend(flux) isa CPU && _get_backend(phi.mesh) isa CPU ||
        throw(ArgumentError("Planar diffusion flux requires CPU storage"))
    Base.require_one_based_indexing(flux)
    length(flux) == length(c.measure) || throw(DimensionMismatch("incorrect overlap flux length"))
    Base.mightalias(flux,phi.values) && throw(ArgumentError("flux output must not alias field values"))
    isfinite(gamma.values) && gamma.values > 0 || throw(ArgumentError("diffusivity must be positive and finite"))
    for k in eachindex(flux)
        flux[k] = _planar_conductance(v,k,gamma.values)*(phi[c.cell_a[k]]-phi[c.cell_b[k]])
    end
    flux
end
