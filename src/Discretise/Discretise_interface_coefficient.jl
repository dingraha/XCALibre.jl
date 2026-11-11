export InterfacePressureCoefficient, update_interface_coefficient!

"""
    InterfacePressureCoefficient(mesh, boundary)

CPU pressure mobility storage for a boundary from `construct_planar_pressure`.
`values` are ordinary face mobilities; `overlaps` are overlap mobilities and
`conductance` stores mobility*length/normal_distance. Initialise and refresh with
`update_interface_coefficient!` from the cell field rD=V/aP. Initially unit density
and one coupling per field. Do not mutate coefficient arrays between assembly and
flux correction; refresh coefficients and reassemble together.
"""
struct InterfacePressureCoefficient{V,C,G,M} <: AbstractScalarField
    values::V
    overlaps::V
    conductance::V
    coupling::C
    geometry::G
    mesh::M
end
Adapt.@adapt_structure InterfacePressureCoefficient

function InterfacePressureCoefficient(mesh::Mesh2,bc)
    bc isa PlanarPressure || throw(ArgumentError("construct pressure coefficients from a PlanarPressure boundary"))
    v = bc.value
    # Reuse the transport field's CPU, topology, and numeric-type checks.
    transport = InterfaceTransportFlux(mesh,v.coupling)
    InterfacePressureCoefficient(transport.values,transport.overlaps,
        zeros(eltype(transport.overlaps),length(transport.overlaps)),v.coupling,v,mesh)
end

"""
    update_interface_coefficient!(coefficient, rD::ScalarField)

Validate all proposed storage values before copying them (not thread-safe against concurrent updates). Arithmetic
interpolation uses ordinary mesh weights internally and normal-distance weights
on overlaps. Physical interface face values are length-weighted overlap averages;
interface assembly and correction use individual stored conductances instead.
All cell mobilities and resulting conductances must be positive and finite.
"""
function update_interface_coefficient!(coefficient::InterfacePressureCoefficient,rD::ScalarField)
    mesh,v = coefficient.mesh,coefficient.geometry
    rD.mesh === mesh || throw(ArgumentError("pressure mobility and coefficient must use the same mesh"))
    _get_backend(mesh) isa CPU && get_backend(coefficient.values) isa CPU &&
        get_backend(coefficient.overlaps) isa CPU && get_backend(coefficient.conductance) isa CPU &&
        get_backend(rD.values) isa CPU || throw(ArgumentError("pressure coefficients require CPU storage"))
    length(rD.values) == length(mesh.cells) || throw(DimensionMismatch("incorrect cell mobility length"))
    all(x -> isfinite(x) && x > 0,rD.values) || throw(ArgumentError("cell mobilities must be positive and finite"))
    T = eltype(coefficient.values)
    values = zeros(T,length(mesh.faces))
    for f in eachindex(values)
        a,b = mesh.face_ownerCells[f]
        w = mesh.face_weight[f]
        values[f] = a == b ? rD[a] : w*rD[a]+(one(w)-w)*rD[b]
    end
    c = coefficient.coupling
    overlaps,conductance = similar(coefficient.overlaps),similar(coefficient.conductance)
    for k in eachindex(overlaps)
        wa,wb = _planar_weights(v,k)
        overlaps[k] = wa*rD[c.cell_a[k]]+wb*rD[c.cell_b[k]]
        conductance[k] = _planar_conductance(v,k,overlaps[k])
    end
    adj = c.face_adjacency
    for row in eachindex(adj.IDs)
        total,measure = zero(T),zero(T)
        for entry in adj.ranges[row]
            k = adj.overlaps[entry]
            total += c.measure[k]*overlaps[k]
            measure += c.measure[k]
        end
        values[adj.IDs[row]] = total/measure
    end
    all(x -> isfinite(x) && x > 0,values) &&
        all(x -> isfinite(x) && x > 0,overlaps) &&
        all(x -> isfinite(x) && x > 0,conductance) ||
        throw(ArgumentError("pressure coefficient conversion produced invalid values"))
    copyto!(coefficient.values,values)
    copyto!(coefficient.overlaps,overlaps)
    copyto!(coefficient.conductance,conductance)
    coefficient
end
