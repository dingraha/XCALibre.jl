# SPDX-License-Identifier: MPL-2.0
unsupported(operation) = throw(ArgumentError("NCC: $operation unsupported by frozen scalar facade"))
for operation in (:run!, :simple!, :piso!, :csimple!, :cpiso!, :simple_MRF!,
                  :laplace!, :multiphase!, :godunov!, :filmModel!, :potential_flow!)
    @eval Solvers.$operation(h::NCCHandle, args...; kwargs...) = unsupported($(string(operation)))
end
Adapt.adapt_structure(to, h::NCCHandle) = unsupported("adaptation")
Distribute.distribute(h::NCCHandle; kwargs...) = unsupported("distribution")
IOFormats.initialise_writer(format::IOFormats.VTK, h::NCCHandle) = unsupported("VTK writer")
IOFormats.initialise_writer(format::IOFormats.OpenFOAM, h::NCCHandle) = unsupported("OpenFOAM writer")
Discretise.assign(h::NCCHandle; kwargs...) = unsupported("stock boundary assignment")
# No Physics conversion, AbstractMesh subtype or raw field/equation access is provided.
