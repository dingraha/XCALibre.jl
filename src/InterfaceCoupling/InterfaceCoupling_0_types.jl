export PlanarInterface, InterfaceAdjacency

"""Deterministic gather rows. `signs` are +1 on side A and -1 on side B."""
struct InterfaceAdjacency{VI,VR,VS}
    IDs::VI
    ranges::VR
    overlaps::VI
    signs::VS
end
Adapt.@adapt_structure InterfaceAdjacency

"""
    PlanarInterface

Shared overlap geometry for two coincident straight Mesh2 patches. In 2D, `measure`
is length (unit depth); `area_vector` points out of side A. Flux values are stored
separately and are integrated over each overlap, not flux densities.

`face_adjacency` and `cell_adjacency` contain sorted global IDs and stable overlap
ordering. Construct after mesh reordering; rebuild after any geometry or ID change.
This object alone does not install a CFD boundary condition.
"""
struct PlanarInterface{I,VI,VF,VV,A}
    patch_a::I
    patch_b::I
    face_a::VI
    face_b::VI
    cell_a::VI
    cell_b::VI
    measure::VF
    centre::VV
    area_vector::VV
    face_adjacency::A
    cell_adjacency::A
end
Adapt.@adapt_structure PlanarInterface

function _interface_adjacency(a::Vector{I}, b::Vector{I}) where I<:Integer
    n = length(a)
    Mesh._check_index_capacity(I, 2n, "interface adjacency count")
    owners = [a; b]
    perm = sortperm(owners; alg=MergeSort)
    IDs = I[]
    ranges = UnitRange{I}[]
    overlaps = I[]
    signs = Int8[]
    for (j, k) in enumerate(perm)
        if j == 1 || owners[k] != last(IDs)
            push!(IDs, owners[k])
            push!(ranges, I(j):I(j))
        else
            ranges[end] = first(ranges[end]):I(j)
        end
        push!(overlaps, I(k <= n ? k : k - n))
        push!(signs, k <= n ? Int8(1) : Int8(-1))
    end
    InterfaceAdjacency(IDs, ranges, overlaps, signs)
end
