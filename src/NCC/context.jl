# SPDX-License-Identifier: MPL-2.0
function context_coefficients(g, policy)
    d = g._data
    factors = zeros(length(d.areas))
    corrections = fill(zero(Vec3),length(d.areas))
    for (i,f) in enumerate(d.input.faces)
        linked = f.key.class === :internal || d.paired[i] != 0
        if linked && policy isa FoundationCorrected
            denominator = max(dot(d.normals[i],d.deltas[i]),0.05d.distances[i])
            factors[i] = 1/denominator
            corrections[i] = d.normals[i]-d.deltas[i]/denominator
        else
            factors[i] = 1/d.distances[i]
        end
    end
    require_ncc(all(isfinite,factors) && all(finite3,corrections), "nonfinite coefficients")
    factors, corrections
end
function owned_values(values, n, label)
    require_ncc(values isa Vector{Float64}, label * " requires Vector{Float64}")
    require_ncc(length(values) == n, label * " length")
    require_ncc(all(isfinite,values), label * " nonfinite")
    copy(values)
end
function ncc_scalar(c::NCCScalarContext; values)
    v = owned_values(values,length(c._geometry._data.input.cells),"scalar")
    ScalarValues(c,v,Val(:owned))
end
function ncc_mobility(c::NCCScalarContext; values)
    v = owned_values(values,length(c._geometry._data.input.faces),"mobility")
    require_ncc(all(x -> x > 0,v), "mobility nonpositive")
    MobilityValues(c,v,Val(:owned))
end
function ncc_face_flux(c::NCCScalarContext; values)
    v = owned_values(values,length(c._geometry._data.input.faces),"q")
    for (i,j) in enumerate(c._geometry._data.paired)
        j == 0 && continue
        require_ncc(abs(v[i]+v[j]) <= 1e-12*max(1.0,abs(v[i]),abs(v[j])), "q unpaired")
    end
    FluxValues(c,v,Val(:owned))
end
function same_context(c, h)
    require_ncc(h isa Union{ScalarValues,MobilityValues,FluxValues,BoundaryValues} && h._context._identity === c._identity, "foreign_context")
    nothing
end
# Reject handles where plain owned input vectors are required, never implicitly unwrap.
for fn in (:ncc_scalar,:ncc_mobility,:ncc_face_flux)
    @eval function $fn(c::NCCScalarContext, h::NCCHandle)
        throw(ArgumentError("NCC: plain copied values required; handles cannot be transferred"))
    end
end
function ncc_boundaries(c::NCCScalarContext; physical)
    require_ncc(physical isa AbstractVector{<:Pair}, "physical must be named pairs")
    d = c._geometry._data
    names = first.(physical)
    require_ncc(all(x -> x isa Symbol,names) && allunique(names), "physical duplicate/name")
    patches = [p for p in d.input.patches if p.kind !== :coupled]
    require_ncc(Set(names) == Set(p.name for p in patches), "physical missing/override")
    result = Pair{Symbol,ScalarBoundaryInput}[]
    for p in patches
        bc = last(physical[findfirst(==(p.name),names)])
        if p.kind in (:empty,:error)
            require_ncc(bc isa NoRuntimeFaces, "empty runtime values")
            push!(result,p.name => NoRuntimeFaces())
        elseif bc isa FixedScalarValues
            values = owned_values(bc.values,length(p.keys),"physical value")
            push!(result,p.name => FixedScalarValues(values))
        else
            require_ncc(bc isa ZeroGradient, "unsupported physical boundary")
            push!(result,p.name => ZeroGradient())
        end
    end
    BoundaryValues(c,result,Val(:owned))
end

function ncc_snapshot(g::FrozenNCCGeometry)
    d = g._data; s = d.input
    input = (schema=s.schema,
        cells=[(id=c.id,centre=c.centre,volume=c.volume) for c in s.cells],
        faces=[(key=f.key,owner=f.owner,neighbour=f.neighbour,Sf=f.Sf,Cf=f.Cf,physical_face=f.physical_face) for f in s.faces],
        patches=[(id=p.id,name=p.name,kind=p.kind,keys=copy(p.keys),partner=p.partner,translation=p.translation) for p in s.patches],
        pairs=copy(s.pairs), physical=s.physical === nothing ? nothing : (points=copy(s.physical.points),polygons=deepcopy(s.physical.polygons)))
    deepcopy(merge(d,(input=input,)))
end
ncc_snapshot(c::NCCScalarContext) = (
    geometry=ncc_snapshot(c._geometry), policy=Symbol(nameof(typeof(c._policy))),
    factors=copy(c._factors), corrections=copy(c._corrections))
ncc_snapshot(h::Union{ScalarValues,MobilityValues,FluxValues}) = (
    values=copy(h._values), context=ncc_snapshot(h._context))
ncc_snapshot(h::BoundaryValues) = (
    physical=[(name=first(p),kind=Symbol(nameof(typeof(last(p)))),values=last(p) isa FixedScalarValues ? copy(last(p).values) : nothing) for p in h._physical], context=ncc_snapshot(h._context))
