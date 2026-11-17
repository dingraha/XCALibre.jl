# SPDX-License-Identifier: MPL-2.0
function ncc_native_boundaries(m::StationaryNCCMesh;physical)
    # Reuse the owned facade specification validator, not its numerical assembly.
    specified=ncc_boundaries(m._binding._context;physical=physical)._physical
    result=NativeScalarBoundary[]
    g=m._binding._context._geometry._data
    for p in g.input.patches
        range=m.boundaries[p.id].IDs_range
        if p.kind===:coupled
            push!(result,NCCCoupledScalar(p.id,(mesh=m,kind=:coupled,values=nothing),range))
        else
            bc=last(specified[findfirst(x->first(x)===p.name,specified)])
            kind=bc isa FixedScalarValues ? :fixed : bc isa ZeroGradient ? :zero_gradient : :no_runtime
            values=bc isa FixedScalarValues ? copy(bc.values) : nothing
            record=(mesh=m,kind=kind,values=values)
            push!(result,kind===:no_runtime ? NCCNoRuntimeScalar(p.id,record,range) : NCCPhysicalScalar(p.id,record,range))
        end
    end
    bcs=Tuple(result)
    native_boundary_snapshot(m,bcs)
    bcs
end
function native_boundary_snapshot(m,bcs)
    require_ncc(bcs isa Tuple && all(b->b isa NativeScalarBoundary,bcs),"native_boundary_tuple")
    require_ncc(length(bcs)==length(m.boundaries),"native_boundary_coverage")
    ids=Int32[b.ID for b in bcs]
    require_ncc(sort(ids)==Int32.(1:length(m.boundaries)),"native_boundary_coverage")
    Discretise.validate_boundary_coverage(:native_scalar,bcs,m.boundaries)
    g=m._binding._context._geometry._data
    records=NamedTuple[]
    for p in g.input.patches
        b=bcs[findfirst(==(p.id),ids)]
        require_ncc(b.value.mesh===m && b.IDs_range==m.boundaries[p.id].IDs_range,"native_boundary_binding")
        if p.kind===:coupled
            require_ncc(b isa NCCCoupledScalar && b.value.kind===:coupled && b.value.values===nothing,"native_coupled_override")
        elseif p.kind in (:empty,:error)
            require_ncc(b isa NCCNoRuntimeScalar && b.value.kind===:no_runtime && isempty(b.IDs_range) && b.value.values===nothing,"native_empty_boundary")
        else
            require_ncc(b isa NCCPhysicalScalar && b.value.kind in (:fixed,:zero_gradient),"native_physical_boundary")
            b.value.kind===:fixed ? owned_values(b.value.values,length(b.IDs_range),"native_fixed_values") :
                require_ncc(b.value.values===nothing,"native_zero_gradient_values")
        end
        push!(records,(id=p.id,kind=b.value.kind,range=b.IDs_range,values=b.value.values===nothing ? nothing : copy(b.value.values)))
    end
    records
end
function native_copy_boundaries(m,bcs)
    records=native_boundary_snapshot(m,bcs)
    Tuple(begin
        value=(mesh=m,kind=r.kind,values=r.values===nothing ? nothing : copy(r.values))
        r.kind===:coupled ? NCCCoupledScalar(r.id,value,r.range) :
            r.kind===:no_runtime ? NCCNoRuntimeScalar(r.id,value,r.range) : NCCPhysicalScalar(r.id,value,r.range)
    end for r in records)
end
Discretise._assignment_preflight(::StationaryNCCMesh)=unsupported("native stock assign; use ncc_native_boundaries")
ncc_snapshot(b::NativeScalarBoundary)=(id=b.ID,range=b.IDs_range,kind=b.value.kind,values=b.value.values===nothing ? nothing : copy(b.value.values))
