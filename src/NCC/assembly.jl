# SPDX-License-Identifier: MPL-2.0
# Integrated finite-volume algebra; intentionally not a stock AbstractEquation.
mutable struct ScalarTransport{A} <: NCCHandle
    _context::NCCScalarContext
    _identity::IdentityToken
    _form::Symbol
    _scheme::Any
    _input::NamedTuple
    _matrix::A
    _slots::Dict{Tuple{Int32,Int32},Int32}
    _compact::Vector{Float64}
    _full::Vector{Float64}
    _vs::Vector{Float64}
    _applied::Vector{Float64}
    _epoch::UInt64
    _state::Symbol
    ScalarTransport(c,f,s,input,A,slots,n,::Val{:owned}) = new{typeof(A)}(
        c,new_identity(),f,s,input,A,slots,zeros(n),zeros(n),zeros(n),
        zeros(length(c._geometry._data.input.faces)),UInt64(0),:unassembled)
end
has_convection(f) = f in (:convection,:convection_minus_diffusion)
diffusion_sign(f) = f === :positive_diffusion ? 1 : f in (:negative_diffusion,:convection_minus_diffusion) ? -1 : 0
function next_epoch(e)
    require_ncc(e._epoch < typemax(UInt64), "epoch_overflow")
    e._epoch+UInt64(1)
end
function equation_input(c,form,p,q,mobility,source,boundaries)
    require_ncc(p isa ScalarValues, "scalar_handle");same_context(c,p)
    require_ncc(boundaries isa BoundaryValues, "boundary_handle");same_context(c,boundaries)
    if has_convection(form)
        require_ncc(q isa FluxValues, "missing_q");same_context(c,q)
    else
        require_ncc(q === nothing, "irrelevant_q")
    end
    if diffusion_sign(form) != 0
        require_ncc(mobility isa MobilityValues, "missing_mobility");same_context(c,mobility)
        for (i,j) in enumerate(c._geometry._data.paired)
            j == 0 && continue
            a,b=mobility._values[i],mobility._values[j]
            require_ncc(abs(a-b)<=1e-12*max(1.0,abs(a),abs(b)), "paired_mobility")
        end
    else
        require_ncc(mobility === nothing, "irrelevant_mobility")
    end
    s=owned_values(source,length(c._geometry._data.input.cells),"source")
    (p=copy(p._values),q=q===nothing ? nothing : copy(q._values),
     mobility=mobility===nothing ? nothing : copy(mobility._values),source=s,
     boundaries=deepcopy(boundaries._physical))
end
function transport_structure(c)
    cells=c._geometry._data.input.cells;faces=c._geometry._data.input.faces
    pairs=Set((x.id,x.id) for x in cells)
    for f in faces
        f.neighbour === nothing && continue
        push!(pairs,(f.owner,f.neighbour))
        f.key.class === :internal && push!(pairs,(f.neighbour,f.owner))
    end
    ordered=sort!(collect(pairs))
    require_ncc(length(ordered)+1 <= typemax(Int32), "CSR_capacity")
    A=ModelFramework._build_A(CPU(),Int32[first(p) for p in ordered],
        Int32[last(p) for p in ordered],zeros(length(ordered)),length(cells))
    nz,col,ptr=ModelFramework.get_sparse_fields(A)
    require_ncc(eltype(col)===Int32 && eltype(ptr)===Int32, "CSR_index_type")
    slots=Dict{Tuple{Int32,Int32},Int32}()
    for i in eachindex(cells), k in ptr[i]:ptr[i+1]-1
        slots[(Int32(i),col[k])]=Int32(k)
    end
    require_ncc(Set(keys(slots))==pairs && length(nz)==length(pairs), "CSR_structure")
    A,slots
end
function ncc_equation(c::NCCScalarContext,p;form,convection=nothing,q=nothing,mobility=nothing,source,boundaries)
    require_ncc(form in (:convection,:positive_diffusion,:negative_diffusion,:convection_minus_diffusion), "unsupported_form")
    require_ncc(has_convection(form) ? convection isa Union{Discretise.Linear,Discretise.Upwind} : convection===nothing, "unsupported_scheme")
    input=equation_input(c,form,p,q,mobility,source,boundaries)
    A,slots=transport_structure(c)
    e=ScalarTransport(c,form,convection,input,A,slots,length(input.p),Val(:owned))
    compact_terms(e,input) # reject derived overflow before publishing constructor
    e
end
function ncc_update!(e::ScalarTransport;p,q,mobility,source,boundaries)
    epoch=next_epoch(e)
    input=equation_input(e._context,e._form,p,q,mobility,source,boundaries)
    compact_terms(e,input)
    e._input=input;e._epoch=epoch;e._state=:unassembled
    e
end
function physical_trace(input,geometry,f)
    patch=geometry.input.patches[f.key.patch]
    bc=last(input.boundaries[findfirst(p->first(p)===patch.name,input.boundaries)])
    bc isa FixedScalarValues ? (true,bc.values[f.key.local_id]) : (false,input.p[f.owner])
end
function compact_terms(e,input)
    g=e._context._geometry._data
    nz=zeros(length(first(ModelFramework.get_sparse_fields(e._matrix))))
    vs=[c.volume*input.source[i] for (i,c) in enumerate(g.input.cells)]
    b=copy(vs);sigma=diffusion_sign(e._form)
    function add(r,c,v)
        nz[e._slots[(r,c)]]+=v
    end
    for (i,f) in enumerate(g.input.faces)
        q=input.q===nothing ? 0.0 : input.q[i]
        k=input.mobility===nothing ? 0.0 : input.mobility[i]*g.areas[i]*e._context._factors[i]
        require_ncc(isfinite(k), "coefficient_overflow")
        if f.neighbour !== nothing
            alpha=e._scheme isa Discretise.Upwind ? max(q,0.0) : q*g.weights[i]
            beta=e._scheme isa Discretise.Upwind ? min(q,0.0) : q*(1-g.weights[i])
            a=alpha-sigma*k;d=beta+sigma*k
            add(f.owner,f.owner,a);add(f.owner,f.neighbour,d)
            if f.key.class===:internal
                add(f.neighbour,f.owner,-a);add(f.neighbour,f.neighbour,-d)
            end
        else
            fixed,pb=physical_trace(input,g,f)
            if fixed
                add(f.owner,f.owner,-sigma*k)
                b[f.owner]-=(q+sigma*k)*pb
            else
                add(f.owner,f.owner,q)
            end
        end
    end
    require_ncc(all(isfinite,vs), "source_overflow")
    require_ncc(all(isfinite,nz) && all(isfinite,b), "assembly_overflow")
    nz,b,vs
end
function ncc_assemble!(e::ScalarTransport)
    epoch=next_epoch(e)
    nz,b,vs=compact_terms(e,e._input)
    # All scratch results finite before committing; every structural slot reset.
    copyto!(first(ModelFramework.get_sparse_fields(e._matrix)),nz)
    e._compact=b;e._full=copy(b);e._vs=vs;e._applied=zeros(length(e._applied))
    e._epoch=epoch;e._state=:compact
    e
end
function integrated_divergence(c,flux)
    g=c._geometry._data
    result=zeros(length(g.gathers))
    for i in eachindex(result), (f,sign) in g.gathers[i]
        result[i]+=sign*flux[f]
    end
    result
end
function input_snapshot(input)
    (p=copy(input.p),q=input.q===nothing ? nothing : copy(input.q),
     mobility=input.mobility===nothing ? nothing : copy(input.mobility),source=copy(input.source),
     boundaries=[(name=first(p),kind=Symbol(nameof(typeof(last(p)))),
       values=last(p) isa FixedScalarValues ? copy(last(p).values) : nothing) for p in input.boundaries])
end
function ncc_snapshot(e::ScalarTransport)
    nz,col,ptr=ModelFramework.get_sparse_fields(e._matrix)
    (context=ncc_snapshot(e._context),form=e._form,
     scheme=e._scheme===nothing ? nothing : Symbol(nameof(typeof(e._scheme))),
     inputs=input_snapshot(e._input),rowptr=copy(ptr),colval=copy(col),nzval=copy(nz),
     compact_source=copy(e._compact),full_source=copy(e._full),volume_source=copy(e._vs),
     applied_correction=copy(e._applied),epoch=e._epoch,state=e._state,
     buffers_current=e._state!==:unassembled)
end
