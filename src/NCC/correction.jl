# SPDX-License-Identifier: MPL-2.0
struct TraceGradient <: NCCHandle
    _equation::ScalarTransport
    _epoch::UInt64
    _trace::Vector{Float64}
    _gradient::Vector{Vec3}
    TraceGradient(e,t,g,::Val{:owned})=new(e,e._epoch,t,g)
end
struct ExplicitCorrection <: NCCHandle
    _equation::ScalarTransport
    _epoch::UInt64
    _values::Vector{Float64}
    _divergence::Vector{Float64}
    ExplicitCorrection(e,v,d,::Val{:owned})=new(e,e._epoch,v,d)
end
function current_buffer(e,b)
    require_ncc(b._equation._identity === e._identity, "foreign_equation")
    require_ncc(b._epoch==e._epoch, "stale_buffer")
end
function ncc_trace_gradient(e::ScalarTransport)
    require_ncc(e._state===:compact, "trace_requires_compact")
    g=e._context._geometry._data;input=e._input
    trace=zeros(length(g.input.faces))
    for (i,f) in enumerate(g.input.faces)
        trace[i]=f.neighbour===nothing ? last(physical_trace(input,g,f)) :
            g.weights[i]*input.p[f.owner]+(1-g.weights[i])*input.p[f.neighbour]
    end
    grad=fill(zero(Vec3),length(g.input.cells))
    for c in eachindex(grad)
        for (f,sign) in g.gathers[c]
            grad[c]+=sign*trace[f]*g.input.faces[f].Sf
        end
        grad[c]/=g.input.cells[c].volume
    end
    require_ncc(all(isfinite,trace) && all(finite3,grad), "gradient_overflow")
    TraceGradient(e,trace,grad,Val(:owned))
end
function ncc_correction(e::ScalarTransport,t::TraceGradient)
    require_ncc(e._state===:compact, "correction_requires_compact")
    current_buffer(e,t)
    g=e._context._geometry._data;values=zeros(length(g.input.faces))
    if diffusion_sign(e._form)!=0 && e._context._policy isa FoundationCorrected
        for (i,f) in enumerate(g.input.faces)
            f.neighbour===nothing && continue
            w=g.correction_weights[i]
            grad=w*t._gradient[f.owner]+(1-w)*t._gradient[f.neighbour]
            values[i]=e._input.mobility[i]*g.areas[i]*dot(e._context._corrections[i],grad)
        end
    end
    div=integrated_divergence(e._context,values)
    require_ncc(all(isfinite,values) && all(isfinite,div), "correction_overflow")
    ExplicitCorrection(e,values,div,Val(:owned))
end
function ncc_complete!(e::ScalarTransport,b::ExplicitCorrection)
    require_ncc(e._state===:compact, "completion_requires_compact")
    current_buffer(e,b)
    full=e._compact-diffusion_sign(e._form)*b._divergence
    require_ncc(all(isfinite,full), "completion_overflow")
    e._full=full;e._applied=copy(b._values);e._state=:complete
    e
end
function applied_buffer(e,b)
    require_ncc(e._state===:complete, "flux_requires_complete")
    if b===nothing
        require_ncc(diffusion_sign(e._form)==0 || e._context._policy isa FoundationUncorrected, "correction_required")
    else
        require_ncc(b isa ExplicitCorrection, "correction_handle")
        current_buffer(e,b)
        require_ncc(isequal(e._applied,b._values), "unapplied_correction")
    end
end
ncc_snapshot(t::TraceGradient)=(epoch=t._epoch,trace=copy(t._trace),gradient=copy(t._gradient))
ncc_snapshot(b::ExplicitCorrection)=(epoch=b._epoch,values=copy(b._values),divergence=copy(b._divergence))
