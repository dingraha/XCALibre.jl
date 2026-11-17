# SPDX-License-Identifier: MPL-2.0
struct TransportFlux <: NCCHandle
    _context::NCCScalarContext
    _values::Vector{Float64}
    _divergence::Vector{Float64}
    _form::Symbol
    _epoch::UInt64
    TransportFlux(e,v,d,::Val{:owned})=new(e._context,v,d,e._form,e._epoch)
end
struct PressureComponent <: NCCHandle
    _equation::ScalarTransport
    _epoch::UInt64
    _p::Vector{Float64}
    _input::NamedTuple
    _correction::Vector{Float64}
    _increment::Vector{Float64}
    PressureComponent(e,v,::Val{:owned})=new(e,e._epoch,copy(e._input.p),deepcopy(e._input),copy(e._applied),v)
end
struct RecoveredFlux <: NCCHandle
    _context::NCCScalarContext
    _values::Vector{Float64}
    _increment::Vector{Float64}
    _divergence::Vector{Float64}
    RecoveredFlux(c,v,inc,d,::Val{:owned})=new(c,v,inc,d)
end
function ncc_flux(e::ScalarTransport,b=nothing)
    applied_buffer(e,b)
    g=e._context._geometry._data;input=e._input
    values=zeros(length(g.input.faces));sigma=diffusion_sign(e._form)
    for (i,f) in enumerate(g.input.faces)
        po=input.p[f.owner];q=input.q===nothing ? 0.0 : input.q[i]
        k=input.mobility===nothing ? 0.0 : input.mobility[i]*g.areas[i]*e._context._factors[i]
        if f.neighbour===nothing
            fixed,pb=physical_trace(input,g,f)
            values[i]=q*pb+(fixed ? sigma*k*(pb-po) : 0.0)
        else
            pn=input.p[f.neighbour]
            fc=e._scheme isa Discretise.Upwind ? max(q,0.0)*po+min(q,0.0)*pn : q*(g.weights[i]*po+(1-g.weights[i])*pn)
            values[i]=fc+sigma*(k*(pn-po)+e._applied[i])
        end
    end
    div=integrated_divergence(e._context,values)
    require_ncc(all(isfinite,values) && all(isfinite,div), "flux_overflow")
    TransportFlux(e,values,div,Val(:owned))
end
function ncc_pressure_component(e::ScalarTransport,b=nothing)
    require_ncc(e._form===:negative_diffusion, "pressure_requires_negative_diffusion")
    require_ncc(all(iszero,e._input.source), "pressure_requires_zero_source")
    f=ncc_flux(e,b)
    PressureComponent(e,copy(f._values),Val(:owned))
end
function ncc_recover(p::PressureComponent;predictor,previous)
    e=p._equation;current_buffer(e,p)
    require_ncc(e._state===:complete, "stale_component")
    require_ncc(predictor isa Union{FluxValues,TransportFlux,RecoveredFlux}, "predictor_handle")
    require_ncc(predictor._context._identity===e._context._identity, "foreign_context")
    require_ncc(previous isa ScalarValues, "previous_handle");same_context(e._context,previous)
    require_ncc(isequal(previous._values,p._p), "previous_mismatch")
    values=predictor._values+p._increment
    div=integrated_divergence(e._context,values)
    require_ncc(all(isfinite,values) && all(isfinite,div), "recovery_overflow")
    RecoveredFlux(e._context,values,copy(p._increment),div,Val(:owned))
end
function flux_snapshot(c,v,d)
    (keys=[f.key for f in c._geometry._data.input.faces],values=copy(v),divergence=copy(d))
end
ncc_snapshot(f::TransportFlux)=merge(flux_snapshot(f._context,f._values,f._divergence),(form=f._form,epoch=f._epoch))
ncc_snapshot(f::RecoveredFlux)=merge(flux_snapshot(f._context,f._values,f._divergence),(increment=copy(f._increment),))
ncc_snapshot(p::PressureComponent)=(epoch=p._epoch,scalar=copy(p._p),inputs=input_snapshot(p._input),correction=copy(p._correction),increment=copy(p._increment))
