# SPDX-License-Identifier: MPL-2.0
using XCALibre, StaticArrays, LinearAlgebra, Test
const NO = XCALibre.NCC
const OV = SVector{3,Float64}
const OPERATOR_MARKERS = Tuple[]
const OP_G = (:ordinary_skew,:duplicate_skew,:translated_skew,:rewired_skew,:residual_tiny,:empty_error)
const OP_P = (NO.FoundationUncorrected(),NO.FoundationCorrected())
const OP_Q = (:proportional,:distinct_signed)
const OP_T = (:convection_Linear,:convection_Upwind,:positive_diffusion,:negative_diffusion,:mixed_Linear,:mixed_Upwind)
opmark(group,ids...)=push!(OPERATOR_MARKERS,(group,ids...))

# New algebra-only fixtures, not closed meshes. Internal12 and coupled12 coincide;
# duplicate adds coupled13; rewire replaces13 by23, preserving seven directed CSR slots.
# Nonzero tangential centre displacement, unequal positive projected face distances.
function operator_geometry(name; volume_scale=1.0, area_scale=1.0)
    centres=[OV(0.,0.,0.),OV(1.,.2,.1),OV(2.,-.1,.3)]
    volumes=[1.1,.8,1.3]*volume_scale
    cells=[NO.NCCCellInput(Int32(i),centres[i],volumes[i]) for i in 1:3]
    ik=NO.NCCFaceKey(:internal,nothing,Int32(1))
    faces=[NO.NCCFaceInput(ik,Int32(1),Int32(2),area_scale*OV(1.,0.,0.),.35centres[2],Int32(1))]
    patches=NO.NCCPatchInput[];pairs=Tuple{NO.NCCFaceKey,NO.NCCFaceKey}[]
    key(p,i)=NO.NCCFaceKey(:boundary,Int32(p),Int32(i))
    if name!==:ordinary_skew
        links=name in (:duplicate_skew,:rewired_skew) ? [(1,2),(name===:rewired_skew ? 2 : 1,3)] : [(1,3)]
        shift=name===:translated_skew ? OV(.2,.1,0.) : zero(OV)
        for (i,(o,n)) in enumerate(links)
            cf=centres[o]+.4*(centres[n]+shift-centres[o]);sf=area_scale*OV(.6,0.,0.)
            a,b=key(1,i),key(2,i)
            push!(faces,NO.NCCFaceInput(a,Int32(o),Int32(n),sf,cf,Int32(1)))
            push!(faces,NO.NCCFaceInput(b,Int32(n),Int32(o),-sf,cf-shift,Int32(1)))
            push!(pairs,(a,b))
        end
        push!(patches,NO.NCCPatchInput(Int32(1),:a,:coupled,[key(1,i) for i in eachindex(links)],Int32(2),shift))
        push!(patches,NO.NCCPatchInput(Int32(2),:b,:coupled,[key(2,i) for i in eachindex(links)],Int32(1),-shift))
    end
    for (pname,kind,o,sf,offset) in ((:fixed,name===:residual_tiny ? :residual : :physical,1,OV(-1.,0.,0.),OV(-.4,.1,0.)),(:zero,:physical,2,OV(0.,1.,0.),OV(0.,.5,.1)))
        pid=Int32(length(patches)+1);k=key(pid,1)
        a=name===:residual_tiny && pname===:fixed ? 1e-18 : 1.
        push!(faces,NO.NCCFaceInput(k,Int32(o),nothing,area_scale*a*sf,centres[o]+offset,Int32(1)))
        push!(patches,NO.NCCPatchInput(pid,pname,kind,[k],nothing,zero(OV)))
    end
    if name===:empty_error
        for (pname,kind) in ((:front,:empty),(:error,:error))
            push!(patches,NO.NCCPatchInput(Int32(length(patches)+1),pname,kind,NO.NCCFaceKey[],nothing,zero(OV)))
        end
    end
    NO.FrozenNCCGeometry(NO.NCCGeometryInput(;cells,faces,patches,pairs))
end
function operator_data(c,pattern,step=1;source_zero=false,large_scalar=false)
    g=NO.ncc_snapshot(c).geometry;nf=length(g.input.faces)
    p=large_scalar ? fill(1e308,3) : [1.3+.1step,-.7+.2step,.4-.05step]
    mob=[1.2+.03i for i in 1:nf];q=zeros(nf)
    for (i,f) in enumerate(g.input.faces)
        if g.paired[i]!=0
            j=g.paired[i];mob[i]=1.2+.03min(i,j)
            q[i]=(i<j ? 1. : -1.)*(pattern===:proportional ? .3mob[i] : (iseven(min(i,j)) ? -.4 : .7))
        else
            q[i]=pattern===:proportional ? .3mob[i] : .15*(-1.)^i
        end
    end
    physical=[x.name => (x.kind in (:empty,:error) ? NO.NoRuntimeFaces() : x.name===:fixed ? NO.FixedScalarValues([large_scalar ? 1e308 : 1.7+.1step]) : NO.ZeroGradient()) for x in g.input.patches if x.kind!==:coupled]
    (p=NO.ncc_scalar(c;values=p),q=NO.ncc_face_flux(c;values=q),mobility=NO.ncc_mobility(c;values=mob),source=source_zero ? zeros(3) : [.11,-.2,.07]*step,boundaries=NO.ncc_boundaries(c;physical))
end
function make_operator(c,t,data)
    conv=t in (:convection_Linear,:convection_Upwind,:mixed_Linear,:mixed_Upwind)
    form=t in (:mixed_Linear,:mixed_Upwind) ? :convection_minus_diffusion : conv ? :convection : t
    scheme=conv ? (t in (:convection_Upwind,:mixed_Upwind) ? XCALibre.Upwind() : XCALibre.Linear()) : nothing
    NO.ncc_equation(c,data.p;form,convection=scheme,q=conv ? data.q : nothing,mobility=form===:convection ? nothing : data.mobility,source=data.source,boundaries=data.boundaries)
end
function update_operator(e,t,d)
    NO.ncc_update!(e;p=d.p,q=t in (:convection_Linear,:convection_Upwind,:mixed_Linear,:mixed_Upwind) ? d.q : nothing,mobility=t in (:convection_Linear,:convection_Upwind) ? nothing : d.mobility,source=d.source,boundaries=d.boundaries)
end
function complete_operator(e)
    NO.ncc_assemble!(e);before=NO.ncc_snapshot(e)
    tg=NO.ncc_trace_gradient(e);b=NO.ncc_correction(e,tg)
    @test isequal(before,NO.ncc_snapshot(e))
    NO.ncc_complete!(e,b)
    tg,b,NO.ncc_flux(e,b)
end
function dense_operator(e)
    s=NO.ncc_snapshot(e);a=zeros(3,3)
    for i in 1:3,k in s.rowptr[i]:s.rowptr[i+1]-1;a[i,s.colval[k]]=s.nzval[k];end
    a
end
opclose(a,b)=@test maximum(abs,a-b)<=1e-12*max(1.,maximum(abs,a),maximum(abs,b))

# Independent small dense FV oracle: reconstruct distances/weights from input records,
# not production factors, correction helpers, traces or slot mappings.
function operator_oracle(c,t,d)
    s=NO.ncc_snapshot(c).geometry.input
    p=NO.ncc_snapshot(d.p).values;q=NO.ncc_snapshot(d.q).values;m=NO.ncc_snapshot(d.mobility).values
    corrected=c._policy isa NO.FoundationCorrected
    conv=t in (:convection_Linear,:convection_Upwind,:mixed_Linear,:mixed_Upwind)
    sigma=t===:positive_diffusion ? 1 : t in (:negative_diffusion,:mixed_Linear,:mixed_Upwind) ? -1 : 0
    up=t in (:convection_Upwind,:mixed_Upwind)
    A=zeros(3,3);b=[s.cells[i].volume*d.source[i] for i in 1:3];tr=zeros(length(s.faces))
    coeff=zeros(length(s.faces));wc=zeros(length(s.faces));cv=fill(zero(OV),length(s.faces))
    function addface(i,f,alpha,beta,constant)
        A[f.owner,f.owner]+=alpha
        f.neighbour!==nothing && (A[f.owner,f.neighbour]+=beta)
        b[f.owner]-=constant
        if f.key.class===:internal
            A[f.neighbour,f.owner]-=alpha;A[f.neighbour,f.neighbour]-=beta;b[f.neighbour]+=constant
        end
    end
    for (i,f) in enumerate(s.faces)
        sf=f.Sf;ar=norm(sf);n=sf/ar;co=s.cells[f.owner].centre
        if f.neighbour!==nothing
            shift=f.key.class===:internal ? zero(OV) : s.patches[f.key.patch].translation
            delta=s.cells[f.neighbour].centre+shift-co
            near=dot(n,f.Cf-co);far=dot(n,s.cells[f.neighbour].centre+shift-f.Cf)
            w=far/(near+far);wc[i]=far/dot(n,delta)
            den=corrected ? max(dot(n,delta),.05*norm(delta)) : norm(delta)
            coeff[i]=m[i]*ar/den;cv[i]=corrected ? n-delta/den : zero(OV)
            tr[i]=w*p[f.owner]+(1-w)*p[f.neighbour]
            alpha=conv ? (up ? max(q[i],0.) : q[i]*w) : 0.
            beta=conv ? (up ? min(q[i],0.) : q[i]*(1-w)) : 0.
            addface(i,f,alpha-sigma*coeff[i],beta+sigma*coeff[i],0.)
        else
            name=s.patches[f.key.patch].name
            fixed=name===:fixed;pb=fixed ? NO.ncc_snapshot(d.boundaries).physical[findfirst(x->x.name===name,NO.ncc_snapshot(d.boundaries).physical)].values[1] : p[f.owner]
            tr[i]=pb;coeff[i]=m[i]*ar/norm(f.Cf-co)
            addface(i,f,fixed ? -sigma*coeff[i] : conv ? q[i] : 0.,0.,fixed ? ((conv ? q[i] : 0.)+sigma*coeff[i])*pb : 0.)
        end
    end
    grad=fill(zero(OV),3)
    for (i,f) in enumerate(s.faces)
        grad[f.owner]+=tr[i]*f.Sf
        f.key.class===:internal && (grad[f.neighbour]-=tr[i]*f.Sf)
    end
    for i in 1:3;grad[i]/=s.cells[i].volume;end
    E=zeros(length(s.faces));flux=zeros(length(s.faces));divE=zeros(3);div=zeros(3)
    for (i,f) in enumerate(s.faces)
        if f.neighbour!==nothing
            sigma!=0 && (E[i]=m[i]*norm(f.Sf)*dot(cv[i],wc[i]*grad[f.owner]+(1-wc[i])*grad[f.neighbour]))
            fc=conv ? (up ? max(q[i],0.)*p[f.owner]+min(q[i],0.)*p[f.neighbour] : q[i]*tr[i]) : 0.
            flux[i]=fc+sigma*(coeff[i]*(p[f.neighbour]-p[f.owner])+E[i])
        else
            fixed=s.patches[f.key.patch].name===:fixed
            flux[i]=(conv ? q[i]*tr[i] : 0.)+(fixed ? sigma*coeff[i]*(tr[i]-p[f.owner]) : 0.)
        end
        divE[f.owner]+=E[i];div[f.owner]+=flux[i]
        if f.key.class===:internal;divE[f.neighbour]-=E[i];div[f.neighbour]-=flux[i];end
    end
    (A=A,compact=b,full=b-sigma*divE,trace=tr,gradient=grad,E=E,flux=flux,div=div)
end
function compare_operator(e,c,t,d,tg,b,f)
    o=operator_oracle(c,t,d);s=NO.ncc_snapshot(e)
    opclose(dense_operator(e),o.A);opclose(s.compact_source,o.compact);opclose(s.full_source,o.full)
    opclose(NO.ncc_snapshot(tg).trace,o.trace)
    opclose(reduce(vcat,collect.(NO.ncc_snapshot(tg).gradient)),reduce(vcat,collect.(o.gradient)))
    opclose(NO.ncc_snapshot(b).values,o.E);opclose(NO.ncc_snapshot(f).values,o.flux)
    opclose(o.A*NO.ncc_snapshot(d.p).values-o.full+s.volume_source,o.div)
    opclose(dense_operator(e)*NO.ncc_snapshot(d.p).values-s.full_source+s.volume_source,NO.ncc_snapshot(f).divergence)
    @test all(isfinite,s.nzval)
end

const OP_R = (:foreign_build,:foreign_policy,:foreign_equation,:source_length,:source_nonfinite,
    :missing_q,:irrelevant_q,:missing_mobility,:irrelevant_mobility,:unsupported_scheme,
    :paired_mobility,:stale_trace,:stale_correction,:stale_component,:duplicate_complete,
    :unassembled_trace,:unassembled_flux,:incomplete_flux,:mixed_pressure,:positive_pressure,
    :source_pressure,:wrong_predictor,:wrong_previous,:previous_mismatch,:update_invalid,:update_wrong_form)
const OP_V = (:caller_source,:equation_snapshot,:trace_snapshot,:correction_snapshot,
    :flux_snapshot,:component_snapshot,:recovery_snapshot)
function expected_operator_markers()
    result=Set{Tuple}()
    ps=Symbol.(nameof.(typeof.(OP_P)))
    for g in OP_G,p in ps,q in OP_Q,t in OP_T
        push!(result,(:path,g,p,q,t))
        for step in 1:3;push!(result,(:step,g,p,q,t,step));end
    end
    for g in OP_G,p in ps,q in OP_Q,step in 1:3
        push!(result,(:pressure,g,p,q,step))
        for scheme in (:Linear,:Upwind);push!(result,(:composition,g,p,q,scheme,step));end
    end
    for p in ps,q in OP_Q,t in OP_T,group in (:rebuild,:old_preserved)
        push!(result,(group,p,q,t))
    end
    for p in ps,r in OP_R;push!(result,(:reject,p,r));end
    for p in ps,v in OP_V;push!(result,(:ownership,p,v));end
    for p in ps,o in (:coefficient_overflow,:source_overflow,:gradient_overflow,:recovery_overflow)
        push!(result,(:overflow,p,o))
    end
    push!(result,(:overflow,:FoundationCorrected,:completion_overflow))
    push!(result,(:completion_zero_identity,:FoundationUncorrected))
    result
end
function rejected_unchanged(call,e,category)
    before=NO.ncc_snapshot(e)
    error=try
        call();nothing
    catch ex
        ex
    end
    @test error isa ArgumentError
    @test error isa ArgumentError && occursin(category,sprint(showerror,error))
    @test isequal(before,NO.ncc_snapshot(e))
end
function operator_rejections!()
    for policy in OP_P,r in OP_R
        c=NO.NCCScalarContext(operator_geometry(:duplicate_skew);policy)
        d=operator_data(c,:distinct_signed;source_zero=true)
        e=make_operator(c,:negative_diffusion,d)
        ctor(;p=d.p,form=:negative_diffusion,convection=nothing,q=nothing,mobility=d.mobility,source=d.source,boundaries=d.boundaries)=
            NO.ncc_equation(c,p;form,convection,q,mobility,source,boundaries)
        category="";call=()->nothing
        if r in (:foreign_build,:foreign_policy)
            pc=r===:foreign_policy ? (policy isa NO.FoundationCorrected ? NO.FoundationUncorrected() : NO.FoundationCorrected()) : policy
            other=NO.NCCScalarContext(operator_geometry(:duplicate_skew);policy=pc)
            foreign=operator_data(other,:distinct_signed;source_zero=true)
            call=()->ctor(p=foreign.p);category="foreign_context"
        elseif r===:foreign_equation
            other=make_operator(c,:negative_diffusion,d);NO.ncc_assemble!(other)
            tg=NO.ncc_trace_gradient(other);NO.ncc_assemble!(e)
            call=()->NO.ncc_correction(e,tg);category="foreign_equation"
        elseif r===:source_length
            call=()->ctor(source=zeros(2));category="source length"
        elseif r===:source_nonfinite
            call=()->ctor(source=[NaN,0.,0.]);category="source nonfinite"
        elseif r===:missing_q
            call=()->ctor(form=:convection,convection=XCALibre.Linear(),mobility=nothing);category="missing_q"
        elseif r===:irrelevant_q
            call=()->ctor(q=d.q);category="irrelevant_q"
        elseif r===:missing_mobility
            call=()->ctor(mobility=nothing);category="missing_mobility"
        elseif r===:irrelevant_mobility
            call=()->ctor(form=:convection,convection=XCALibre.Linear(),q=d.q);category="irrelevant_mobility"
        elseif r===:unsupported_scheme
            call=()->ctor(form=:convection,convection=:Linear,q=d.q,mobility=nothing);category="unsupported_scheme"
        elseif r===:paired_mobility
            values=copy(NO.ncc_snapshot(d.mobility).values);values[2]*=2
            bad=NO.ncc_mobility(c;values)
            call=()->ctor(mobility=bad);category="paired_mobility"
        elseif r in (:stale_trace,:stale_correction,:stale_component)
            tg,b,f=complete_operator(e);component=NO.ncc_pressure_component(e,b)
            update_operator(e,:negative_diffusion,d);NO.ncc_assemble!(e)
            if r===:stale_trace
                call=()->NO.ncc_correction(e,tg)
            elseif r===:stale_correction
                call=()->NO.ncc_complete!(e,b)
            else
                call=()->NO.ncc_recover(component;predictor=d.q,previous=d.p)
            end
            category="stale_buffer"
        elseif r===:duplicate_complete
            _,b,_=complete_operator(e);call=()->NO.ncc_complete!(e,b);category="completion_requires_compact"
        elseif r===:unassembled_trace
            call=()->NO.ncc_trace_gradient(e);category="trace_requires_compact"
        elseif r===:unassembled_flux
            call=()->NO.ncc_flux(e);category="flux_requires_complete"
        elseif r===:incomplete_flux
            NO.ncc_assemble!(e);call=()->NO.ncc_flux(e);category="flux_requires_complete"
        elseif r in (:mixed_pressure,:positive_pressure,:source_pressure)
            t=r===:mixed_pressure ? :mixed_Linear : r===:positive_pressure ? :positive_diffusion : :negative_diffusion
            dd=r===:source_pressure ? operator_data(c,:distinct_signed) : d
            e=make_operator(c,t,dd);_,b,_=complete_operator(e)
            call=()->NO.ncc_pressure_component(e,b)
            category=r===:source_pressure ? "pressure_requires_zero_source" : "pressure_requires_negative_diffusion"
        elseif r in (:wrong_predictor,:wrong_previous,:previous_mismatch)
            _,b,_=complete_operator(e);component=NO.ncc_pressure_component(e,b)
            predictor=r===:wrong_predictor ? d.p : d.q
            previous=r===:wrong_previous ? d.q : r===:previous_mismatch ? NO.ncc_scalar(c;values=zeros(3)) : d.p
            call=()->NO.ncc_recover(component;predictor,previous)
            category=r===:wrong_predictor ? "predictor_handle" : r===:wrong_previous ? "previous_handle" : "previous_mismatch"
        elseif r===:update_invalid
            complete_operator(e)
            call=()->NO.ncc_update!(e;p=d.p,q=nothing,mobility=d.mobility,source=[Inf,0.,0.],boundaries=d.boundaries)
            category="source nonfinite"
        elseif r===:update_wrong_form
            complete_operator(e)
            # A convection-shaped replacement cannot change the fixed negative-D form.
            call=()->NO.ncc_update!(e;p=d.p,q=d.q,mobility=nothing,source=d.source,boundaries=d.boundaries)
            category="irrelevant_q"
        end
        rejected_unchanged(call,e,category)
        opmark(:reject,Symbol(nameof(typeof(policy))),r)
    end
end
function altered_snapshot!(s,field)
    values=getproperty(s,field)
    values[1]=values[1]+123.
end
function operator_overflow_ownership!()
    for policy in OP_P
        pname=Symbol(nameof(typeof(policy)))
        # Finite inputs reach the named arithmetic operation, not a preceding schema error.
        c=NO.NCCScalarContext(operator_geometry(:ordinary_skew;area_scale=10.);policy)
        d=operator_data(c,:distinct_signed;source_zero=true);e=make_operator(c,:negative_diffusion,d)
        huge=NO.ncc_mobility(c;values=fill(1e308,length(NO.ncc_snapshot(d.mobility).values)))
        rejected_unchanged(()->NO.ncc_update!(e;p=d.p,q=nothing,mobility=huge,source=d.source,boundaries=d.boundaries),e,"coefficient_overflow")
        opmark(:overflow,pname,:coefficient_overflow)
        c=NO.NCCScalarContext(operator_geometry(:ordinary_skew);policy)
        d=operator_data(c,:distinct_signed;source_zero=true);e=make_operator(c,:negative_diffusion,d)
        rejected_unchanged(()->NO.ncc_update!(e;p=d.p,q=nothing,mobility=d.mobility,source=[0.,0.,1.7e308],boundaries=d.boundaries),e,"source_overflow")
        opmark(:overflow,pname,:source_overflow)
        csmall=NO.NCCScalarContext(operator_geometry(:ordinary_skew;volume_scale=1e-308);policy)
        ds=operator_data(csmall,:distinct_signed;source_zero=true)
        ds=merge(ds,(p=NO.ncc_scalar(csmall;values=[10.,-10.,5.]),))
        es=make_operator(csmall,:negative_diffusion,ds);NO.ncc_assemble!(es)
        rejected_unchanged(()->NO.ncc_trace_gradient(es),es,"gradient_overflow")
        opmark(:overflow,pname,:gradient_overflow)
        # Linked mobility huge, physical mobility ordinary: coefficients remain finite.
        c=NO.NCCScalarContext(operator_geometry(:ordinary_skew);policy)
        d=operator_data(c,:distinct_signed;source_zero=true)
        mv=NO.ncc_snapshot(d.mobility).values;mv[1]=1e308
        d=merge(d,(p=NO.ncc_scalar(c;values=[.5,-.3,.1]),mobility=NO.ncc_mobility(c;values=mv)))
        e=make_operator(c,:negative_diffusion,d);_,b,_=complete_operator(e)
        pc=NO.ncc_pressure_component(e,b)
        q=NO.ncc_snapshot(d.q).values;q[1]=1.7e308
        predictor=NO.ncc_face_flux(c;values=q)
        pred_before=NO.ncc_snapshot(predictor);previous_before=NO.ncc_snapshot(d.p)
        rejected_unchanged(()->NO.ncc_recover(pc;predictor,previous=d.p),e,"recovery_overflow")
        @test isequal(pred_before,NO.ncc_snapshot(predictor))
        @test isequal(previous_before,NO.ncc_snapshot(d.p))
        opmark(:overflow,pname,:recovery_overflow)
        if policy isa NO.FoundationCorrected
            # E_owner is positive from negative neighbour y-gradient; source alone
            # puts owner compact b near Float64 maximum before +div(E) completion.
            source=[1.79e308/1.1,0.,0.]
            e=make_operator(c,:negative_diffusion,merge(d,(source=source,)))
            NO.ncc_assemble!(e);tg=NO.ncc_trace_gradient(e);b=NO.ncc_correction(e,tg)
            @test NO.ncc_snapshot(b).divergence[1]>0.
            rejected_unchanged(()->NO.ncc_complete!(e,b),e,"completion_overflow")
            opmark(:overflow,pname,:completion_overflow)
        else
            e=make_operator(c,:negative_diffusion,d);NO.ncc_assemble!(e)
            tg=NO.ncc_trace_gradient(e);b=NO.ncc_correction(e,tg);before=NO.ncc_snapshot(e)
            @test all(iszero,NO.ncc_snapshot(b).values)
            NO.ncc_complete!(e,b);after=NO.ncc_snapshot(e)
            @test isequal(before.compact_source,after.full_source)
            @test isequal(before.compact_source,after.compact_source)
            @test isequal(before.nzval,after.nzval)
            @test after.state===:complete
            opmark(:completion_zero_identity,pname)
        end
        c=NO.NCCScalarContext(operator_geometry(:duplicate_skew);policy)
        d=operator_data(c,:distinct_signed;source_zero=true)
        for v in OP_V
            e=make_operator(c,:negative_diffusion,d)
            if v===:caller_source
                source=[.1,.2,.3];dd=merge(d,(source=source,));e=make_operator(c,:negative_diffusion,dd)
                before=NO.ncc_snapshot(e);source[1]=100.
                @test isequal(before,NO.ncc_snapshot(e))
            else
                tg,b,f=complete_operator(e);pc=NO.ncc_pressure_component(e,b)
                recovery=NO.ncc_recover(pc;predictor=d.q,previous=d.p)
                handle=v===:equation_snapshot ? e : v===:trace_snapshot ? tg : v===:correction_snapshot ? b : v===:flux_snapshot ? f : v===:component_snapshot ? pc : recovery
                before=NO.ncc_snapshot(handle);snapshot=NO.ncc_snapshot(handle)
                field=v===:equation_snapshot ? :nzval : v===:trace_snapshot ? :trace : v===:component_snapshot ? :increment : :values
                altered_snapshot!(snapshot,field)
                @test !isequal(snapshot,before)
                @test isequal(before,NO.ncc_snapshot(handle))
            end
            opmark(:ownership,pname,v)
        end
    end
end

@testset "NCC scalar operators (unregistered)" begin
    empty!(OPERATOR_MARKERS)
    for name in OP_G,policy in OP_P,pattern in OP_Q,t in OP_T
        c=NO.NCCScalarContext(operator_geometry(name);policy);d=operator_data(c,pattern);e=make_operator(c,t,d)
        tag=(name,Symbol(nameof(typeof(policy))),pattern,t);opmark(:path,tag...)
        for step in 1:3
            if step>1
                old=NO.ncc_snapshot(e);d=operator_data(c,pattern,step);update_operator(e,t,d)
                @test NO.ncc_snapshot(e).state===:unassembled
                @test isequal(old.nzval,NO.ncc_snapshot(e).nzval)
                fill!(first(XCALibre.ModelFramework.get_sparse_fields(e._matrix)),123.)
                fill!(e._compact,456.);fill!(e._full,789.)
            end
            tg,b,f=complete_operator(e);compare_operator(e,c,t,d,tg,b,f)
            opmark(:step,tag...,step)
        end
    end
    for name in OP_G,policy in OP_P,pattern in OP_Q,step in 1:3
        c=NO.NCCScalarContext(operator_geometry(name);policy);d=operator_data(c,pattern,step;source_zero=true)
        e=make_operator(c,:negative_diffusion,d);tg,b,f=complete_operator(e);pc=NO.ncc_pressure_component(e,b)
        pred=NO.ncc_snapshot(d.q);prev=NO.ncc_snapshot(d.p)
        r=NO.ncc_recover(pc;predictor=d.q,previous=d.p);rs=NO.ncc_snapshot(r)
        opclose(rs.increment,NO.ncc_snapshot(f).values);opclose(rs.values,pred.values+rs.increment)
        @test isequal(NO.ncc_snapshot(d.q),pred);@test isequal(NO.ncc_snapshot(d.p),prev)
        opmark(:pressure,name,Symbol(nameof(typeof(policy))),pattern,step)
        for scheme in (:Linear,:Upwind)
            ct=scheme===:Linear ? :convection_Linear : :convection_Upwind
            mt=scheme===:Linear ? :mixed_Linear : :mixed_Upwind
            ce=make_operator(c,ct,d);_,cb,cf=complete_operator(ce)
            md=operator_data(c,pattern,step);me=make_operator(c,mt,md);_,mb,mf=complete_operator(me)
            composed=NO.ncc_recover(pc;predictor=cf,previous=d.p)
            opclose(NO.ncc_snapshot(composed).values,NO.ncc_snapshot(mf).values)
            opmark(:composition,name,Symbol(nameof(typeof(policy))),pattern,scheme,step)
        end
    end
    for policy in OP_P,pattern in OP_Q,t in OP_T
        oldc=NO.NCCScalarContext(operator_geometry(:duplicate_skew);policy);newc=NO.NCCScalarContext(operator_geometry(:rewired_skew);policy)
        od=operator_data(oldc,pattern);nd=operator_data(newc,pattern)
        oe=make_operator(oldc,t,od);ne=make_operator(newc,t,nd);complete_operator(oe);before=NO.ncc_snapshot(oe);complete_operator(ne)
        os=Set(keys(oe._slots));ns=Set(keys(ne._slots))
        @test length(os)==length(ns)==7
        @test os==Set((Int32(i),Int32(j)) for (i,j) in [(1,1),(2,2),(3,3),(1,2),(2,1),(1,3),(3,1)])
        @test ns==Set((Int32(i),Int32(j)) for (i,j) in [(1,1),(2,2),(3,3),(1,2),(2,1),(2,3),(3,2)])
        @test (Int32(1),Int32(3)) in os && !((Int32(1),Int32(3)) in ns)
        @test (Int32(2),Int32(3)) in ns && !((Int32(2),Int32(3)) in os)
        tag=(Symbol(nameof(typeof(policy))),pattern,t);opmark(:rebuild,tag...)
        @test isequal(before,NO.ncc_snapshot(oe));opmark(:old_preserved,tag...)
    end
    operator_rejections!()
    operator_overflow_ownership!()
    @test Set(OPERATOR_MARKERS)==expected_operator_markers()
    @test length(OPERATOR_MARKERS)==length(unique(OPERATOR_MARKERS))==916
end
