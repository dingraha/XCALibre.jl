# SPDX-License-Identifier: MPL-2.0
# Actual native fields, CSR and runtime addressing; no facade matrix injection.
function native_inputs(s::NativeScalarStorage)
    m=s._mesh;o=s._operands
    p=native_field_values(o.p,m,length(m.cells),"p")
    source=native_field_values(o.source,m,length(m.cells),"source")
    q=o.q===nothing ? nothing : native_field_values(o.q,m,length(m.faces),"q")
    mobility=o.mobility===nothing ? nothing : native_field_values(o.mobility,m,length(m.faces),"mobility")
    g=m._binding._context._geometry._data; order=m._binding._native_to_input
    for (i,j) in enumerate(order)
        partner=g.paired[j]
        if partner!=0
            k=m._binding._input_to_native[partner]
            q===nothing || require_ncc(abs(q[i]+q[k])<=1e-12*max(1.0,abs(q[i]),abs(q[k])),"native_q_pair")
            mobility===nothing || require_ncc(abs(mobility[i]-mobility[k])<=1e-12*max(1.0,abs(mobility[i]),abs(mobility[k])),"native_mobility_pair")
        end
    end
    mobility===nothing || require_ncc(all(x->x>0,mobility),"native_mobility_positive")
    (p=p,q=q,mobility=mobility,source=source,boundaries=native_boundary_snapshot(m,s._boundaries))
end
function native_current(e::NativeScalarEquation)
    s=e.equation
    require_ncc(e.model===s._model,"native_foreign_model")
    require_ncc(s._input!==nothing && isequal(native_inputs(s),s._input),"native_changed_operands")
    s
end
function native_buffer(e::NativeScalarEquation,h)
    native_current(e)
    require_ncc(h._equation===e,"native_foreign_equation")
    require_ncc(h._epoch==e.equation._epoch,"native_stale_buffer")
end
function native_structure(m)
    r=m._binding._runtime
    i,j,_=ModelFramework.sparse_matrix_connectivity(r)
    g=m._binding._context._geometry._data
    for k in m._binding._native_to_input
        f=g.input.faces[k]
        if f.key.class===:boundary && f.neighbour!==nothing
            push!(i,f.owner);push!(j,f.neighbour)
        end
    end
    require_ncc(length(j)+1<=typemax(Int32),"native_CSR_capacity")
    A=ModelFramework._build_A(CPU(),i,j,zeros(length(j)),length(m.cells))
    nz,col,ptr=ModelFramework.get_sparse_fields(A)
    require_ncc(eltype(col)===Int32 && eltype(ptr)===Int32,"native_CSR_index_type")
    expected=Set(zip(i,j));actual=Set{Tuple{Int32,Int32}}()
    for row in eachindex(m.cells)
        columns=col[ptr[row]:ptr[row+1]-1]
        require_ncc(issorted(columns) && allunique(columns),"native_CSR_sorted_unique")
        for c in columns;push!(actual,(Int32(row),c));end
    end
    require_ncc(actual==expected && length(nz)==length(actual),"native_CSR_structure")
    diag,face=ModelFramework.nz_index_maps(r,A,CPU())
    require_ncc(all(x->x>0,diag) && all(x->x>0,face),"native_CSR_slots")
    n=length(m.cells)
    ModelFramework.ScalarEquation(A,ModelFramework._build_opA(A),zeros(n),zeros(n),zeros(n),diag,face)
end
function ncc_native_equation(m::StationaryNCCMesh,p::Fields.ScalarField;
    form,convection=nothing,q=nothing,mobility=nothing,source,boundaries)
    require_ncc(form in (:convection,:positive_diffusion,:negative_diffusion,:convection_minus_diffusion),"native_form")
    require_ncc(has_convection(form) ? convection isa Union{Discretise.Linear,Discretise.Upwind} : convection===nothing,"native_scheme")
    require_ncc(has_convection(form) ? q isa Fields.FaceScalarField : q===nothing,"native_q_form")
    require_ncc(diffusion_sign(form)!=0 ? mobility isa Fields.FaceScalarField : mobility===nothing,"native_mobility_form")
    require_ncc(source isa Fields.ScalarField,"native_source_field")
    bcs=native_copy_boundaries(m,boundaries)
    s=NativeScalarStorage(native_structure(m),m,form,convection,(p=p,q=q,mobility=mobility,source=source),bcs,Val(:owned))
    input=native_inputs(s)
    terms=ModelFramework.Operator[]
    has_convection(form) && push!(terms,ModelFramework.Operator(q,p,1,ModelFramework.Divergence{typeof(convection)}()))
    sigma=diffusion_sign(form)
    sigma!=0 && push!(terms,ModelFramework.Operator(mobility,p,sigma,ModelFramework.Laplacian{Discretise.Linear}()))
    tt=Tuple(terms);src=(ModelFramework.Src(source,1),)
    model=ModelFramework.Model{length(tt),1}(tt,src)
    s._model=model
    # Validate all derived compact quantities before returning a native equation.
    nz,b,vs=native_internal_terms(s,input)
    native_boundary_terms(s,input,nz,b)
    ModelFramework.ModelEquation(ModelFramework.ScalarModel(),model,s,(),())
end
function native_add!(s,nz,row,column,value)
    _,col,ptr=ModelFramework.get_sparse_fields(s.A)
    slot=ModelFramework.spindex(ptr,col,row,column)
    require_ncc(slot>0,"native_missing_slot")
    nz[slot]+=value
end
function native_link_coefficients(s,input,fid)
    m=s._mesh
    q=input.q===nothing ? 0.0 : input.q[fid]
    k=input.mobility===nothing ? 0.0 : input.mobility[fid]*m.face_gDiff[fid]
    require_ncc(isfinite(k),"native_coefficient_overflow")
    w=m.face_weight[fid];a=0.0;b=0.0
    for term in s._model.terms
        if term.type isa ModelFramework.Divergence
            up=term.type isa ModelFramework.Divergence{Discretise.Upwind}
            a+=term.sign*(up ? max(q,0.0) : q*w)
            b+=term.sign*(up ? min(q,0.0) : q*(1-w))
        elseif term.type isa ModelFramework.Laplacian
            a-=term.sign*k;b+=term.sign*k
        else
            unsupported("native term")
        end
    end
    a,b
end
function native_internal_terms(s,input)
    m=s._mesh; nz=zeros(length(first(ModelFramework.get_sparse_fields(s.A))))
    vs=zeros(length(m.cells))
    for term in s._model.sources
        require_ncc(term.field===s._operands.source && term.sign==1,"native_source_model")
        vs .+= term.sign .* m.cell_volume .* input.source
    end
    b=copy(vs)
    # Each native internal incidence owns one row; duplicates share additive CSR slots.
    for row in eachindex(m.cells), incidence in m.cell_faces_range[row]
        fid=m.cell_faces[incidence];neighbour=m.cell_neighbours[incidence]
        a,c=native_link_coefficients(s,input,fid)
        if m.cell_nsign[incidence]==1
            native_add!(s,nz,Int32(row),Int32(row),a);native_add!(s,nz,Int32(row),neighbour,c)
        else
            native_add!(s,nz,Int32(row),Int32(row),-c);native_add!(s,nz,Int32(row),neighbour,-a)
        end
    end
    require_ncc(all(isfinite,nz) && all(isfinite,b) && all(isfinite,vs),"native_internal_overflow")
    nz,b,vs
end
function native_physical(input,m,fid)
    key=m._binding._context._geometry._data.input.faces[m._binding._native_to_input[fid]].key
    record=input.boundaries[key.patch]
    fixed=record.kind===:fixed
    fixed, fixed ? record.values[key.local_id] : input.p[m.boundary_cellsID[fid]]
end
function native_boundary_terms(s,input,nz,b)
    m=s._mesh;g=m._binding._context._geometry._data
    for owner_group in eachindex(m.boundary_cells), index in m.boundary_cell_faces_range[owner_group]
        fid=m.boundary_cell_faces[index];owner=m.boundary_cellsID[fid]
        f=g.input.faces[m._binding._native_to_input[fid]]
        if f.neighbour!==nothing
            a,c=native_link_coefficients(s,input,fid)
            native_add!(s,nz,owner,owner,a);native_add!(s,nz,owner,f.neighbour,c)
        else
            fixed,pb=native_physical(input,m,fid)
            q=input.q===nothing ? 0.0 : input.q[fid]
            k=input.mobility===nothing ? 0.0 : input.mobility[fid]*m.face_gDiff[fid]
            require_ncc(isfinite(k),"native_coefficient_overflow")
            sigma=diffusion_sign(s._form)
            if fixed
                native_add!(s,nz,owner,owner,-sigma*k);b[owner]-=(q+sigma*k)*pb
            else
                native_add!(s,nz,owner,owner,q)
            end
        end
    end
    require_ncc(all(isfinite,nz) && all(isfinite,b),"native_boundary_overflow")
    nz,b
end
function Discretise.discretise!(e::NativeScalarEquation,prev,config;
    rho_prev=e.model.terms[1].flux,time=nothing)
    hardware=native_config(config);s=e.equation
    require_ncc(e.model===s._model && prev===s._operands.p,"native_model_operand")
    require_ncc(time===nothing && rho_prev===e.model.terms[1].flux,"native_no_history")
    require_ncc(s._epoch<typemax(UInt64),"native_epoch_overflow")
    input=native_inputs(s)
    nz,b,vs=native_internal_terms(s,input)
    # Preflight full boundary assembly too: no partially admissible model is published.
    native_boundary_terms(s,input,copy(nz),copy(b))
    copyto!(first(ModelFramework.get_sparse_fields(s.A)),nz);copyto!(s.b,b)
    s._compact=copy(b);s._vs=vs;s._applied=zeros(length(s._applied))
    s._input=input;s._hardware=hardware;s._epoch+=UInt64(1);s._state=:internal
    e
end
function Discretise.apply_boundary_conditions!(e::NativeScalarEquation,bcs,component,time,config)
    hardware=native_config(config);s=native_current(e)
    require_ncc(isequal(hardware,s._hardware),"native_changed_config")
    require_ncc(s._state===:internal && component===nothing && time===nothing,"native_boundary_stage")
    require_ncc(isequal(native_boundary_snapshot(s._mesh,bcs),s._input.boundaries),"native_foreign_boundaries")
    nz=copy(first(ModelFramework.get_sparse_fields(s.A)));b=copy(s.b)
    native_boundary_terms(s,s._input,nz,b)
    copyto!(first(ModelFramework.get_sparse_fields(s.A)),nz);copyto!(s.b,b)
    s._compact=copy(b);s._state=:compact
    e
end
function ncc_native_complete!(e::NativeScalarEquation,h::NativeCorrection)
    s=native_current(e);native_buffer(e,h)
    require_ncc(s._state===:compact,"native_completion_stage")
    full=s._compact-diffusion_sign(s._form)*h._divergence
    require_ncc(all(isfinite,full),"native_completion_overflow")
    copyto!(s.b,full);s._applied=copy(h._values);s._state=:complete
    e
end
function native_applied(e,h)
    s=native_current(e)
    require_ncc(s._state===:complete,"native_flux_stage")
    if h===nothing
        require_ncc(diffusion_sign(s._form)==0 || s._mesh._binding._context._policy isa FoundationUncorrected,"native_correction_required")
    else
        require_ncc(h isa NativeCorrection,"native_correction_handle");native_buffer(e,h)
        require_ncc(isequal(h._values,s._applied),"native_unapplied_correction")
    end
    s
end
function ncc_native_flux(e::NativeScalarEquation,h=nothing)
    s=native_applied(e,h);m=s._mesh;input=s._input;g=m._binding._context._geometry._data
    v=zeros(length(m.faces));sigma=diffusion_sign(s._form)
    for fid in eachindex(v)
        f=g.input.faces[m._binding._native_to_input[fid]];po=input.p[f.owner]
        q=input.q===nothing ? 0.0 : input.q[fid]
        k=input.mobility===nothing ? 0.0 : input.mobility[fid]*m.face_gDiff[fid]
        if f.neighbour===nothing
            fixed,pb=native_physical(input,m,fid)
            v[fid]=q*pb+(fixed ? sigma*k*(pb-po) : 0.0)
        else
            pn=input.p[f.neighbour];w=m.face_weight[fid]
            fc=s._scheme isa Discretise.Upwind ? max(q,0.0)*po+min(q,0.0)*pn : q*(w*po+(1-w)*pn)
            v[fid]=fc+sigma*(k*(pn-po)+s._applied[fid])
        end
    end
    d=native_integrated_div(m,v)
    require_ncc(all(isfinite,v) && all(isfinite,d),"native_flux_overflow")
    NativeTransportFlux(e,s._epoch,v,d)
end
function ncc_native_pressure_component(e::NativeScalarEquation,h=nothing)
    s=native_current(e)
    require_ncc(s._form===:negative_diffusion && all(iszero,s._input.source),"native_pressure_form_source")
    f=ncc_native_flux(e,h)
    NativePressureComponent(e,s._epoch,copy(s._input.p),copy(f._values))
end
function ncc_native_recover(h::NativePressureComponent;predictor,previous)
    e=h._equation;native_buffer(e,h);s=e.equation;m=s._mesh
    require_ncc(s._state===:complete,"native_component_stage")
    require_ncc(predictor isa Fields.FaceScalarField && previous isa Fields.ScalarField,"native_recovery_fields")
    q=native_field_values(predictor,m,length(m.faces),"predictor")
    p=native_field_values(previous,m,length(m.cells),"previous")
    require_ncc(isequal(p,h._p),"native_previous_mismatch")
    v=q+h._increment;d=native_integrated_div(m,v)
    require_ncc(all(isfinite,v) && all(isfinite,d),"native_recovery_overflow")
    NativeRecoveredFlux(e,s._epoch,v,copy(h._increment),d)
end
function ncc_snapshot(e::NativeScalarEquation)
    s=e.equation;nz,col,ptr=ModelFramework.get_sparse_fields(s.A)
    (mesh=ncc_snapshot(s._mesh),form=s._form,scheme=s._scheme===nothing ? nothing : Symbol(nameof(typeof(s._scheme))),
     inputs=deepcopy(s._input),hardware=s._hardware,rowptr=copy(ptr),colval=copy(col),nzval=copy(nz),compact_source=copy(s._compact),
     full_source=copy(s.b),volume_source=copy(s._vs),applied_correction=copy(s._applied),
     diag_nz=copy(s.diag_nz),face_nz=copy(s.face_nz),epoch=s._epoch,state=s._state)
end
ncc_snapshot(h::NativeTraceGradient)=(epoch=h._epoch,trace=copy(h._trace),gradient=copy(h._gradient))
ncc_snapshot(h::NativeCorrection)=(epoch=h._epoch,values=copy(h._values),divergence=copy(h._divergence))
ncc_snapshot(h::NativePressureComponent)=(epoch=h._epoch,scalar=copy(h._p),increment=copy(h._increment))
function ncc_snapshot(h::Union{NativeTransportFlux,NativeRecoveredFlux})
    m=h._equation.equation._mesh
    data=(epoch=h._epoch,keys=[m._binding._context._geometry._data.input.faces[j].key for j in m._binding._native_to_input],
        values=copy(h._values),divergence=copy(h._divergence),face_field=Fields.FaceScalarField(copy(h._values),m))
    h isa NativeRecoveredFlux ? merge(data,(increment=copy(h._increment),)) : data
end
