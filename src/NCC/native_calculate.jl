# SPDX-License-Identifier: MPL-2.0
function native_integrated_div(m,v)
    owned_values(v,length(m.faces),"native_integrated_flux")
    result=zeros(length(m.cells))
    for row in eachindex(result), k in m.cell_faces_range[row]
        result[row]+=m.cell_nsign[k]*v[m.cell_faces[k]]
    end
    for j in eachindex(m.boundary_cells), k in m.boundary_cell_faces_range[j]
        result[m.boundary_cells[j]]+=v[m.boundary_cell_faces[k]]
    end
    require_ncc(all(isfinite,result),"native_divergence_overflow")
    result
end
function native_trace(m,p,boundaries=nothing)
    g=m._binding._context._geometry._data
    v=zeros(length(m.faces))
    for (fid,j) in enumerate(m._binding._native_to_input)
        f=g.input.faces[j]
        if f.neighbour!==nothing
            w=m.face_weight[fid];v[fid]=w*p[f.owner]+(1-w)*p[f.neighbour]
        else
            v[fid]=p[f.owner]
            if boundaries!==nothing
                bc=boundaries[f.key.patch]
                bc.kind===:fixed && (v[fid]=bc.values[f.key.local_id])
            end
        end
    end
    require_ncc(all(isfinite,v),"native_trace_overflow")
    v
end
function native_gradient(m,trace)
    # Same cell/internal and boundary-owner addressing as native Gauss kernels.
    result=fill(zero(Vec3),length(m.cells))
    for row in eachindex(result)
        for k in m.cell_faces_range[row]
            fid=m.cell_faces[k]
            result[row]+=m.cell_nsign[k]*trace[fid]*(m.face_area[fid]*m.face_normal[fid])
        end
        result[row]/=m.cell_volume[row]
    end
    for j in eachindex(m.boundary_cells), k in m.boundary_cell_faces_range[j]
        fid=m.boundary_cell_faces[k];owner=m.boundary_cells[j]
        result[owner]+=trace[fid]*(m.face_area[fid]*m.face_normal[fid])/m.cell_volume[owner]
    end
    require_ncc(all(finite3,result),"native_gradient_overflow")
    result
end
function ncc_native_trace_gradient(e::NativeScalarEquation)
    s=native_current(e);require_ncc(s._state===:compact,"native_trace_stage")
    m=s._mesh;config=(hardware=s._hardware,)
    trace=Fields.FaceScalarField(zeros(length(m.faces)),m)
    grad=Calculate.Grad{Discretise.Gauss}(s._operands.p)
    Calculate.grad!(grad,trace,s._operands.p,s._boundaries,nothing,config)
    gradient=Vec3[grad[i] for i in eachindex(m.cells)]
    require_ncc(all(isfinite,trace.values) && all(finite3,gradient),"native_gradient_overflow")
    NativeTraceGradient(e,s._epoch,copy(trace.values),gradient)
end
function ncc_native_correction(e::NativeScalarEquation,t::NativeTraceGradient)
    s=native_current(e);native_buffer(e,t)
    require_ncc(s._state===:compact,"native_correction_stage")
    m=s._mesh;c=m._binding._context;g=c._geometry._data;v=zeros(length(m.faces))
    if diffusion_sign(s._form)!=0 && c._policy isa FoundationCorrected
        for (fid,j) in enumerate(m._binding._native_to_input)
            f=g.input.faces[j];f.neighbour===nothing && continue
            w=g.correction_weights[j]
            grad=w*t._gradient[f.owner]+(1-w)*t._gradient[f.neighbour]
            v[fid]=s._input.mobility[fid]*m.face_area[fid]*dot(c._corrections[j],grad)
        end
    end
    require_ncc(all(isfinite,v),"native_correction_overflow")
    d=native_integrated_div(m,v)
    NativeCorrection(e,s._epoch,v,d)
end
function Calculate.interpolate!(out::Fields.FaceScalarField,p::NativeScalarField,config)
    native_config(config);m=p.mesh
    native_field_values(out,m,length(m.faces),"trace_output")
    v=native_trace(m,native_field_values(p,m,length(m.cells),"trace_operand"))
    copyto!(out.values,v);out
end
function Discretise.correct_boundaries!(out::Fields.FaceScalarField,p::NativeScalarField,bcs,time,config)
    native_config(config);require_ncc(time===nothing,"native_no_boundary_time")
    m=p.mesh;v=native_field_values(out,m,length(m.faces),"trace_output")
    trace=native_trace(m,native_field_values(p,m,length(m.cells),"trace_operand"),native_boundary_snapshot(m,bcs))
    n=length(m.boundary_cellsID);v[1:n]=trace[1:n]
    copyto!(out.values,v);out
end
function Calculate.div!(out::Fields.ScalarField,f::NativeFaceScalarField,config)
    native_config(config);m=f.mesh
    native_field_values(out,m,length(m.cells),"div_output")
    values=native_field_values(f,m,length(m.faces),"div_operand")
    v=zeros(length(m.cells))
    for row in eachindex(v)
        for k in m.cell_faces_range[row];v[row]+=m.cell_nsign[k]*values[m.cell_faces[k]];end
        v[row]/=m.cell_volume[row]
    end
    for j in eachindex(m.boundary_cells), k in m.boundary_cell_faces_range[j]
        v[m.boundary_cells[j]]+=values[m.boundary_cell_faces[k]]/m.cell_volume[m.boundary_cells[j]]
    end
    require_ncc(all(isfinite,v),"native_intensive_divergence_overflow")
    # Reuse the existing native scalar-divergence kernels after screening and overflow preflight.
    invoke(Calculate.div!,Tuple{Fields.ScalarField,Fields.FaceScalarField,Any},out,f,config)
    out
end
function Calculate.Grad{S}(p::NativeScalarField) where S
    require_ncc(S===Discretise.Gauss,"native_gradient_scheme")
    m=p.mesh;native_field_values(p,m,length(m.cells),"gradient_operand")
    result=Fields.VectorField(ncc_native_scalar(m;values=zeros(length(m.cells))),
        ncc_native_scalar(m;values=zeros(length(m.cells))),ncc_native_scalar(m;values=zeros(length(m.cells))),m)
    Calculate.Grad{S,typeof(p),typeof(result),Int32,typeof(m)}(p,result,Int32(1),false,m)
end
function Calculate.green_gauss!(grad::Calculate.Grad{S,F,R,I,M},f,config) where {S,F,R<:Fields.VectorField,I,M<:StationaryNCCMesh}
    native_config(config);require_ncc(S===Discretise.Gauss && !grad.correct,"native_gradient_scheme")
    m=grad.mesh;require_ncc(grad.result.mesh===m && grad.field.mesh===m,"native_gradient_output_mesh")
    for part in (grad.result.x,grad.result.y,grad.result.z)
        native_field_values(part,m,length(m.cells),"gradient_output")
    end
    trace=native_field_values(f,m,length(m.faces),"gradient_trace")
    native_gradient(m,trace) # overflow preflight, no publication yet
    backend=config.hardware.backend;workgroup=config.hardware.workgroup
    x,y,z=grad.result.x,grad.result.y,grad.result.z
    Calculate._sized(Calculate._green_gauss!,backend,workgroup,length(m.cells))(x,y,z,f)
    Calculate._sized(Calculate.boundary_faces_contribution!,backend,workgroup,length(m.boundary_cells))(x,y,z,f)
    grad
end
# Screen the reverse direction too: a marked output cannot import a foreign operand.
Calculate.interpolate!(out::NativeFaceScalarField,p::Fields.ScalarField,config)=unsupported("foreign native interpolation operand")
Calculate.interpolate!(out::NativeFaceScalarField,p::NativeScalarField,config)=
    invoke(Calculate.interpolate!,Tuple{Fields.FaceScalarField,NativeScalarField,Any},out,p,config)
Discretise.correct_boundaries!(out::NativeFaceScalarField,p::Fields.ScalarField,bcs,time,config)=unsupported("foreign native boundary operand")
Discretise.correct_boundaries!(out::NativeFaceScalarField,p::NativeScalarField,bcs,time,config)=
    invoke(Discretise.correct_boundaries!,Tuple{Fields.FaceScalarField,NativeScalarField,Any,Any,Any},out,p,bcs,time,config)
Calculate.div!(out::NativeScalarField,f::Fields.FaceScalarField,config)=unsupported("foreign native divergence operand")
Calculate.div!(out::NativeScalarField,f::NativeFaceScalarField,config)=
    invoke(Calculate.div!,Tuple{Fields.ScalarField,NativeFaceScalarField,Any},out,f,config)

function Calculate.grad!(grad::Calculate.Grad{Discretise.Gauss,F,R,I,M},f,p,bcs,time,config) where {F,R<:Fields.VectorField,I,M<:StationaryNCCMesh}
    native_config(config);require_ncc(time===nothing && p===grad.field,"native_gradient_operand_time")
    m=grad.mesh;trace=native_trace(m,native_field_values(p,m,length(m.cells),"gradient_operand"),native_boundary_snapshot(m,bcs))
    native_field_values(f,m,length(m.faces),"gradient_trace")
    native_gradient(m,trace)
    # Validate output storage before modifying either published field.
    require_ncc(grad.result.mesh===m && p.mesh===m && !grad.correct,"native_gradient_output_mesh")
    for part in (grad.result.x,grad.result.y,grad.result.z)
        native_field_values(part,m,length(m.cells),"gradient_output")
    end
    copyto!(f.values,trace);Calculate.green_gauss!(grad,f,config)
end
