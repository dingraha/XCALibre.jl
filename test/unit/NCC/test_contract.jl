# SPDX-License-Identifier: MPL-2.0
using XCALibre, StaticArrays, Test
const NC = XCALibre.NCC
const NV = SVector{3,Float64}
const CONTRACT_MARKERS = Tuple{Symbol,Symbol,Symbol}[]
contract_marker(group, case, policy) = push!(CONTRACT_MARKERS,(group,case,policy))

# Newly authored algebra fixtures, not closed domains/intersection output.
# All cells have non-unit volumes. Internal plane x=.5; duplicated coupled
# 1->3 at image midpoint (or rebuilt 2->3); ordinary internal 1->2 unchanged.
function contract_input(case)
    cells=[NC.NCCCellInput(Int32(i),NV(i-1.,0.,0.),1.0 + 0.1*i) for i in 1:3]
    key(p,i)=NC.NCCFaceKey(:boundary,Int32(p),Int32(i))
    ik=NC.NCCFaceKey(:internal,nothing,Int32(1))
    faces=[NC.NCCFaceInput(ik,Int32(1),Int32(2),NV(1.,0.,0.),NV(.5,0.,0.),Int32(1))]
    patches=NC.NCCPatchInput[];pairs=Tuple{NC.NCCFaceKey,NC.NCCFaceKey}[]
    if case !== :ordinary_pair
        count=case === :duplicate_pair ? 2 : 1
        shift=case === :translated_pair ? NV(.2,0.,0.) : zero(NV)
        owner=case === :rewired_pair ? 2 : 1
        cf=(cells[owner].centre+cells[3].centre+shift)/2
        for i in 1:count
            a,b=key(1,i),key(2,i)
            push!(faces,NC.NCCFaceInput(a,Int32(owner),Int32(3),NV(1/count,0.,0.),cf,Int32(1)))
            push!(faces,NC.NCCFaceInput(b,Int32(3),Int32(owner),NV(-1/count,0.,0.),cf-shift,Int32(1)))
            push!(pairs,(a,b))
        end
        push!(patches,NC.NCCPatchInput(Int32(1),:left,:coupled,[key(1,i) for i in 1:count],Int32(2),shift))
        push!(patches,NC.NCCPatchInput(Int32(2),:right,:coupled,[key(2,i) for i in 1:count],Int32(1),-shift))
    end
    pid=length(patches)+1;k=key(pid,1)
    area=case === :residual_tiny ? 1e-18 : 1.
    push!(faces,NC.NCCFaceInput(k,Int32(1),nothing,NV(-area,0.,0.),NV(-.5,0.,0.),Int32(1)))
    push!(patches,NC.NCCPatchInput(Int32(pid),:wall,case === :residual_tiny ? :residual : :physical,[k],nothing,zero(NV)))
    if case === :empty_error
        for (name,kind) in ((:front,:empty),(:error,:error))
            push!(patches,NC.NCCPatchInput(Int32(length(patches)+1),name,kind,NC.NCCFaceKey[],nothing,zero(NV)))
        end
    end
    NC.NCCGeometryInput(cells=cells,faces=faces,patches=patches,pairs=pairs)
end
physical_input(s;values=[1.7]) = [p.name => (p.kind in (:empty,:error) ? NC.NoRuntimeFaces() : NC.FixedScalarValues(copy(values))) for p in s.patches if p.kind !== :coupled]
function rebuild_input(s;schema=s.schema,cells=s.cells,faces=s.faces,patches=s.patches,pairs=s.pairs,physical=s.physical)
    NC.NCCGeometryInput(;schema,cells,faces,patches,pairs,physical)
end
function change_cell(s,i;id=s.cells[i].id,centre=s.cells[i].centre,volume=s.cells[i].volume)
    c=copy(s.cells);c[i]=NC.NCCCellInput(id,centre,volume);rebuild_input(s;cells=c)
end
function change_face(s,i;key=s.faces[i].key,owner=s.faces[i].owner,neighbour=s.faces[i].neighbour,Sf=s.faces[i].Sf,Cf=s.faces[i].Cf,physical_face=s.faces[i].physical_face)
    f=copy(s.faces);f[i]=NC.NCCFaceInput(key,owner,neighbour,Sf,Cf,physical_face);rebuild_input(s;faces=f)
end
function change_patch(s,i;id=s.patches[i].id,name=s.patches[i].name,kind=s.patches[i].kind,keys=s.patches[i].keys,partner=s.patches[i].partner,translation=s.patches[i].translation)
    p=copy(s.patches);p[i]=NC.NCCPatchInput(id,name,kind,copy(keys),partner,translation);rebuild_input(s;patches=p)
end
# Explicit malformed cases target individual validation predicates.
function malformed(label,s)
    label===:schema && return rebuild_input(s;schema=2)
    label===:empty_cells && return rebuild_input(s;cells=NC.NCCCellInput[])
    label===:cell_ids && return change_cell(s,1;id=Int32(7))
    label===:patch_ids && return change_patch(s,1;id=Int32(7))
    label===:patch_names && return change_patch(s,2;name=s.patches[1].name)
    label===:face_keys && return change_face(s,3;key=s.faces[2].key)
    label===:patch_coverage && return change_patch(s,1;keys=NC.NCCFaceKey[])
    label===:owner && return change_face(s,1;owner=Int32(9))
    label===:neighbour && return change_face(s,1;neighbour=Int32(9))
    label===:internal_self && return change_face(s,1;neighbour=Int32(1))
    label===:physical_provenance && return change_face(s,1;physical_face=Int32(0))
    label===:pair_missing && return rebuild_input(s;pairs=Tuple{NC.NCCFaceKey,NC.NCCFaceKey}[])
    label===:pair_reciprocity && return rebuild_input(s;pairs=vcat(s.pairs,s.pairs))
    label===:partner_cells && return change_face(s,2;neighbour=Int32(2))
    label===:partner_patch && return change_patch(s,1;partner=Int32(1))
    label===:translation_inverse && return change_patch(s,1;translation=NV(.2,0.,0.))
    label===:paired_Cf && return change_face(s,2;Cf=s.faces[2].Cf+NV(0.,.1,0.))
    label===:paired_Sf && return change_face(s,2;Sf=s.faces[2].Sf*2)
    label===:nonfinite_C && return change_cell(s,1;centre=NV(NaN,0.,0.))
    label===:nonfinite_Sf && return change_face(s,1;Sf=NV(NaN,0.,0.))
    label===:nonfinite_Cf && return change_face(s,1;Cf=NV(NaN,0.,0.))
    label===:nonpositive_V && return change_cell(s,1;volume=0.)
    label===:zero_area && return change_face(s,1;Sf=zero(NV))
    label===:zero_delta && return change_cell(s,2;centre=s.cells[1].centre)
    label===:owner_projection && return change_face(s,1;Cf=s.cells[1].centre)
    label===:neighbour_projection && return change_face(s,1;Cf=s.cells[2].centre)
    label===:physical_zero_delta && return change_face(s,length(s.faces);Cf=s.cells[1].centre)
    error("unknown malformed case $label")
end
function argument_failure(fn,text)
    err=try fn();nothing catch e;e end
    @test err isa ArgumentError
    @test occursin(text,sprint(showerror,err))
end

@testset "NCC owned frozen scalar contract (unregistered)" begin
    empty!(CONTRACT_MARKERS)
    cases=(:ordinary_pair,:duplicate_pair,:translated_pair,:residual_tiny,:empty_error,:rewired_pair)
    policies=(NC.FoundationUncorrected(),NC.FoundationCorrected())
    for case in cases,policy in policies
        pname=Symbol(nameof(typeof(policy)));s=contract_input(case)
        g=NC.FrozenNCCGeometry(s);c=NC.NCCScalarContext(g;policy)
        before=NC.ncc_snapshot(c)
        @test length(before.geometry.input.cells)==3
        @test length(before.geometry.input.faces)==(case===:ordinary_pair ? 2 : case===:duplicate_pair ? 6 : 4)
        @test all(isfinite,before.factors)
        case===:residual_tiny && @test minimum(before.geometry.areas)==1e-18
        contract_marker(:accepted,case,pname)
        for target in (:caller_geometry,:snapshot_geometry,:caller_scalar,:snapshot_scalar,:caller_mobility,:caller_q,:caller_physical_values)
            if target===:caller_geometry
                s.faces[1]=NC.NCCFaceInput(s.faces[1].key,Int32(1),Int32(2),NV(9.,0.,0.),NV(.5,0.,0.),Int32(1))
                @test isequal(NC.ncc_snapshot(c),before)
            elseif target===:snapshot_geometry
                v=NC.ncc_snapshot(c);fill!(v.factors,7.);fill!(v.geometry.areas,7.)
                @test isequal(NC.ncc_snapshot(c),before)
            elseif target in (:caller_scalar,:snapshot_scalar)
                v=[1.,2.,3.];h=NC.ncc_scalar(c;values=v);old=NC.ncc_snapshot(h)
                target===:caller_scalar ? fill!(v,7.) : fill!(NC.ncc_snapshot(h).values,7.)
                @test isequal(NC.ncc_snapshot(h),old)
            elseif target===:caller_mobility
                v=ones(length(s.faces));h=NC.ncc_mobility(c;values=v);old=NC.ncc_snapshot(h);fill!(v,7.)
                @test isequal(NC.ncc_snapshot(h),old)
            elseif target===:caller_q
                v=zeros(length(s.faces));h=NC.ncc_face_flux(c;values=v);old=NC.ncc_snapshot(h);fill!(v,7.)
                @test isequal(NC.ncc_snapshot(h),old)
            else
                v=physical_input(s);h=NC.ncc_boundaries(c;physical=v);old=NC.ncc_snapshot(h)
                for p in v;last(p) isa NC.FixedScalarValues && fill!(last(p).values,7.);end
                @test isequal(NC.ncc_snapshot(h),old)
            end
            contract_marker(:ownership,Symbol(string(case)*"_"*string(target)),pname)
        end
    end
    labels=(:schema,:index_type,:backend,:precision,:empty_cells,:cell_ids,:patch_ids,:patch_names,:face_keys,:patch_coverage,:owner,:neighbour,:internal_self,:physical_provenance,:pair_missing,:pair_reciprocity,:partner_cells,:partner_patch,:translation_inverse,:paired_Cf,:paired_Sf,:rotation,:motion,:nonfinite_C,:nonfinite_Sf,:nonfinite_Cf,:nonpositive_V,:zero_area,:zero_delta,:owner_projection,:neighbour_projection,:physical_zero_delta)
    for policy in policies,label in labels
        s=contract_input(:duplicate_pair);c=NC.NCCScalarContext(NC.FrozenNCCGeometry(s);policy);old=NC.ncc_snapshot(c)
        fn=label===:index_type ? ()->NC.FrozenNCCGeometry(s;index_type=Int64) : label===:backend ? ()->NC.NCCScalarContext(c._geometry;policy,backend=:gpu) : label===:precision ? ()->NC.NCCScalarContext(c._geometry;policy,precision=Float32) : label===:rotation ? ()->NC.FrozenNCCGeometry(s;rotation=ones(3,3)) : label===:motion ? ()->NC.FrozenNCCGeometry(s;moving=true) : ()->NC.FrozenNCCGeometry(malformed(label,s))
        argument_failure(fn,label===:index_type ? "index_type" : label===:backend ? "backend" : label===:precision ? "precision" : string(label===:motion ? :motion : label))
        @test isequal(NC.ncc_snapshot(c),old)
        contract_marker(:constructor_rejected,label,Symbol(nameof(typeof(policy))))
    end
    for policy in policies
        s=contract_input(:duplicate_pair);g=NC.FrozenNCCGeometry(s);c=NC.NCCScalarContext(g;policy);other=NC.NCCScalarContext(g;policy)
        n=length(s.faces);old=NC.ncc_snapshot(c);bc=physical_input(s)
        tests=[
            (:foreign_context,()->NC.same_context(c,NC.ncc_scalar(other;values=ones(3))),"foreign_context"),
            (:scalar_length,()->NC.ncc_scalar(c;values=ones(2)),"scalar length"),
            (:scalar_nonfinite,()->NC.ncc_scalar(c;values=[NaN,1.,1.]),"scalar nonfinite"),
            (:mobility_length,()->NC.ncc_mobility(c;values=ones(n-1)),"mobility length"),
            (:mobility_nonpositive,()->NC.ncc_mobility(c;values=zeros(n)),"mobility nonpositive"),
            (:q_length,()->NC.ncc_face_flux(c;values=zeros(n-1)),"q length"),
            (:q_nonfinite,()->NC.ncc_face_flux(c;values=fill(NaN,n)),"q nonfinite"),
            (:q_unpaired,()->NC.ncc_face_flux(c;values=ones(n)),"q unpaired"),
            (:physical_missing,()->NC.ncc_boundaries(c;physical=Pair{Symbol,NC.ScalarBoundaryInput}[]),"physical missing/override"),
            (:physical_duplicate,()->NC.ncc_boundaries(c;physical=vcat(bc,bc)),"physical duplicate/name"),
            (:physical_coupled_override,()->NC.ncc_boundaries(c;physical=vcat(bc,[:left=>NC.ZeroGradient()])),"physical missing/override"),
            (:physical_value_length,()->NC.ncc_boundaries(c;physical=physical_input(s;values=Float64[])),"physical value length"),
            (:physical_value_nonfinite,()->NC.ncc_boundaries(c;physical=physical_input(s;values=[NaN])),"physical value nonfinite"),
            (:empty_runtime_values,()->NC.ncc_boundaries(NC.NCCScalarContext(NC.FrozenNCCGeometry(contract_input(:empty_error));policy);physical=[:wall=>NC.ZeroGradient(),:front=>NC.FixedScalarValues(Float64[]),:error=>NC.NoRuntimeFaces()]),"empty runtime values")]
        for (label,fn,text) in tests
            argument_failure(fn,text);@test isequal(NC.ncc_snapshot(c),old)
            contract_marker(:field_rejected,label,Symbol(nameof(typeof(policy))))
        end
    end
    for (case,h) in ((:geometry,NC.FrozenNCCGeometry(contract_input(:ordinary_pair))),(:context,NC.NCCScalarContext(NC.FrozenNCCGeometry(contract_input(:ordinary_pair));policy=policies[1])))
        old=NC.ncc_snapshot(h)
        for op in (:run!,:simple!,:piso!,:csimple!,:cpiso!,:simple_MRF!,:laplace!,:multiphase!,:godunov!,:filmModel!,:potential_flow!)
            argument_failure(()->getfield(XCALibre.Solvers,op)(h,nothing),"unsupported")
            @test isequal(NC.ncc_snapshot(h),old);contract_marker(:guard,op,case)
        end
        for (name,fn) in ((:adapt_GPU,()->XCALibre.adapt(:gpu,h)),(:distribute,()->XCALibre.Distribute.distribute(h)),(:writer_VTK,()->XCALibre.IOFormats.initialise_writer(XCALibre.IOFormats.VTK(),h)),(:writer_OpenFOAM,()->XCALibre.IOFormats.initialise_writer(XCALibre.IOFormats.OpenFOAM(),h)),(:stock_assign,()->XCALibre.Discretise.assign(h;region=nothing)))
            argument_failure(fn,"unsupported");@test isequal(NC.ncc_snapshot(h),old);contract_marker(:guard,name,case)
        end
    end
    policy_names=(:FoundationUncorrected,:FoundationCorrected)
    ownership_names=(:caller_geometry,:snapshot_geometry,:caller_scalar,:snapshot_scalar,:caller_mobility,:caller_q,:caller_physical_values)
    field_names=(:foreign_context,:scalar_length,:scalar_nonfinite,:mobility_length,:mobility_nonpositive,:q_length,:q_nonfinite,:q_unpaired,:physical_missing,:physical_duplicate,:physical_coupled_override,:physical_value_length,:physical_value_nonfinite,:empty_runtime_values)
    guard_names=(:run!,:simple!,:piso!,:csimple!,:cpiso!,:simple_MRF!,:laplace!,:multiphase!,:godunov!,:filmModel!,:potential_flow!,:adapt_GPU,:distribute,:writer_VTK,:writer_OpenFOAM,:stock_assign)
    expected=Set{Tuple{Symbol,Symbol,Symbol}}()
    union!(expected,((:accepted,g,p) for g in cases for p in policy_names))
    union!(expected,((:ownership,Symbol(string(g)*"_"*string(t)),p) for g in cases for t in ownership_names for p in policy_names))
    union!(expected,((:constructor_rejected,r,p) for r in labels for p in policy_names))
    union!(expected,((:field_rejected,r,p) for r in field_names for p in policy_names))
    union!(expected,((:guard,r,h) for r in guard_names for h in (:geometry,:context)))
    @test length(expected)==220
    @test Set(CONTRACT_MARKERS)==expected
    @test length(CONTRACT_MARKERS)==length(expected)
    @test allunique(CONTRACT_MARKERS)
end
