# SPDX-License-Identifier: MPL-2.0
# Deliberately unregistered. Implementation-stage tests; not executed/qualified yet.
using XCALibre, StaticArrays, LinearAlgebra, Test, Adapt
const NN=XCALibre.NCC
const NVN=SVector{3,Float64}
function native_test_geometry(which)
    C=[NVN(0,0,0),NVN(1,.2,.1),NVN(2,-.1,.3)]
    cells=[NN.NCCCellInput(Int32(i),C[i],[1.1,.8,1.3][i]) for i in 1:3]
    key(p,i)=NN.NCCFaceKey(:boundary,Int32(p),Int32(i))
    internal=NN.NCCFaceKey(:internal,nothing,Int32(1))
    faces=[NN.NCCFaceInput(internal,Int32(1),Int32(2),NVN(1,0,0),.35*C[2],Int32(1))]
    patches=NN.NCCPatchInput[];pairs=Tuple{NN.NCCFaceKey,NN.NCCFaceKey}[]
    links=which===:ordinary ? Tuple{Int,Int}[] : which in (:duplicate,:rewired) ? [(1,2),(which===:rewired ? 2 : 1,3)] : [(1,3)]
    shift=which===:translated ? NVN(.2,.1,0) : zero(NVN)
    for (i,(o,n)) in enumerate(links)
        a,b=key(1,i),key(2,i);cf=C[o]+.4*(C[n]+shift-C[o]);sf=NVN(.6,0,0)
        push!(faces,NN.NCCFaceInput(a,Int32(o),Int32(n),sf,cf,Int32(1)))
        push!(faces,NN.NCCFaceInput(b,Int32(n),Int32(o),-sf,cf-shift,Int32(1)))
        push!(pairs,(a,b))
    end
    if !isempty(links)
        push!(patches,NN.NCCPatchInput(Int32(1),:a,:coupled,[key(1,i) for i in eachindex(links)],Int32(2),shift))
        push!(patches,NN.NCCPatchInput(Int32(2),:b,:coupled,[key(2,i) for i in eachindex(links)],Int32(1),-shift))
    end
    for (name,o,sf,offset) in ((:wall,1,NVN(-1,0,0),NVN(-.4,.1,0)),(:out,2,NVN(0,1,0),NVN(0,.5,.1)))
        pid=Int32(length(patches)+1);k=key(pid,1);a=which===:tiny && name===:wall ? 1e-18 : 1.0
        push!(faces,NN.NCCFaceInput(k,Int32(o),nothing,a*sf,C[o]+offset,Int32(1)))
        push!(patches,NN.NCCPatchInput(pid,name,which===:tiny && name===:wall ? :residual : :physical,[k],nothing,zero(NVN)))
    end
    if which===:empty
        for (name,kind) in ((:front,:empty),(:error,:error))
            push!(patches,NN.NCCPatchInput(Int32(length(patches)+1),name,kind,NN.NCCFaceKey[],nothing,zero(NVN)))
        end
    end
    # Deliberately retain non-native input ordering.
    NN.NCCGeometryInput(;cells,faces,patches,pairs)
end
# Independent dense oracle: derives coefficients, interpolation and gradient directly
# from the mathematical input, not facade assembly/native implementation helpers.
function native_dense_oracle(input,corrected,form,scheme,p,q,mu,source)
    n=length(p);nf=length(input.faces);A=zeros(n,n);b=[input.cells[i].volume*source[i] for i in 1:n]
    trace=zeros(nf);k=zeros(nf);cw=zeros(nf);corr=fill(zero(NVN),nf);grad=fill(zero(NVN),n)
    sigma=form===:positive_diffusion ? 1 : form===:convection ? 0 : -1
    conv=form in (:convection,:convection_minus_diffusion)
    div(v)=begin
        d=zeros(n)
        for (i,f) in enumerate(input.faces)
            d[f.owner]+=v[i]
            f.key.class===:internal && (d[f.neighbour]-=v[i])
        end
        d
    end
    for (i,f) in enumerate(input.faces)
        o=f.owner;area=norm(f.Sf);normal=f.Sf/area
        if f.neighbour===nothing
            fixed=input.patches[f.key.patch].name===:wall;pb=fixed ? 1.8 : p[o]
            trace[i]=pb;k[i]=sigma==0 ? 0.0 : mu[i]*area/norm(f.Cf-input.cells[o].centre)
            if fixed
                A[o,o]-=sigma*k[i];b[o]-=((conv ? q[i] : 0.0)+sigma*k[i])*pb
            elseif conv
                A[o,o]+=q[i]
            end
        else
            neighbour=f.neighbour;Co=input.cells[o].centre
            shift=f.key.class===:internal ? zero(NVN) : input.patches[f.key.patch].translation
            d=input.cells[neighbour].centre+shift-Co
            w=dot(normal,d-(f.Cf-Co))/dot(normal,d);cw[i]=w
            denom=corrected ? max(dot(normal,d),.05*norm(d)) : norm(d)
            k[i]=sigma==0 ? 0.0 : mu[i]*area/denom
            corr[i]=corrected ? normal-d/denom : zero(NVN)
            trace[i]=w*p[o]+(1-w)*p[neighbour]
            a=(conv ? (scheme===:upwind ? max(q[i],0.0) : q[i]*w) : 0.0)-sigma*k[i]
            c=(conv ? (scheme===:upwind ? min(q[i],0.0) : q[i]*(1-w)) : 0.0)+sigma*k[i]
            A[o,o]+=a;A[o,neighbour]+=c
            if f.key.class===:internal
                A[neighbour,o]-=a;A[neighbour,neighbour]-=c
            end
        end
        grad[o]+=trace[i]*f.Sf
        f.key.class===:internal && (grad[f.neighbour]-=trace[i]*f.Sf)
    end
    for i in 1:n;grad[i]/=input.cells[i].volume;end
    E=zeros(nf);flux=zeros(nf)
    for (i,f) in enumerate(input.faces)
        if f.neighbour!==nothing
            E[i]=sigma==0 ? 0.0 : mu[i]*norm(f.Sf)*dot(corr[i],cw[i]*grad[f.owner]+(1-cw[i])*grad[f.neighbour])
            fc=conv ? (scheme===:upwind ? max(q[i],0.0)*p[f.owner]+min(q[i],0.0)*p[f.neighbour] : q[i]*trace[i]) : 0.0
            flux[i]=fc+sigma*(k[i]*(p[f.neighbour]-p[f.owner])+E[i])
        else
            fixed=input.patches[f.key.patch].name===:wall
            flux[i]=(conv ? q[i]*trace[i] : 0.0)+(fixed ? sigma*k[i]*(trace[i]-p[f.owner]) : 0.0)
        end
    end
    (A=A,b=b,trace=trace,gradient=grad,E=E,full=b-sigma*div(E),flux=flux,div=div(flux))
end
function native_dense(snapshot)
    A=zeros(3,3)
    for i in 1:3,slot in snapshot.rowptr[i]:snapshot.rowptr[i+1]-1
        A[i,snapshot.colval[slot]]=snapshot.nzval[slot]
    end
    A
end
@testset "S1 native frozen scalar — unqualified implementation tests" begin
    config=(hardware=(backend=XCALibre.CPU(),workgroup=1),)
    saved=Dict()
    for which in (:ordinary,:duplicate,:translated,:rewired,:tiny,:empty), corrected in (false,true),
        trial in (:linear,:upwind,:positive,:negative,:mixed_linear,:mixed_upwind)
        input=native_test_geometry(which)
        ctx=NN.NCCScalarContext(NN.FrozenNCCGeometry(input);policy=corrected ? NN.FoundationCorrected() : NN.FoundationUncorrected())
        binding=NN.StationaryNCCBinding(ctx);m=NN.StationaryNCCMesh(binding)
        ms=NN.ncc_snapshot(m);order=ms.native_to_input;inverse=ms.input_to_native;nf=length(input.faces)
        @test inverse[order]==Int32.(1:nf)
        @test length(m.cell_faces)==2
        @test sort(m.boundary_cell_faces)==Int32.(1:length(m.boundary_cellsID))
        @test isempty(m.nodes) && isempty(m.face_nodes) && all(isempty,m.face_nodes_range)
        @test all(x->x[1]==x[2],m.face_ownerCells[1:length(m.boundary_cellsID)])
        q=[f.neighbour===nothing ? -.15 : .3 for f in input.faces];mu=fill(1.2,nf)
        for (a,b) in input.pairs
            ia=findfirst(f->f.key==a,input.faces);ib=findfirst(f->f.key==b,input.faces);q[ib]=-q[ia]
        end
        p=NN.ncc_native_scalar(m;values=[1.4,-.5,.35]);src=NN.ncc_native_scalar(m;values=[.11,-.2,.07])
        qf=NN.ncc_native_face_scalar(m;values=q,order=:input);mf=NN.ncc_native_face_scalar(m;values=mu,order=:input)
        physical=[x.name=>(x.kind in (:empty,:error) ? NN.NoRuntimeFaces() : x.name===:wall ? NN.FixedScalarValues([1.8]) : NN.ZeroGradient()) for x in input.patches if x.kind!==:coupled]
        bcs=NN.ncc_native_boundaries(m;physical)
        form=trial===:positive ? :positive_diffusion : trial===:negative ? :negative_diffusion : trial in (:linear,:upwind) ? :convection : :convection_minus_diffusion
        conv=trial in (:linear,:upwind,:mixed_linear,:mixed_upwind);up=trial in (:upwind,:mixed_upwind)
        scheme=conv ? (up ? XCALibre.Upwind() : XCALibre.Linear()) : nothing
        e=NN.ncc_native_equation(m,p;form,convection=scheme,q=conv ? qf : nothing,mobility=form===:convection ? nothing : mf,source=src,boundaries=bcs)
        oracle=native_dense_oracle(input,corrected,form,up ? :upwind : :linear,p.values,q,mu,src.values)
        old=nothing
        for dirty in 1:3
            fill!(first(XCALibre.ModelFramework.get_sparse_fields(e.equation.A)),123.0);fill!(e.equation.b,-99.0)
            XCALibre.discretise!(e,p,config)
            @test NN.ncc_snapshot(e).state===:internal
            XCALibre.apply_boundary_conditions!(e,bcs,nothing,nothing,config)
            snap=NN.ncc_snapshot(e)
            @test native_dense(snap)≈oracle.A atol=1e-12 rtol=1e-12
            @test snap.compact_source≈oracle.b atol=1e-12 rtol=1e-12
            tg=NN.ncc_native_trace_gradient(e);E=NN.ncc_native_correction(e,tg)
            @test NN.ncc_snapshot(tg).trace≈oracle.trace[order] atol=1e-12 rtol=1e-12
            @test reduce(vcat,NN.ncc_snapshot(tg).gradient)≈reduce(vcat,oracle.gradient) atol=1e-12 rtol=1e-12
            @test NN.ncc_snapshot(E).values≈oracle.E[order] atol=1e-12 rtol=1e-12
            old===nothing || @test_throws ArgumentError NN.ncc_native_complete!(e,old)
            NN.ncc_native_complete!(e,E)
            @test_throws ArgumentError NN.ncc_native_complete!(e,E)
            flux=NN.ncc_snapshot(NN.ncc_native_flux(e,E))
            @test flux.values≈oracle.flux[order] atol=1e-12 rtol=1e-12
            @test flux.divergence≈oracle.div atol=1e-12 rtol=1e-12
            snap=NN.ncc_snapshot(e)
            @test snap.full_source≈oracle.full atol=1e-12 rtol=1e-12
            @test oracle.A*p.values-snap.full_source+snap.volume_source≈flux.divergence atol=1e-12 rtol=1e-12
            out=NN.ncc_native_scalar(m;values=zeros(3));XCALibre.div!(out,flux.face_field,config)
            @test out.values.*m.cell_volume≈flux.divergence atol=1e-12 rtol=1e-12
            old=E
        end
        saved[(which,corrected,trial)]=NN.ncc_snapshot(e)
        @test_throws ArgumentError XCALibre.assign((p=(),);region=m)
        @test_throws ArgumentError Adapt.adapt(XCALibre.CPU(),m)
        @test_throws ArgumentError XCALibre.ModelFramework.ScalarEquation(p,bcs)
        @test_throws ArgumentError XCALibre.Solve.solve_system!(e,nothing,p,nothing,config)
        @test_throws ArgumentError XCALibre.run!(e,config)
        before=NN.ncc_snapshot(e)
        detached=XCALibre.Fields.ScalarField(copy(p.values),())
        @test_throws ArgumentError NN.ncc_native_equation(m,detached;form,convection=scheme,q=conv ? qf : nothing,mobility=form===:convection ? nothing : mf,source=src,boundaries=bcs)
        foreign=NN.StationaryNCCMesh(NN.StationaryNCCBinding(ctx))
        foreign_p=NN.ncc_native_scalar(foreign;values=copy(p.values))
        @test_throws ArgumentError XCALibre.interpolate!(qf,foreign_p,config)
        snap=NN.ncc_snapshot(e);fill!(snap.nzval,888.0)
        @test NN.ncc_snapshot(e).nzval==before.nzval
        snapshot_binding=NN.ncc_snapshot(binding);fill!(snapshot_binding.runtime.cell_volume,999.0)
        @test m.cell_volume==[1.1,.8,1.3]
        epoch=e.equation._epoch;e.equation._epoch=typemax(UInt64)
        @test_throws ArgumentError XCALibre.discretise!(e,p,config)
        @test NN.ncc_snapshot(e).nzval==before.nzval
        e.equation._epoch=epoch
        original_source=copy(src.values);fill!(src.values,floatmax(Float64))
        @test_throws ArgumentError XCALibre.discretise!(e,p,config)
        @test NN.ncc_snapshot(e).full_source==before.full_source
        @test NN.ncc_snapshot(e).epoch==before.epoch
        copyto!(src.values,original_source)
        p.values[1]+=1.0
        @test_throws ArgumentError NN.ncc_native_flux(e,old)
        @test NN.ncc_snapshot(e).nzval==before.nzval
        p.values[1]-=1.0 # Explicit reassembly below; subtraction is not assumed bitwise restoration.
        src.values[1]=NaN
        @test_throws ArgumentError XCALibre.discretise!(e,p,config)
        @test NN.ncc_snapshot(e).nzval==before.nzval
        @test NN.ncc_snapshot(e).epoch==before.epoch
        src.values[1]=0.0;src.values[2]=0.0;src.values[3]=0.0
        if form===:negative_diffusion
            XCALibre.discretise!(e,p,config);XCALibre.apply_boundary_conditions!(e,bcs,nothing,nothing,config)
            E=NN.ncc_native_correction(e,NN.ncc_native_trace_gradient(e));NN.ncc_native_complete!(e,E)
            component=NN.ncc_native_pressure_component(e,E)
            recovered=NN.ncc_snapshot(NN.ncc_native_recover(component;predictor=qf,previous=p))
            @test recovered.values≈qf.values+NN.ncc_snapshot(component).increment atol=1e-12 rtol=1e-12
            XCALibre.discretise!(e,p,config)
            @test_throws ArgumentError NN.ncc_native_recover(component;predictor=qf,previous=p)
        end
    end
    for corrected in (false,true),trial in (:linear,:upwind,:positive,:negative,:mixed_linear,:mixed_upwind)
        a=saved[(:duplicate,corrected,trial)];b=saved[(:rewired,corrected,trial)]
        @test length(a.nzval)==length(b.nzval)
        @test (a.rowptr,a.colval)!=(b.rowptr,b.colval)
        @test native_dense(a)!=native_dense(b)
    end
end
