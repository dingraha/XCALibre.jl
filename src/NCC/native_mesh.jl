# SPDX-License-Identifier: MPL-2.0
function StationaryNCCBinding(ctx::NCCScalarContext)
    c=NCCScalarContext(FrozenNCCGeometry(deepcopy(ctx._geometry._data.input));policy=ctx._policy)
    g=c._geometry._data; nf=length(g.input.faces); nc=length(g.input.cells)
    order=Int32[]
    boundaries=Mesh.Boundary{Symbol,UnitRange{Int32}}[]
    for p in g.input.patches
        start=Int32(length(order)+1)
        append!(order,Int32[g.keymap[k] for k in p.keys])
        push!(boundaries,Mesh.Boundary(p.name,start:Int32(length(order))))
    end
    nb=length(order)
    append!(order,Int32[g.keymap[NCCFaceKey(:internal,nothing,Int32(i))] for i in 1:nf-nb])
    require_ncc(sort(order)==Int32.(1:nf),"native_permutation")
    inverse=zeros(Int32,nf)
    for (i,j) in enumerate(order); inverse[j]=Int32(i); end
    empty_range=Int32(1):Int32(0)
    owners=SVector{2,Int32}[]
    for j in order
        f=g.input.faces[j]
        push!(owners,SVector{2,Int32}(f.owner,f.key.class===:internal ? f.neighbour : f.owner))
    end
    faces=Mesh.FaceArrays{Mesh.Face3D}(fill(empty_range,nf),owners,
        Vec3[g.input.faces[j].Cf for j in order],g.normals[order],
        Vec3[g.deltas[j]/g.distances[j] for j in order],g.areas[order],g.distances[order],g.weights[order])
    cell_faces=Int32[]; neighbours=Int32[]; signs=Int8[]; ranges=UnitRange{Int32}[]
    for cell in 1:nc
        start=Int32(length(cell_faces)+1)
        for fid in nb+1:nf
            o,n=owners[fid]
            if cell==o || cell==n
                push!(cell_faces,Int32(fid));push!(neighbours,cell==o ? n : o)
                push!(signs,cell==o ? Int8(1) : Int8(-1))
            end
        end
        push!(ranges,start:Int32(length(cell_faces)))
    end
    cells=Mesh.CellArrays(Vec3[x.centre for x in g.input.cells],Float64[x.volume for x in g.input.cells],fill(empty_range,nc),ranges)
    nodes=Mesh.NodeArrays(Vec3[],UnitRange{Int32}[])
    boundary_owners=Int32[x[1] for x in owners[1:nb]]
    groups=Mesh.boundary_cell_faces(boundary_owners)
    runtime=Mesh.Mesh3(cells,Int32[],cell_faces,neighbours,signs,faces,Int32[],
        Float64[g.areas[j]*c._factors[j] for j in order],boundaries,nodes,Int32[],
        Float64[0.0],Int32[0],boundary_owners,groups...)
    require_ncc(length(cell_faces)==2*(nf-nb),"native_internal_incidence")
    require_ncc(sort(runtime.boundary_cell_faces)==Int32.(1:nb),"native_boundary_gather")
    require_ncc(all(isfinite,runtime.face_gDiff),"native_coefficient_overflow")
    StationaryNCCBinding(c,runtime,order,inverse,Val(:owned))
end
function native_field_values(field,mesh,n,label)
    require_ncc(field isa Union{Fields.ScalarField,Fields.FaceScalarField} && field.mesh===mesh,"native_"*label*"_mesh")
    owned_values(field.values,n,"native_"*label)
end
function ncc_native_scalar(m::StationaryNCCMesh;values::Vector{Float64})
    Fields.ScalarField(owned_values(values,length(m.cells),"native_scalar"),m)
end
function ncc_native_face_scalar(m::StationaryNCCMesh;values::Vector{Float64},order::Symbol)
    require_ncc(order in (:input,:native),"native_face_order")
    v=owned_values(values,length(m.faces),"native_face_scalar")
    Fields.FaceScalarField(order===:input ? v[m._binding._native_to_input] : v,m)
end
function native_config(config)
    require_ncc(hasproperty(config,:hardware) && config.hardware.backend isa CPU,"native_CPU_config")
    require_ncc(hasproperty(config.hardware,:workgroup),"native_workgroup")
    w=config.hardware.workgroup
    require_ncc(w isa Calculate.AutoTune || (w isa Integer && w>0),"native_workgroup")
    (backend=config.hardware.backend,workgroup=w)
end
function native_runtime_snapshot(m)
    r=m._binding._runtime
    (; cell_centre=copy(r.cell_centre),cell_volume=copy(r.cell_volume),
       cell_nodes_range=copy(r.cell_nodes_range),cell_faces_range=copy(r.cell_faces_range),
       cell_nodes=copy(r.cell_nodes),cell_faces=copy(r.cell_faces),cell_neighbours=copy(r.cell_neighbours),cell_nsign=copy(r.cell_nsign),
       face_nodes_range=copy(r.face_nodes_range),face_ownerCells=copy(r.face_ownerCells),face_centre=copy(r.face_centre),
       face_normal=copy(r.face_normal),face_e=copy(r.face_e),face_area=copy(r.face_area),face_delta=copy(r.face_delta),
       face_weight=copy(r.face_weight),face_nodes=copy(r.face_nodes),face_gDiff=copy(r.face_gDiff),
       boundaries=[(name=b.name,range=b.IDs_range) for b in r.boundaries],node_coords=copy(r.node_coords),
       node_cells_range=copy(r.node_cells_range),node_cells=copy(r.node_cells),get_float=copy(r.get_float),get_int=copy(r.get_int),
       boundary_cellsID=copy(r.boundary_cellsID),boundary_cells=copy(r.boundary_cells),
       boundary_cell_faces_range=copy(r.boundary_cell_faces_range),boundary_cell_faces=copy(r.boundary_cell_faces))
end
ncc_snapshot(b::StationaryNCCBinding)=(context=ncc_snapshot(b._context),native_to_input=copy(b._native_to_input),input_to_native=copy(b._input_to_native),runtime=native_runtime_snapshot(StationaryNCCMesh(b)))
ncc_snapshot(m::StationaryNCCMesh)=ncc_snapshot(m._binding)
