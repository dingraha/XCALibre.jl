# SPDX-License-Identifier: MPL-2.0
# Explicit public-domain containment. Reflection/private raw storage access is not protected.
const NativePhysics = ModelPhysics.Physics{T,F,SO,M,Tu,E,D,BI} where {T,F,SO,M,Tu,E,D<:StationaryNCCMesh,BI}
for operation in (:run!,:simple!,:piso!,:csimple!,:cpiso!,:simple_MRF!,:laplace!,:multiphase!,:godunov!,:filmModel!,:potential_flow!)
    @eval begin
        Solvers.$operation(m::StationaryNCCMesh,config;kwargs...)=unsupported($(string(operation)))
        Solvers.$operation(e::NativeScalarEquation,config;kwargs...)=unsupported($(string(operation)))
        Solvers.$operation(p::NativePhysics,config;kwargs...)=unsupported($(string(operation)))
    end
end
# Stock run! signatures restrict time/fluid/energy independently of domain. Add their
# intersections, including legacy seven-parameter partial Physics signatures, rather than
# assuming the broad native-domain guard wins dispatch. Conduction intersections also
# dominate simultaneous fluid and energy matches.
for (time,fluid) in ((ModelPhysics.Steady,ModelPhysics.Incompressible),
    (ModelPhysics.Steady,ModelPhysics.Incompressible_MRF),
    (ModelPhysics.Transient,ModelPhysics.Incompressible),
    (ModelPhysics.Steady,ModelPhysics.WeaklyCompressible),
    (ModelPhysics.Steady,ModelPhysics.Compressible),
    (ModelPhysics.Transient,ModelPhysics.WeaklyCompressible),
    (ModelPhysics.Transient,ModelPhysics.Compressible),
    (Any,ModelPhysics.Multiphase),(Any,ModelPhysics.SupersonicFlow))
    @eval begin
        Solvers.run!(p::ModelPhysics.Physics{T,F,SO,M,Tu,E,D,BI},config;kwargs...) where {T<:$time,F<:$fluid,SO,M,Tu,E,D<:StationaryNCCMesh,BI}=unsupported("run!")
        Solvers.run!(p::ModelPhysics.Physics{T,F,SO,M,Tu,E,D,BI},config;kwargs...) where {T<:$time,F<:$fluid,SO,M,Tu,E<:ModelPhysics.Conduction,D<:StationaryNCCMesh,BI}=unsupported("run!")
    end
end
Solvers.run!(p::ModelPhysics.Physics{T,F,SO,M,Tu,E,D,BI},config;kwargs...) where {T,F,SO,M,Tu,E<:ModelPhysics.Conduction,D<:StationaryNCCMesh,BI}=unsupported("run!")
Solvers.run!(p::ModelPhysics.Physics{T,F,SO,M,Tu,E,D,BI},config;kwargs...) where {T<:ModelPhysics.Transient,F<:ModelPhysics.Incompressible,SO,M<:ModelPhysics.EFM,Tu,E,D<:StationaryNCCMesh,BI}=unsupported("run!")
Solvers.run!(p::ModelPhysics.Physics{T,F,SO,M,Tu,E,D,BI},config;kwargs...) where {T<:ModelPhysics.Transient,F<:ModelPhysics.Incompressible,SO,M<:ModelPhysics.EFM,Tu,E<:ModelPhysics.Conduction,D<:StationaryNCCMesh,BI}=unsupported("run!")

Adapt.adapt_structure(to,m::StationaryNCCMesh)=unsupported("native adaptation")
Adapt.adapt_structure(to,s::NativeScalarStorage)=unsupported("native storage adaptation")
Adapt.adapt_structure(to,e::NativeScalarEquation)=unsupported("native equation adaptation")
Adapt.adapt_structure(to,b::NativeScalarBoundary)=unsupported("native boundary adaptation")
for field in (:NativeScalarField,:NativeFaceScalarField,:NativeVectorField)
    @eval Adapt.adapt_structure(to,f::$field)=unsupported("native field adaptation")
end
Adapt.adapt_structure(to,p::NativePhysics)=unsupported("native Physics adaptation")
Adapt.adapt_structure(to,g::Calculate.Grad{S,F,R,I,M}) where {S,F,R,I,M<:StationaryNCCMesh}=unsupported("native gradient adaptation")
ModelPhysics.original(::StationaryNCCMesh)=unsupported("native Physics construction")
ModelFramework.ScalarEquation(p::NativeScalarField,bcs)=unsupported("unchecked ScalarEquation; use ncc_native_equation")
ModelFramework.VectorEquation(p::Fields.VectorField{S1,S2,S3,M},bcs) where {S1<:Fields.ScalarField,S2,S3,M<:StationaryNCCMesh}=unsupported("native vector equation")
ModelFramework.:(→)(model::ModelFramework.Model,s::NativeScalarStorage)=unsupported("unchecked native model attachment")
Solve.solve_equation!(e::NativeScalarEquation,p,bcs,setup,config;kwargs...)=unsupported("native solve_equation!")
Solve.solve_system!(e::NativeScalarEquation,setup,result,component,config)=unsupported("native solve_system!")
Solvers.correct_mass_flux!(q,e::NativeScalarEquation,config;kwargs...)=unsupported("stock pressure recovery; use ncc_native_recover")
Solve.setReference!(e::NativeScalarEquation,ref,cell,config)=unsupported("native pressure reference")
Solve.implicit_relaxation!(e::NativeScalarEquation,p,alpha,component,config)=unsupported("native relaxation")
Solve.implicit_relaxation_diagdom!(e::NativeScalarEquation,p,alpha,component,config)=unsupported("native relaxation")
Solve.make_symmetric!(e::NativeScalarEquation,config)=unsupported("native matrix alteration")
Distribute.distribute(m::StationaryNCCMesh;kwargs...)=unsupported("native distribution")
Distribute.partition_mesh(m::StationaryNCCMesh,n::Integer;kwargs...)=unsupported("native partition")
Distribute.partition_cells(m::StationaryNCCMesh,n::Integer;kwargs...)=unsupported("native partition")
Distribute.decompose(m::StationaryNCCMesh,n::Integer;kwargs...)=unsupported("native decomposition")
Distribute.extract_subdomain(m::StationaryNCCMesh,parts,n::Integer;kwargs...)=unsupported("native subdomain")
IOFormats.initialise_writer(format::IOFormats.VTK,m::StationaryNCCMesh)=unsupported("native VTK writer")
IOFormats.initialise_writer(format::IOFormats.OpenFOAM,m::StationaryNCCMesh)=unsupported("native OpenFOAM writer")
Calculate.laplacian!(out,f,p::NativeScalarField,bcs,time,config;kwargs...)=unsupported("generic native laplacian!")
Calculate.surface_gradient!(out,f,p::NativeScalarField,bcs,time,config)=unsupported("generic native surface_gradient!")
Calculate.Grad{S}(p::NativeFaceScalarField) where S=unsupported("native face-gradient constructor; use screened scalar Gauss")
Calculate.Grad{S}(p::Fields.VectorField{S1,S2,S3,M}) where {S,S1<:Fields.ScalarField,S2,S3,M<:StationaryNCCMesh}=unsupported("native vector gradient")
Calculate.interpolate_upwind!(out::Fields.FaceScalarField,p::NativeScalarField,q,config)=unsupported("native generic upwind interpolation")
Calculate.interpolate_harmonic!(out::Fields.FaceScalarField,p::NativeScalarField,config)=unsupported("native harmonic interpolation")
Calculate.interpolate_vanleer!(out::Fields.FaceScalarField,p::NativeScalarField,g::Calculate.Grad,q,config)=unsupported("native VanLeer interpolation")
Calculate.interpolate_midpoint!(out::Fields.FaceScalarField,p::NativeScalarField,config)=unsupported("native midpoint interpolation")
Calculate.grad!(g::Calculate.Grad{Discretise.Gauss,F,R,I,M},f,p,time,config) where {F,R<:Fields.VectorField,I,M<:StationaryNCCMesh}=unsupported("native Gauss without explicit boundary provenance")
