# Experimental frozen NCC scalar facade

Status: B2b-1 implementation only, **unexecuted and unqualified**. Accessible as `XCALibre.NCC`, not reexported into the top-level API. This page is deliberately not registered in documentation navigation yet. No scalar equation assembly, flux recovery or solver support is implemented in this stage.

The facade accepts explicit frozen computational geometry, both coupled sides and retained provenance. It does not compute polygon intersections or move a mesh. CPU Float64, Int32 addressing, identity tensor/translational images only. All inputs are copied; public snapshots are copies. Internal arrays are mutable/read-only-by-contract, not immutable. Accessing private implementation fields bypasses this contract; Julia privacy is not a security guarantee.

## Geometry schema

Use NCCCellInput(id,centre,volume), NCCFaceKey(class,patch,local_id), NCCFaceInput(key,owner,neighbour,Sf,Cf,physical_face), NCCPatchInput(id,name,kind,keys,partner,translation), and NCCGeometryInput(;schema=1,cells,faces,patches,pairs,physical=nothing). IDs are positive dense Int32. Internal face key class :internal, patch nothing; boundary class :boundary, patch positive. Internal neighbour positive; uncoupled boundary neighbour nothing. Coupled faces present on BOTH sides, with reciprocal cells and one pair declaration per two face keys.

Patch kinds :physical, :residual, :coupled, :empty, :error. Empty/error patches have zero runtime keys. Per-patch keys are ordered local IDs starting at1; input face-vector order is authoritative for face handles. Translation maps partner coordinates into owner's image. No rotation input. Optional NCCPhysicalInput contains owned points/polygon indices and bounds-checked physical provenance; no writer capability is implied by populated metadata. Absence explicitly retained as nothing.

FrozenNCCGeometry checks finite positive geometry, dense/unique IDs, patch coverage, reciprocal coupling, image Cf/Sf/inverse translations and positive owner/neighbour normal projections. Unsupported distance-magnitude fallback branches are rejected, not approximated. Tiny positive residual faces are not deleted. Pair comparison1e-11*max(1,maxabs sides) is a declared screen, not universal tiny-geometry robustness. Area derived from Sf; no authoritative reference coefficient inputs.

## Context and owned values

NCCScalarContext(geometry;policy=FoundationCorrected()) or FoundationUncorrected() explicitly selects scalar coefficient caches. Neither changes stock mesh/Periodic/correction defaults. Policy factors cover linked faces; physical factor is reciprocal owner-face distance with zero explicit correction. There is no diffusion operation yet.

ncc_scalar(context;values=Vector{Float64}) copies cell values. ncc_mobility requires positive finite face values. ncc_face_flux copies finite integrated retained q, with both-side opposite pair consistency1e-12 scaled; it does not require cellwise divergence zero or regenerate flux from velocity.

ncc_boundaries(context;physical=[name=>bc,...]) requires exact noncoupled patch coverage. Local ZeroGradient()/FixedScalarValues(vector) specify scalar traces; empty/error patches require NoRuntimeFaces(). Coupled boundaries are automatic and cannot be overridden. No stock assign or exposed generic AbstractBoundary interoperability.

ncc_snapshot returns plain copied input/derived geometry, coefficient and value data, without live raw stock mesh/BC/field/equation handles. Snapshot mutation never intentionally modifies context. Private identity tokens distinguish contexts; no automatic mutation tracking or field/history remap.

## Unsupported routes

NCC handles are not AbstractMesh/Physics/stock fields/equations. Public stock flow/conduction solvers, Adapt (including CPU clone in this initial stage), distribute, VTK/OpenFOAM writer initialization and stock assign reject them explicitly. No Physics conversion. Direct private algorithm/setup calls have no advertised interoperability; failure/bypass assumptions remain untested pending guard qualification. No vector/tensor, Euler/CN/TimeFlux, stress, turbulence/MRF, MPI/GPU/Float32, solves/reference/relaxation, histories, motion/ALE or analytical accuracy claim.

Geometry/context rebuild explicitly creates new handles; old handles remain unchanged. Corrected/uncorrected contexts have independent caches. Stage2 proposes scalar equations and epochs but is not implemented/approved by stage1 permission.

## Qualification and provenance

Unregistered independently authored test: test/unit/NCC/test_contract.jl. Proposed220 exact markers; no execution results yet. Existing passed external GPL validation certifies the earlier prototype, not these production handles. See ncc_provenance.md and root B2b implementation plan. Licence route selected by owner is independent MPL implementation with exposure disclosed, not clean-room or legal clearance. No GPL validation dependency or copied fixture/test assets.
