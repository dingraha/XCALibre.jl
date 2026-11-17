# Experimental frozen NCC scalar facade

Status: **B2b-1 implemented and qualified within the declared stationary scalar contract scope** (external V2 qualification). Accessible as `XCALibre.NCC`, not reexported into the top-level API. This page is deliberately not registered in documentation navigation yet. B2b-2 scalar algebra is **implemented and qualified within the separately declared stationary frozen algebra scope**; this is not general solver readiness. B2b-3 V2 additionally passed bounded frozen Foundation scalar reference parity. S1 V5 additionally passed the separately declared stationary native scalar bridge qualification; **stock solver support remains refused**.

The facade accepts explicit frozen computational geometry, both coupled sides and retained provenance. It does not compute polygon intersections or move a mesh. CPU Float64, Int32 addressing, identity tensor/translational images only. All inputs are copied; public snapshots are copies. Internal arrays are mutable/read-only-by-contract, not immutable. Accessing private implementation fields bypasses this contract; Julia privacy is not a security guarantee.

## Geometry schema

Use NCCCellInput(id,centre,volume), NCCFaceKey(class,patch,local_id), NCCFaceInput(key,owner,neighbour,Sf,Cf,physical_face), NCCPatchInput(id,name,kind,keys,partner,translation), and NCCGeometryInput(;schema=1,cells,faces,patches,pairs,physical=nothing). IDs are positive dense Int32. Internal face key class :internal, patch nothing; boundary class :boundary, patch positive. Internal neighbour positive; uncoupled boundary neighbour nothing. Coupled faces present on BOTH sides, with reciprocal cells and one pair declaration per two face keys.

Patch kinds :physical, :residual, :coupled, :empty, :error. Empty/error patches have zero runtime keys. Per-patch keys are ordered local IDs starting at1; input face-vector order is authoritative for face handles. Translation maps partner coordinates into owner's image. No rotation input. Optional NCCPhysicalInput contains owned points/polygon indices and bounds-checked physical provenance; no writer capability is implied by populated metadata. Absence explicitly retained as nothing.

FrozenNCCGeometry checks finite positive geometry, dense/unique IDs, patch coverage, reciprocal coupling, image Cf/Sf/inverse translations and positive owner/neighbour normal projections. Unsupported distance-magnitude fallback branches are rejected, not approximated. Tiny positive residual faces are not deleted. Pair comparison1e-11*max(1,maxabs sides) is a declared screen, not universal tiny-geometry robustness. Area derived from Sf; no authoritative reference coefficient inputs.

## Context and owned values

NCCScalarContext(geometry;policy=FoundationCorrected()) or FoundationUncorrected() explicitly selects scalar coefficient caches. Neither changes stock mesh/Periodic/correction defaults. Policy factors cover linked faces; physical factor is reciprocal owner-face distance with zero explicit correction. Stage2 scalar diffusion passed the bounded independently checked algebra qualification described below; bounded frozen reference parity subsequently passed B2b-3 V2; native solver integration remains separate.

ncc_scalar(context;values=Vector{Float64}) copies cell values. ncc_mobility requires positive finite face values. ncc_face_flux copies finite integrated retained q, with both-side opposite pair consistency1e-12 scaled; it does not require cellwise divergence zero or regenerate flux from velocity.

ncc_boundaries(context;physical=[name=>bc,...]) requires exact noncoupled patch coverage. Local ZeroGradient()/FixedScalarValues(vector) specify scalar traces; empty/error patches require NoRuntimeFaces(). Coupled boundaries are automatic and cannot be overridden. No stock assign or exposed generic AbstractBoundary interoperability.

ncc_snapshot returns plain copied input/derived geometry, coefficient and value data, without live raw stock mesh/BC/field/equation handles. Snapshot mutation never intentionally modifies context. Private identity tokens distinguish contexts; no automatic mutation tracking or field/history remap.

## Unsupported routes

NCC handles are not AbstractMesh/Physics/stock fields/equations. Public stock flow/conduction solvers, Adapt (including CPU clone in this initial stage), distribute, VTK/OpenFOAM writer initialization and stock assign reject them explicitly. No Physics conversion. Declared public rejection routes passed bounded guard qualification. Direct private algorithm/setup calls have no advertised interoperability; broader failure/bypass assumptions remain untested. No vector/tensor, Euler/CN/TimeFlux, stress, turbulence/MRF, MPI/GPU/Float32, solves/reference/relaxation, histories, motion/ALE or analytical accuracy claim.

Geometry/context rebuild explicitly creates new handles; old handles remain unchanged. Corrected/uncorrected contexts have independent caches. Stage2 implementation was separately approved; its scalar equations and epochs passed separate bounded qualification below. It does not inherit stage1 or earlier prototype qualification.

## Stage2 scalar algebra — bounded qualification

The separately approved stage2 implementation adds these namespace-local callables:

```
ncc_equation(context, p; form, convection=nothing, q=nothing, mobility=nothing, source, boundaries)
ncc_update!(equation; p, q, mobility, source, boundaries)
ncc_assemble!(equation)
ncc_trace_gradient(equation)
ncc_correction(equation, trace_gradient)
ncc_complete!(equation, correction)
ncc_flux(equation, correction=nothing)
ncc_pressure_component(equation, correction=nothing)
ncc_recover(component; predictor, previous)
```

Forms: `:convection`, `:positive_diffusion`, `:negative_diffusion`, `:convection_minus_diffusion`. Convection requires existing XCALibre `Linear()` or `Upwind()` and retained q; diffusion requires mobility, with paired values checked to 1e-12 scaled. Irrelevant q/mobility/scheme arguments are rejected. Inputs are same-context owned scalar/q/mobility/boundary handles; source is a copied finite cell `Vector{Float64}`. No regenerated velocity flux or implicit mobility interpolation.

Rows are integrated: `A*p - b_full + V*s = integrated_div(F)`. Positive diffusion has face flux `k*(pn-po)+E`; mixed convection-minus-diffusion has diffusion sign -1. Compact source includes `V*s` once. Completion uses `b_full=b_compact-sigma*integrated_div(E)` without changing A. Fixed-value physical convection uses prescribed trace for both schemes; zeroGradient physical diffusion is zero. Both coupled records contribute their own owner rows, while internal faces contribute opposite owner/neighbour rows. Duplicate incidences accumulate into unique CSR slots.

Construction/update leave state `:unassembled`; scratch assembly publishes `:compact`. Trace/Gauss gradient and explicit correction are pure producers. Completion publishes `:complete` once; full flux requires completed state and, for corrected diffusion, the applied correction handle. Update/reassembly advance the epoch and invalidate old trace/correction/component handles; equation identity is also checked. Geometry/connectivity/form/scheme changes require explicit rebuild. No reference row, solve, relaxation or mesh conversion is supplied.

Pressure components require **negative diffusion only, exact zero source**. Recovery adds the frozen negative-D component flux to a supplied same-context retained predictor. `previous` must exactly equal the component's frozen scalar operand: it is **not arbitrary previous-time history**, and no D(previous) subtraction is performed. Recovery is independently implemented standalone algebra, not a call to or certificate for stock `correct_mass_flux!`, PISO/Euler or motion history. Output handles and snapshots own their face values and integrated divergence.

Unregistered `test/unit/NCC/test_operators.jl` declares an exact set of 916 markers: 144 paths, 432 steps, 72 pressure controls, 144 composition controls, 24 rebuilds, 24 old-state preservation controls, 52 rejections, 9 arithmetic-overflow rejections, 1 uncorrected zero-completion identity control and 14 ownership controls. The separately approved qualification passed **6,571 Julia assertions per thread** (threads1/4) and the exact 916-marker set with multiplicity one. Independent external analysis passed **291,032 certificate checks with zero misses**, parsed all **90,480 emitted numerical values as finite**, and verified **1,275 preservation entries**. Per thread there were 42,360 numeric snapshot leaves, 5,112 metadata leaves and 720 checks. Full uncompressed tables matched bitwise between threads; maximum independent formula discrepancy was 2.22e-16 on the tested inputs.

Stage2 evidence is in the enclosing development workspace, outside this repository:

- `xcalibre-foundation-ncc-b2b-2-qualification-summary.md`;
- `validation/foundation-ncc/results/b2b-2-operators/gates.json` and compressed tables;
- `validation/foundation-ncc/logs/b2b-2-operators/`.

Qualification executed with Julia1.13.0, CPU Float64/Int32, BLAS1 and the copied external environment; no dependency resolution, repairs or reruns. Resource gates passed. Import/global-binding world-age warnings remain recorded; future Julia/Revise compatibility is not established. Marker-only rejection/rebuild/ownership/overflow controls are Julia assertion evidence, not full external snapshots of every original test object. The external numerical certificate separately repeats main, pressure and composition paths.

Static review found the original uncorrected completion-overflow target unreachable because E is zero. Implementation stopped, and the owner approved replacing that one control prospectively with an exact zero-completion identity check. No run evidence was relabelled and no tolerance widened. Exact marker sets are asserted independently of cardinality in the test. See enclosing-workspace `xcalibre-foundation-ncc-b2b-2-implementation-summary.md` and retained implementation-progress report. Storage/API/fixture compatibility and arithmetic targets passed this bounded run; this does not establish native solver integration, generated mobility/geometry or analytical accuracy. Noncollinear fixtures exercise correction but not the 0.05 cap or fallback. No closed physical-domain, full-suite regression, history/ALE, vector/GPU/MPI or legal certificate is implied.

## Frozen reference parity — B2b-3 V2

The actual public facade passed comparison against existing frozen exports from Foundation dev `fed424ace4164a7685a5035d43fff7ba03f76a0e`, using imported geometry, retained q and interpolated mobility. All nine fixtures and five probes were required. Per thread: 18 geometry-cache paths, 180 direct operator paths, 180 composed C−D paths and 180 restricted component-recovery paths, totaling **558 exact markers**, with **759,020 array rows and 720 checks**.

Independent external analysis passed **4,579,762 certificate checks with zero misses**, parsed **1,522,360 numerical values as finite** across threads1/4 and verified **1,315 preservation entries**. Maximum absolute snapshot/reference discrepancy was **6.55e-15** on tested inputs. Full uncompressed arrays/checks/markers matched bitwise between threads; resource gates passed.

Evidence outside this repository:

- `xcalibre-foundation-ncc-b2b-3-qualification-v2-summary.md`;
- `validation/foundation-ncc/results/b2b-3-parity-v2/gates.json` and compressed tables;
- `validation/foundation-ncc/logs/b2b-3-parity-v2/`.

The original failed run is retained. Its external adapter incorrectly made residual interface patches fixedValue in mode4; the reference changes only the patch named `physical`. A separately approved adapter-only repair retained residual ZeroGradient, followed by a versioned rerun with unchanged coverage/tolerances. No production numerical repair was needed. Prospective storage amendments allowed full tables and preserved failure evidence; historical certificates were not rewritten.

These are stationary full-depth0.1 scalar operator comparisons with both coupled sides and tiny residuals retained. C−D targets compose separate existing exports, not a fresh Foundation mixed solve. Recovery adds negative-D component flux to a convection predictor with `previous=p`, not stock solver recovery or time history. Imported q/mobility/geometry do not certify production generation. Fidelity and bitwise determinism do not establish analytical superiority, stability or parallel speedup. No native Mesh3/Physics/ModelEquation solver integration, generated intersections, ordinary full-suite regressions, unit-depth/cost, vector stress, transient/history/ALE/GCL, GPU/MPI or legal readiness is implied. S1 native scalar qualification is recorded separately below; it is not inferred from this facade result.

## S1 native frozen scalar bridge — bounded V5 qualification

The separately approved S1 implementation provides a marked `StationaryNCCMesh <: AbstractMesh` owning real `Mesh3` numerical storage through `StationaryNCCBinding(context)`. It is **not a public bare Mesh3**, a production intersection generator, or admission to stock Physics/flow solvers. Earlier facade certificates do not qualify this bridge. S1 has now imported and executed successfully in the separate external V5 qualification, after four preserved failed attempts and individually approved repairs. This qualifies only the declared frozen stationary CPU Float64/Int32 scalar scope.

Namespace-local construction:

```julia
binding = NCC.StationaryNCCBinding(context)
mesh = NCC.StationaryNCCMesh(binding)
p = NCC.ncc_native_scalar(mesh; values=cell_values)
q = NCC.ncc_native_face_scalar(mesh; values=face_values, order=:input)
bcs = NCC.ncc_native_boundaries(mesh; physical=physical_specs)
eqn = NCC.ncc_native_equation(mesh, p; form, convection, q, mobility, source, boundaries=bcs)
```

Here `source` is an actual same-domain ScalarField; q/mobility are actual FaceScalarFields when required by the selected form. `order` must be `:input` or `:native`. Boundary-first ordering retains ascending patch/local IDs followed by internal local IDs; both coupled sides remain boundary records with separate remote-owner addressing. No cell reordering, fabricated polygons or writer support. Physical polygon metadata remains provenance only. Binding copies and revalidates the context with a new identity. Numerical geometry is read-only by contract, not immutable or reflection-protected.

Assembly uses native Model/Operator/Src/ModelEquation and real ScalarEquation CSR, native internal incidences and boundary-owner gathers; no facade matrix is copied into native storage. Forms/schemes remain the four scalar forms and Linear/Upwind. Native source sign is **+1**, compact source includes **+V*s**, preserving `A*p-b_full+V*s=integrated_div(F)`. An S0 design sign error was corrected with explicit owner approval; it was not a numerical failure or alteration of qualified facade algebra.

Call `Discretise.discretise!(eqn,p,config)`, then `Discretise.apply_boundary_conditions!(eqn,bcs,nothing,nothing,config)`, with CPU hardware/workgroup metadata. States are `:unassembled → :internal → :compact`; reassembly snapshots native fields and advances the epoch. Changed inputs invalidate output reads until explicit successful reassembly. Boundary completion checks unchanged hardware and operands. Trace/gradient and correction producers are pure; `ncc_native_complete!` applies E once, retaining compact source and applied E. Subsequent `ncc_native_flux` requires completed state and the actual applied correction for corrected diffusion. The corresponding namespace APIs are `ncc_native_trace_gradient`, `ncc_native_correction`, `ncc_native_complete!`, `ncc_native_flux`, `ncc_native_pressure_component`, and `ncc_native_recover`. Recovery remains negative-D/exact-zero-source/frozen `previous=p` only, not solved-pressure provenance or a SIMPLE sweep.

Screened scalar interpolation/boundary correction, scalar Gauss and face-scalar divergence use the same marked domain. Gauss/divergence reuse existing MPL kernels after finite/ownership preflight. Assembly and correction are deterministic serial host loops; there is no parallel performance claim. Snapshots copy numerical storage; flux snapshots include a copied native face field attached to the same read-only-by-contract domain.

Stock assignment is refused via a separately approved **no-op-default positional preflight hook** in `Discretise_4_assign_boundaries.jl`; Julia cannot specialize on keyword `region`. Stock solver/solve/recovery, Physics conversion, Adapt including CPU clone, partition/distribution and writers remain refused. Selected Physics dispatch intersections passed V5 refusal controls, including 20 dispatch-only typed `run!` probes bundled in one control; no stock solver body was executed and no global ambiguity-free claim is made. Unknown private/reflection routes are not a security guarantee; general vector/tensor/history or gradient-scheme interoperability is not advertised.

`test/unit/NCC/test_native_scalar.jl` is new and deliberately **unregistered and unexecuted**. It contains independently authored small dense-oracle controls, duplicate/coincident links, same-nnz rewiring, translations, tiny residuals, empty/error patches, three dirty assemblies, both policies, forms/schemes, stale/rejection/ownership and restricted recovery controls. It is implementation test material, **not a declared external qualification coverage certificate**. The separately approved external harness exercised its own prospectively declared transactional overflow/rejection, ownership, reference/facade, full-depth/unit-depth, selected refusal, ordinary regression and cost coverage. It did not execute this implementation test file or audit every possible guard intersection. No legal readiness, solver readiness or analytical accuracy certificate is inferred.

### S1 V5 evidence and retained failures

Per thread (threads1/4): **3,215 exact markers, 3,133,020 array leaves, 4,063 checks and 24 cost records**. Marker groups: 2,196 reference, 888 synthetic, 128 controls and three ordinary regressions. All nine frozen fixtures, both depths, both coupled sides and declared scalar forms/schemes were covered. Independent external analysis passed **31,260,687 certificate checks with zero misses**, parsed **6,290,610 numerical values as finite** and passed **1,387 preservation checks**. Maximum absolute oracle discrepancy was **6.821210263296962e-13**, within unchanged scaled gates (synthetic 1e-12, reference 1e-10; structural comparisons exact). Entire uncompressed arrays/checks/markers matched by SHA256 and direct byte comparison across threads; costs were not compared bitwise.

Selected ordinary regressions passed per thread: repeated links 2,052 assertions, padded-neighbour reader 18, periodic mass flux 2. This is not a full-suite certificate; the ordinary transient test does not qualify native temporal behavior. Before/after equation numeric storage was independently exported for controls; other epoch/state/ownership preservation is Julia assertion evidence, not complete external snapshots of every object.

Evidence in the enclosing workspace, **outside this repository**:

- `validation/foundation-ncc/results/s1-native-v5/gates.json`, complete compressed tables, run reports and `preserved-hashes.json`;
- `validation/foundation-ncc/logs/s1-native-v5/`;
- `validation/foundation-ncc/s1-native-v5-qualification-summary.md` and prospective `s1-native-gates.md`.

V1 failed on the external harness's direct Adapt import; V2 on typed patch snapshot reconstruction; V3 on the VectorEquation refusal ambiguity; V4 on the vector Grad refusal ambiguity. Each failure and its evidence remains archived. External repairs used the existing exported adaptation route and explicitly reconstructed typed patch inputs. The two package refusal signatures were separately tightened to the stock vector field's `S1<:ScalarField` bound, without changing refusal bodies or admitting vector operations. V3/V4 available numerical comparisons passed but incomplete runs were failures, not retrospectively qualifications.

V5 driver elapsed times were approximately 221/220s; analysis 69s; supervisor aggregate 510s. Sampled process-group RSS stayed below 2 GiB; sampling is not continuous enforcement. Compressed tables were 52,720,502 bytes under 128 MiB. All cost records met prospective time/allocation/summarysize gates; these are not solver benchmarks, factor-memory measurements or evidence of parallel speedup. The tree-only cap was prospectively amended to 320 MiB to preserve failures; logs/JSON cap remained 2 MiB. Historical budgets/certificates were not changed.

Qualification used pinned Julia1.13.0, BLAS1 and the copied environment without dependency resolution. Import/world-age warnings and stale project_hash remain recorded; future Julia/Revise compatibility is unestablished. Imported geometry/q/mobility remain frozen plumbing inputs, not certified production generation. Native vector momentum/stress, H/rD/rDf/predictor, pressure solve/gauge/reference, pressure–velocity/SIMPLE/PISO, Taylor–Green accuracy, histories/transients, generated intersections, motion/ALE/GCL and GPU/MPI remain outside S1. Source-origin/similarity/distribution review is still pending.

## Qualification and provenance

Independently authored test: `test/unit/NCC/test_contract.jl`, still deliberately unregistered in `test/runtests.jl`. The separately approved V2 qualification passed 498 assertions per thread (threads1/4), with exact 220 contract markers, 72 snapshot roots and 32 supplementary no-output guard controls per thread. External analysis passed 81,967 certificate checks with zero misses, independently parsed all 23,152 emitted numerical values as finite, and verified 753 preservation entries. Full uncompressed tables matched bitwise between threads.

Evidence lives in the enclosing development workspace, **outside this repository**, not in a package runtime dependency:

- `xcalibre-foundation-ncc-b2b-1-qualification-v2-summary.md`;
- `validation/foundation-ncc/results/b2b-1-contract-v2/gates.json` and compressed tables;
- `validation/foundation-ncc/logs/b2b-1-contract-v2/`.

The original failed qualification is preserved separately. Its test parse error was repaired by changing `1.+.1i` to `1.0 + 0.1*i`; no tolerance or coverage weakening. Validation is pinned to Julia1.13.0, CPU Float64, BLAS1 and the copied external environment, without dependency resolution. Import and harness world-age warnings remain recorded; future-version compatibility is not established.

All six **B2b-1** algebra fixtures are collinear, so corrected/uncorrected caches coincide. Those stage1 results alone do not qualify oblique/cap/fallback geometry, populated physical polygon ownership/writer lifecycle, all private solver bypasses, ordinary numerical regressions, scalar operators or pressure recovery. B2b-2 separately qualifies its declared noncollinear scalar algebra and restricted component recovery above. Earlier passed B1 external GPL validation certifies its prototype, not native integration of these production handles. Foundation frozen-export parity subsequently passed its separate B2b-3 V2 qualification below; it is not inferred from the mathematical algebra certificate.

See `ncc_provenance.md` for origins and unresolved review questions. Licence route selected by owner is independent MPL implementation with exposure disclosed, not clean-room or legal clearance. No GPL validation dependency or copied fixture/test assets.
