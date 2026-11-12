# XCALibre.jl tests

## Layout
- `runtests.jl`: tier selector + explicit ordered file registry (`TIERS`), the single index of what runs
- `unit/<Module>/test_<topic>.jl`: tiny-mesh checks; folder = `src/` module under test
- `cases/test_<case>.jl`: short end-to-end simulations with physics assertions
- `distributed/`: MPI tier, entry point `gate.jl`
- `gpu/test_<topic>.jl`: CPU/GPU comparisons, not in a tier (need a GPU): `julia --project=<env with XCALibre and CUDA> test/gpu/test_<topic>.jl`
- `grids/`, `CBC_Data/`, `Fluids_NIST_Data/`: test data

## Naming
`test_<topic>.jl`; a file not listed in `TIERS` never runs.

## Running
- All tiers (default): `] test XCALibre`
- Select: `Pkg.test("XCALibre"; test_args=["unit"])` or `["cases", "distributed"]`
- Env: `XCAL_TEST_TIER=unit,cases`; direct: `julia --project=test test/runtests.jl unit`
- Unknown tier names error with the valid list; per-file times and a sorted summary print at the end

## Adding a test
1. Create `unit/<Module>/test_<topic>.jl` (or `cases/test_<case>.jl`)
2. Add its path to the tier in `runtests.jl`
3. Load grids via `pkgdir(XCALibre, ...)`; `workgroupsize(mesh)` comes from `runtests.jl`

## Budget and conventions
- CI, no coverage, one parallel job per tier (unit and cases on Julia 1.10 and the latest release, distributed on the latest release only). Job times, 1.10 / latest: unit ~13 / 15 min, cases ~10 / 13 min, distributed ~11 min; the slowest job sets the wall-clock. About 70% of test time is compilation, so cutting steps in a file rarely helps: cut distinct specialisations instead
- Most unit files take seconds (solver-based checks such as test_crank_nicolson up to ~1 min); each case under ~15 s on a desktop CPU
- Unit: tiny meshes, assert the property (exactness, conservation, sign, types)
- Cases: assert physics (convergence, reference values), not just "it ran"

## Index (tier -> folder -> file)
unit/Mesh: test_mesh_conversion (readers, writers), test_foam_padded_neighbour, test_mesh_reorder, test_face_orientation, test_bounding_box, test_boundary_distance (boundary face delta), test_unv2d_geometry (centroids, face geometry consistency)
unit/InterfaceCoupling: test_circular_momentum (vector Upwind/corrected diffusion, conservative recovery, integrated pressure force, physical predictor augmentation, boundary rejection and manufactured viscous iteration), test_circular_pressure (boundary-aware complete trace chain, variable harmonic mobility, frozen sparse assembly/recovery, standalone deferred projection, component compatibility/gauges and radial solve refinement), test_circular_reconstruction (interface-aware deduplicated LSQ, shared affine traces, frozen constant-coefficient corrected flux/operator consistency, radial refinement and validation), test_circular_metrics (independent curved-volume snapshot, exact affine pressure-area gradients, moment/volume identities, stale geometry and atomic validation), test_circular_interface (geometry-only full-circle angular overlaps, exact arc vectors/moments, conservative gathers and frozen alignments; no curved solver/ALE), test_planar_interface (static nonconforming overlap geometry, coverage, deterministic conservative flux gathers), test_planar_diffusion (symmetric implicit scalar coupling, manufactured solution, flux/operator consistency), test_planar_transport (conservative upwind/linear coupling, unequal-distance interpolation, opposing overlap flows, throughflow solve), test_planar_pressure (variable mobility, conservative pressure operator, closed-domain projection), test_planar_momentum (Cartesian vector coupling, component consistency, convection–diffusion solves), test_planar_predictor (Gauss pressure traces, overlap predictor, momentum-to-projection chain)

unit/IOFormats: test_foam_writer_unset_bcs

### Static scalar interface diffusion

Construct paired boundaries after final mesh reordering and assign both to the same scalar field:

```julia
pair = construct_planar_diffusion(mesh, :interface_a, :interface_b)
BCs = assign((T=[pair..., Zerogradient(:outside)],); region=mesh)
phi = ScalarField(mesh)
gamma = ConstantScalar(2.0)
eqn = (-Laplacian{Linear}(gamma, phi) == Source(ConstantScalar(0.0))) → ScalarEquation(phi, BCs.T)
```

This CPU-only boundary uses shared overlap conductances `gamma*length/(distance_a+distance_b)`. It supports scalar or Cartesian-vector `Laplacian{Linear}` with positive constant diffusivity and ignores boundary contributions from `Time` terms. It does not provide convection, pressure coupling, variable diffusivity, mesh motion, or nonorthogonal interface corrections. On skew owner pairs, the two-point approximation is exact only for fields linear in the normal direction. Rebuild boundaries and equations after geometry or ID changes; initialise preconditioners only after constructing the extended matrix.

Recover physical integrated overlap fluxes (positive outward from A) with `interface_diffusion_flux!(flux, pair[1], phi, gamma)` and gather them using the geometry in `pair[1].value.coupling`. Flux recovery is for diagnostics; do not add these contributions to a balance that already includes the assembled interface operator.

### Static scalar interface transport

Use transport descriptors instead of diffusion-only descriptors when the scalar equation includes convection:

```julia
pair = construct_planar_transport(mesh, :interface_a, :interface_b)
coupling = pair[1].value.coupling
transport = InterfaceTransportFlux(mesh, coupling)
# Fill ordinary interior and exterior faces with integrated transport fluxes.
# Example: prescribed unit velocity in the x direction (unit density).
transport.values .= mesh.face_area .* [n[1] for n in mesh.face_normal]
update_interface_flux!(transport, [S[1] for S in coupling.area_vector])
BCs = assign((T=[pair..., Zerogradient(:outside)],); region=mesh)
phi = ScalarField(mesh)
eqn = (Divergence{Upwind}(transport, phi) - Laplacian{Linear}(ConstantScalar(0.4), phi) == Source(ConstantScalar(0.0))) → ScalarEquation(phi, BCs.T)
```

Choose actual inlet/outlet boundaries for a throughflow solve; the exterior condition above only illustrates assignment. Use a nonsymmetric linear solver, such as `Bicgstab()`, for convection.

Overlap fluxes are integrated volume or mass fluxes, not densities, and are positive outward from A. Call `update_interface_flux!` after changing prescribed overlap fluxes; it also refreshes physical interface face sums without changing other faces or the sparse graph. Assembly uses overlap fluxes individually, so opposing flows on one physical face retain distinct upwind donors. The flux field and boundaries must reference the same coupling object. Initially one coupling per transport field is supported.

Use `Divergence{Upwind}` or `Divergence{Linear}` in the equation. Recover transported scalar fluxes with `interface_transport_flux!(out, pair[1], phi, transport; scheme=Upwind())` or `scheme=Linear()` to match that equation; existing calls without a keyword default to upwind. Both paired descriptors give the same A-oriented result. Gather for diagnostics only, not as an extra source. `correct_boundaries!` supplies geometric interpolation, not an upwind flux trace. Interface transport has zero column sums (conservation); constant-state preservation additionally requires divergence-free complete cell flux balances.

Support is CPU-only, scalar or Cartesian-vector `Divergence{Upwind}` or `Divergence{Linear}`, optional positive constant `Laplacian{Linear}`, and `Time` boundary terms. Linear convection uses overlap owner weights `distance_b/(distance_a+distance_b)` and `distance_a/(distance_a+distance_b)`, consistent with geometric interpolation. This is exact for normal-linear fields but provides no tangential reconstruction on skew cell pairs. Central/linear convection is not inherently bounded and may produce oscillations in convection-dominated problems. Higher-order convection, pressure correction through `PlanarTransport`, and mesh motion are not supported.

### Static Cartesian-vector momentum

Use the same paired `PlanarTransport` boundaries with `VectorField` and `VectorEquation` for prescribed-flux convection and constant-viscosity diffusion. Each Cartesian component uses the same conservative scalar coefficients; both sides must use a common stationary Cartesian frame. `PlanarDiffusion` also supports vector diffusion without convection.

```julia
U = VectorField(mesh)
BCs = assign((U=[pair..., Zerogradient(:outside)],); region=mesh)
zero_source = VectorField(mesh)
initialise!(zero_source, SVector(0.0, 0.0, 0.0))
U_eqn = (Divergence{Upwind}(transport, U) - Laplacian{Linear}(ConstantScalar(0.4), U) == Source(zero_source)) → VectorEquation(U, BCs.U)
momentum_flux = zeros(SVector{3,Float64}, length(coupling.measure))
interface_transport_flux!(momentum_flux, pair[1], U, transport; scheme=Upwind())
interface_diffusion_flux!(momentum_flux, pair[1], U, ConstantScalar(0.4)) # overwrites with diffusion only
```

Import `StaticArrays` for `SVector` and choose the mesh floating type for output storage. Vector recovery outputs can be passed directly to `interface_cell_flux!` and `interface_face_flux!` with matching vector-valued destination arrays. Do not count both gathers as equation contributions. `correct_boundaries!` supports `FaceVectorField` geometric traces; this is not the upwind transported state or an incompressible predictor-flux implementation.

The normal `Mesh2` vector solve advances x and y only; z assembly/interpolation/recovery are supported but z is not solved by that path. Least-squares reconstruction has not been made interface-aware, and no general tangential-linear accuracy is claimed. Variable viscosity, pressure-gradient/predictor integration, SIMPLE/PISO, and moving-frame transformations remain outside this vector increment.

## Static planar pressure projection (CPU primitive)

```julia
pair = construct_planar_pressure(mesh, :interface_a, :interface_b)
coefficient = InterfacePressureCoefficient(mesh, pair[1])
update_interface_coefficient!(coefficient, rD) # cell mobility rD = volume / momentum diagonal
BCs = assign((p=[pair..., Zerogradient(:outside)],); region=mesh)
p_eqn = (-Laplacian{Linear}(coefficient, p) == -Source(predicted_divergence)) → ScalarEquation(p, BCs.p)
# Assemble and solve with a pressure reference; initialise preconditioners after matrix extension.
# Hold coefficients fixed between assembly and pressure-flux correction.
predicted_overlaps = copy(transport.overlaps)
correct_interface_pressure_flux!(transport, predicted_overlaps, pair[1], p, coefficient)
```

`InterfacePressureCoefficient` uses ordinary mesh interpolation internally and arithmetic normal-distance interpolation on overlaps. The shared overlap conductance is `G = interpolated_rD*length/(distance_a+distance_b)`. Both pressure assembly and correction use this stored value, not summed sparse matrix entries. Refresh coefficients and reassemble whenever cell mobilities change; the static interface graph need not change. Its interface face values are length-weighted coefficient averages, not flux sums. Do not modify its storage arrays directly.

`interface_pressure_flux!(out, pair[1], p, coefficient)` recovers `G*(pA-pB)` oriented outward from A, for either descriptor. The correction operation sets (rather than increments) overlap fluxes from a separately saved predictor and refreshes signed physical face sums. Exterior/interior faces are untouched: callers must also correct ordinary face fluxes with the corresponding assembled pressure coefficients. Full divergence can then be calculated from the physical face fluxes; do not add the overlap gather again. A closed-domain Neumann solve requires a compatible globally balanced predictor and a pressure reference.

This supports CPU scalar `Laplacian{Linear}` and boundary `Time` contributions at unit density only. It is **not** integrated into SIMPLE/PISO. The Gauss-gradient and overlap predictor primitives are described below; interface-aware least-squares reconstruction, density weighting, nonorthogonal corrections, moving geometry, GPU, and MPI are not yet supported. Rebuild boundaries/equations after geometry or ID changes.

### Shared geometry and momentum-predictor projection

Construct pressure descriptors from the momentum pair when convection and pressure correction use the same transport field:

```julia
momentum_pair = construct_planar_transport(mesh, :interface_a, :interface_b)
pressure_pair = construct_planar_pressure(momentum_pair)
transport = InterfaceTransportFlux(mesh, momentum_pair[1].value.coupling)
coefficient = InterfacePressureCoefficient(mesh, pressure_pair[1])
# After momentum assembly/solve and removal of its pressure source:
inverse_diagonal!(rD, U_eqn, config) # rD = volume / aP
H!(Hv, U, U_eqn, config) # Hv is H/aP, already velocity-like
interpolate!(Uf, Hv, config)
correct_boundaries!(Uf, Hv, config.boundaries.U, time, config)
face_flux = FaceScalarField(transport.values, mesh) # wrapper, no copy
flux!(face_flux, Uf, config) # ordinary predicted faces
predicted_overlaps = zeros(eltype(transport.overlaps), length(transport.overlaps))
interface_predictor_flux!(predicted_overlaps, momentum_pair[1], Hv)
update_interface_flux!(transport, predicted_overlaps) # replace interface face sums
predicted_faces = copy(transport.values)
div!(predicted_divergence, face_flux, config)
update_interface_coefficient!(coefficient, rD)
# Assemble/solve the pressure equation; correct ordinary faces with their own
# conductances and saved predicted_faces before correcting the interface:
correct_interface_pressure_flux!(transport, predicted_overlaps, pressure_pair[1], p, coefficient)
grad!(gradp, pf, p, config.boundaries.p, time, config) # Grad{Gauss}
correct_velocity!(U, Hv, gradp, rD, config)
```

This is a lifecycle sketch, not a complete solver: allocate fields/configuration and perform the pressure solve and ordinary-face corrections explicitly. For ordinary interior faces use `coefficient.values[f]*mesh.face_gDiff[f]*(p[owner]-p[neighbour])`, added to the saved predicted flux. Exterior correction must match its pressure boundary condition; closed zero-normal-flow/Neumann fixtures need no exterior correction. Avoid accumulating corrections or counting overlap gathers in addition to physical face sums.

`interface_predictor_flux!` writes only its output array, for either descriptor side, with `(wA*HvA+wB*HvB)⋅S_overlap` oriented outward from A. It preserves individual opposing flows and includes no density, pressure, or mesh velocity. Do not multiply `Hv` by `rD` again. Shared descriptors can be constructed before or after assignment. Independent momentum/pressure constructors produce distinct coupling objects and cannot be used interchangeably in pressure correction.

For this straight interface, length-averaged geometric pressure traces passed to the existing Gauss path reproduce overlap pressure-area sums. Constant pressure and normal-linear fields with consistent exterior values are tested. Tangential reconstruction and interface-aware least-squares remain unsupported. Conservative pressure-corrected face fluxes satisfy continuity; interpolating the corrected cell velocity generally does **not** reproduce those fluxes on a collocated mesh. Preserve corrected overlap/face flux storage for convection rather than replacing it with velocity-interpolated fluxes.

The tested primitives remain CPU-only, static, unit-density, and constant-viscosity. The restricted SIMPLE entry point below integrates them; ordinary `run!` and transient PISO do not support these interfaces.

### Restricted static planar SIMPLE

`test/unit/Solvers/test_planar_simple.jl` exercises the first solver integration: nonconforming constant throughflow and viscous channel flow, comparison with ordinary conforming SIMPLE, integrated continuity, all-Neumann compatibility, validation errors, split-run continuation and reinitialisation.

Build momentum descriptors and pressure descriptors sharing their geometry as above, then `assign` them alongside exterior boundaries in `config.boundaries` before creating state:

```julia
state = PlanarSimpleState(model, config; continuity=1e-9)
residuals = planar_simple!(state; iterations=500)
@show residuals.converged residuals.continuity[end]
# Continue using the same state: individual corrected overlaps are retained.
more = planar_simple!(state; iterations=100)
```

Requires steady CPU `Mesh2`, laminar/isothermal incompressible physics, unit density, constant positive viscosity, Upwind momentum convection, Linear diffusion and unlimited Gauss gradients. Prescribed velocity (`Dirichlet`/`DirichletFunction`, zero z) requires pressure `Zerogradient`; velocity `Zerogradient`/`Outlet` permits fixed `Dirichlet` or homogeneous `Zerogradient` pressure. A velocity zero-gradient boundary is not an impermeable wall: prescribe zero velocity for a no-slip wall. No additional momentum sources, transpose-stress correction, SIMPLEC, nonorthogonal correction, output/postprocessing, file restart, transient stepping, or moving geometry. Set `write_interval=-1` and leave postprocessing unset. Unsupported options/boundaries are rejected. All-Neumann pressure requires `pref` and a globally balanced predictor; fixed-pressure boundaries must not be combined with `pref`.

The state allocates extended equations before initialising preconditioners, refreshes mobility/conductance before each pressure assembly, and corrects fluxes using **unrelaxed** pressure before relaxing pressure for Gauss velocity correction. Interface correction runs once per coupling. Convergence requires momentum residual targets, a pressure residual target (normalised or absolute), and a continuity tolerance. `continuity` is the maximum absolute **integrated** cell volume-flow imbalance, not volume-normalised divergence. `p_absolute` reports the L1 initial pressure-equation residual; allowing it below the same tolerance avoids a normalised residual stagnating near a zero-pressure solution.

Histories contain executed iterations only. The state retains cumulative iteration time across calls. Changed cell velocity/pressure invalidates continuation and rebuilds initial fluxes; failures also invalidate it. Rebuild state after mesh, boundary, geometry or configuration changes, and do not mutate its working flux arrays. Physical-face diagnostics are copied to `model.momentum.mdotf`, but cannot recreate overlaps. Use `state.work.transport` for overlap diagnostics, and retain the same state for continuation—not ordinary `run!`.

### Static interface verification examples and robustness

`examples/interface_verification/planar_channel.jl` is a runnable, mesh-file-free channel example demonstrating shared descriptors and stateful continuation. `channel_refinement.jl` compares three nonconforming resolutions with conforming SIMPLE and exercises a stronger mismatch. See `examples/interface_verification/README.md` for commands, metrics, measured results and limitations. The case-tier test `test/cases/test_planar_verification_examples.jl` executes both scripts in isolated modules. Spatial pressure-drop errors are nonmonotonic on coarse grids; no general second-order claim is made.

`test/unit/Solvers/test_planar_simple_robustness.jl` checks initial-state and relaxation variations, signed Couette flow with positive and negative overlaps, all-Neumann pressure-offset invariance, and fresh-state-equivalent recovery after validation and mid-iteration errors. Every failed call now invalidates continuation, including errors before assembly. The restricted solver now removes the complete momentum diagonal augmentation and old-velocity source before projection, using the physical pre-relaxation diagonal for both the predictor and mobility. Tested momentum/pressure relaxation variations must agree in cell fields, ordinary face fluxes and individual overlaps to tight tolerances; ordinary SIMPLE remains unchanged. `test/unit/Solvers/test_planar_physical_predictor.jl` checks the reconstruction against independently reassembled unrelaxed sparse rows, including interface neighbours and diagonal-dominance augmentation at relaxation 1. Positive identical x/y physical diagonals and relaxation factors in (0,1] are required.

## Remaining test index
unit/Multithread: test_kernel_index_pattern (source scan: `@index` bound on its own)
unit/Preprocess: test_setFields (region initialisation)
unit/Calculate: test_wall_distance, test_least_squares, test_venkat_limiter
unit/Discretise: test_physical_boundary_conditions, test_laplace (analytic), test_neumann_sign, test_robin, test_nonorthogonal_flux (flux continuity), test_boundary_accumulation (deterministic sums), test_linear_upwind, test_lustdc, test_lustlu, test_crank_nicolson (temporal order)
unit/Solve: test_smoothers, test_DILU, test_amg_freeze, test_AMG, test_AMG_matrices, test_xvector, test_bicgstab_shadow, test_krylov_breakdown, test_solve_warning, test_mixed_precision, test_stop_criterion (one stopping rule on every path), test_residual_parity (same outer residual on every linear-solve path), test_preconditioner_paths (preconditioner per solve path)
unit/Solvers: test_circular_wall_pressure (affine prescribed normal derivatives, compatibility/flux recovery, impermeability, gauge/continuation and Couette comparison), test_circular_simple (restricted static array-based coupling, constant flow/outlet, Couette relaxation/gauges, continuation/failure recovery and fixed-domain refinement), test_reconstruct (face fluxes), test_potential_flow, test_run_continuation, test_compressible_mass_flux, test_consistent_coupling (SIMPLEC), test_dt_consistent_flux, test_periodic_mass_flux, test_film_dt_consistency
unit/ModelPhysics: test_wall_function_* and test_wall_production_density (wall functions), test_viscosity_models, test_parallel_cell_updates, test_lke_diffusion, test_lke_production, test_transpose_stress, test_viscous_work, test_kinetic_energy_flux, test_thermo_psi_time, test_fluidProperties (NIST), test_turbulence_float_types
unit/Postprocess: test_field_average, test_field_rms, test_reynolds_stress, test_qcriterion_window, test_post_process_transient, test_post_process_steady, test_viscous_force_bcs, test_probe
cases: test_2d_laplace_*, test_transient_verification_examples (runs examples/transient_verification/heat_decay.jl), test_adaptive_dt, test_*incompressible* (pitzdaily_LES: KEquation+Smagorinsky; transient_laminar_BFS: Euler+CrankNicolson), test_2d_taylor_couette_laminar, test_*compressible* (flatplate files hold viscosity/wall-BC variants), test_2d_godunov_supersonic_cylinder (+ _RANS), test_2d_multiphase_*, test_2d_EFM
distributed: gate.jl (MPI gate: one 2-rank gate_session.jl including each file in its own module; test_halo, test_halo_subcomm and test_turbulence_sst_wallfn at 3 ranks, skipped unless `XCALIBRE_TEST_MAX_RANKS` >= 3, which CI sets). Other files and rank counts: `runtests_mpi.jl`

## Not registered (known issues)
- unit/ModelPhysics/test_mixture_multiphase.jl: needs `examples/0_GRIDS/unit_test_stage2.unv` (missing)
