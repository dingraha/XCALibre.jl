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
unit/InterfaceCoupling: test_planar_interface (static nonconforming overlap geometry, coverage, deterministic conservative flux gathers), test_planar_diffusion (symmetric implicit scalar coupling, manufactured solution, flux/operator consistency), test_planar_transport (conservative upwind/linear coupling, unequal-distance interpolation, opposing overlap flows, throughflow solve), test_planar_pressure (variable mobility, conservative pressure operator, closed-domain projection)

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

This CPU-only boundary uses shared overlap conductances `gamma*length/(distance_a+distance_b)`. It supports scalar `Laplacian{Linear}` with positive constant diffusivity and ignores boundary contributions from `Time` terms. It does not provide convection, vector/pressure coupling, variable diffusivity, mesh motion, or nonorthogonal interface corrections. On skew owner pairs, the two-point approximation is exact only for fields linear in the normal direction. Rebuild boundaries and equations after geometry or ID changes; initialise preconditioners only after constructing the extended matrix.

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

Support is CPU-only, scalar `Divergence{Upwind}` or `Divergence{Linear}`, optional positive constant `Laplacian{Linear}`, and `Time` boundary terms. Linear convection uses overlap owner weights `distance_b/(distance_a+distance_b)` and `distance_a/(distance_a+distance_b)`, consistent with geometric interpolation. This is exact for normal-linear fields but provides no tangential reconstruction on skew cell pairs. Central/linear convection is not inherently bounded and may produce oscillations in convection-dominated problems. Higher-order convection, pressure correction through `PlanarTransport`, vector coupling, and mesh motion are not supported.

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

This supports CPU scalar `Laplacian{Linear}` and boundary `Time` contributions at unit density only. It is **not** integrated into SIMPLE/PISO: vector momentum coupling, pressure-gradient reconstruction, predictor construction, density weighting, nonorthogonal corrections, moving geometry, GPU, and MPI are not yet supported. Rebuild boundaries/equations after geometry or ID changes.

## Remaining test index
unit/Multithread: test_kernel_index_pattern (source scan: `@index` bound on its own)
unit/Preprocess: test_setFields (region initialisation)
unit/Calculate: test_wall_distance, test_least_squares, test_venkat_limiter
unit/Discretise: test_physical_boundary_conditions, test_laplace (analytic), test_neumann_sign, test_robin, test_nonorthogonal_flux (flux continuity), test_boundary_accumulation (deterministic sums), test_linear_upwind, test_lustdc, test_lustlu, test_crank_nicolson (temporal order)
unit/Solve: test_smoothers, test_DILU, test_amg_freeze, test_AMG, test_AMG_matrices, test_xvector, test_bicgstab_shadow, test_krylov_breakdown, test_solve_warning, test_mixed_precision, test_stop_criterion (one stopping rule on every path), test_residual_parity (same outer residual on every linear-solve path), test_preconditioner_paths (preconditioner per solve path)
unit/Solvers: test_reconstruct (face fluxes), test_potential_flow, test_run_continuation, test_compressible_mass_flux, test_consistent_coupling (SIMPLEC), test_dt_consistent_flux, test_periodic_mass_flux, test_film_dt_consistency
unit/ModelPhysics: test_wall_function_* and test_wall_production_density (wall functions), test_viscosity_models, test_parallel_cell_updates, test_lke_diffusion, test_lke_production, test_transpose_stress, test_viscous_work, test_kinetic_energy_flux, test_thermo_psi_time, test_fluidProperties (NIST), test_turbulence_float_types
unit/Postprocess: test_field_average, test_field_rms, test_reynolds_stress, test_qcriterion_window, test_post_process_transient, test_post_process_steady, test_viscous_force_bcs, test_probe
cases: test_2d_laplace_*, test_transient_verification_examples (runs examples/transient_verification/heat_decay.jl), test_adaptive_dt, test_*incompressible* (pitzdaily_LES: KEquation+Smagorinsky; transient_laminar_BFS: Euler+CrankNicolson), test_2d_taylor_couette_laminar, test_*compressible* (flatplate files hold viscosity/wall-BC variants), test_2d_godunov_supersonic_cylinder (+ _RANS), test_2d_multiphase_*, test_2d_EFM
distributed: gate.jl (MPI gate: one 2-rank gate_session.jl including each file in its own module; test_halo, test_halo_subcomm and test_turbulence_sst_wallfn at 3 ranks, skipped unless `XCALIBRE_TEST_MAX_RANKS` >= 3, which CI sets). Other files and rank counts: `runtests_mpi.jl`

## Not registered (known issues)
- unit/ModelPhysics/test_mixture_multiphase.jl: needs `examples/0_GRIDS/unit_test_stage2.unv` (missing)
