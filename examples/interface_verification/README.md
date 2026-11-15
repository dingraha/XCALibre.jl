# Static planar-interface verification

## Filled circle in a periodic square (geometry only)

`circular_square_setup.jl` defines `CircularSquareGeometry.build(; na=16, nb=20,
nra=3, nrb=3, phase=0.0, L=1.0, radius=0.25, core_half=0.10)` without running a
simulation. It returns a `Mesh2`, exact circular coupling/metrics, resolved exterior
periodic descriptors and inner/outer cell ranges. Angular counts must be multiples
of four. A rotated square tensor core fills the centre; O-grids connect it to the
circle and the circle to the stationary outer square. Phase rotates the whole inner
zone only. Matching counts/zero phase provide a matched **split** control.
`build(; na=16, nb=16, stitched=true)` instead reuses identical circle node IDs,
removes interface patches and makes ordinary internal chord faces. Stitched mode
requires equal counts and zero phase. Its cell centres agree with the matched split
mesh, but chord-cell volumes differ from exact-arc corrected volumes.

The example assembles polygon owners from exact node-ID incidence because stock
multiblock connections support opposite local edge slots, whereas core/O-grid joins
also use perpendicular slots. This is an example-local construction, not a production
mesher change. No coordinate-proximity node merging or geometry fallback is used.

Tests in `test/cases/test_circular_square_geometry.jl` verify signed cell areas,
strict convexity, topology, filled centre, periodic pairs, full circle coverage,
corrected zone volumes and affine surface moments at three refinements/phases.
Nonmatching polygonal zones have chord gaps/overlaps; exact arc metrics, not raw
polygon volumes, represent the intended square partition. Cell centres remain
point samples. Rebuild snapshots after any geometry/ID change.

**No flow solve, transient coupling, motion or ALE is implemented by this helper.**
The split mesh must not be passed to ordinary PISO as though circular patches were
already coupled.

### Static all-periodic circular scalar operators

`construct_periodic_circular_geometry(mesh, coupling, pairs)` builds an independent
CPU snapshot. `pairs` contains `(face_a, face_b, shift_b_to_a)` exactly once per
periodic pair and must cover every noninterface exterior face. For the helper:

```julia
pairs = [(f, g, -bc.value.transform.distance) for bc in q.periodic if bc isa PeriodicParent for (f, g) in zip(bc.IDs_range, bc.value.face_map)]
h = construct_periodic_circular_geometry(q.mesh, q.coupling, pairs)
```

`periodic_circular_reconstruct!` uses inverse-distance-squared LSQ with translated
neighbour images, preserving distinct images even when cell IDs coincide.
`periodic_circular_gradient!` reconstructs shared ordinary/periodic traces and
integrates circular traces with exact moments. `periodic_circular_terms(h, gradients,
gamma)` freezes positive constant-coefficient scalar negative-Laplacian terms;
`periodic_circular_flux(terms, values)` recovers from those same terms. Each edge
contributes once with opposite owner signs; paired exterior diagnostic fluxes are
exact negatives. Matrix/offset use integrated balance, not volume-divided rows.

Input fields must be periodic scalars. A global affine scalar is **not periodic**;
per-image affine moments and smooth periodic analytic fields are tested separately.
No quasi-periodic pressure jump, rotational vector transform, variable coefficient,
pressure solve, momentum coupling or transient solver is provided here. Existing
nonperiodic circular APIs/defaults remain unchanged. Snapshot arrays are read-only;
rebuild after geometry, IDs or periodic configuration changes.

`test/cases/test_circular_periodic_operators.jl` verifies original matrix/flux identity,
shared conservation, gauge invariance, conditioning, rejection/lifecycle, smooth
periodic gradient refinement and stitched-control topology. These checks are not a
transient Taylor–Green or full Laplacian-accuracy qualification.

### Opt-in quadratic scalar diffusion snapshot

`construct_periodic_circular_quadratic(h; coefficient=1.0)` freezes a deep copy
of `PeriodicCircularGeometry`, fixed two-hop scaled quadratic LSQ stencils and
sparse shared-edge/cell maps. Current scope: static CPU Float64, filled unit square
with unit-axis translated periodic exterior pairs and a split circle. Five resolved
nonconstant coefficients are required; no adaptive enlargement, regularisation or
fallback. `condition_limit` (default100) and `max_rows` (default80) may tighten limits.

```julia
t = construct_periodic_circular_quadratic(h; coefficient=1.0)
p = [sin(2pi*x[1])*cos(2pi*x[2]) for x in h.points]
r = periodic_circular_quadratic_flux(t, p)
# t.edge_matrix*p == r.edges (shared integrated outward flux)
# t.matrix*p ≈ r.balance (integrated negative Laplacian, not divided by volume)
```

This is a **nonsymmetric full reconstructed-gradient operator**, not the original
positive two-point matrix plus deferred correction. Snapshot arrays are read-only;
rebuild after geometry/connectivity/pair changes. No solver registration, gauge row,
pressure/momentum/PISO/ALE/GPU/MPI support or default change. Only constant positive
coefficient supported. Existing terms/flux APIs remain unchanged.

`test/cases/test_circular_periodic_quadratic.jl` checks independent integrated
polynomial-gradient flux, local quadratic coefficient reproduction, sparse-map
closure/conservation, translated images, failure/lifecycle isolation and solved
manufactured refinement. Bounded workspace studies found positive mean-zero scalar
energy on tested fine meshes and convergent GMRES, not an arbitrary-mesh guarantee.
Fine relative RMS was1.3–1.6%; previous coarse RMS<.10 failures remain documented,
not retroactively relabelled. Standalone stitched control remains diagnostic-only.

### Manufactured diffusion application (accuracy study)

`circular_periodic_diffusion.jl` defines `CircularPeriodicDiffusion.evaluate(q)` for
unit-square fixtures. It applies the static scalar diffusion operator to
`sin(2πx)cos(2πy)` and compares integrated cell balances against independent volume
quadrature of `8π²p`. Volume quadrature integrates each bilinear polygon plus signed
circle segments; independent boundary quadrature integrates analytic gradient flux
on straight faces/exact arcs. Orders 8/12 check quadrature convergence. The output
separates circle-adjacent, seam-adjacent and remaining cells, also fixed physical
bands, and distinguishes cell-integrated source from centre point sampling.

The stitched chord control uses an explicitly labelled **diagnostic implementation
of the same two-point/corrected scalar formula**, not the stock XCALibre Laplacian
assembly. Analytical gradients are an additional oracle, never solver data.

In the bounded three-refinement study, global errors decrease but circle-adjacent
and seam-adjacent refinement gates **fail**, including the stitched control. The
operator is not locally truncation-accuracy-qualified. Conservation, quadrature and
matrix/flux certificate tests pass; their test success must not be interpreted as
passing the separate accuracy gates. No solver/operator changes or tolerance tuning
were made in response. Workspace study artifacts/report retain the failed gates
and warmed construction/application/storage measurements.

A separate [circular geometry prototype](circular_geometry.md) validates shared
angular overlaps, consistent curved metrics, standalone operators and a separate
restricted static circular SIMPLE path; it does not extend the planar solver.
See [static circular Couette verification](circular_simple.md), including the
opt-in reconstructed physical predictor. Run `circular_predictor.jl` to compare
it with the unchanged low-order default; pressure improves in the documented
refinement sequence, but gradient and phase-sensitive accuracy remain limited.

These CPU examples build their own meshes and write no output files. Use an environment
containing XCALibre and its dependencies:

```sh
julia --threads=4 --project=<environment> examples/interface_verification/planar_channel.jl
julia --threads=4 --project=<environment> examples/interface_verification/channel_refinement.jl
```

Paths assume the package root. No external mesh files or plotting packages are required.
`channel_setup.jl` defines reusable `channel_case` and `channel_metrics` helpers without
running a simulation. The executable scripts leave their results in `results`.

## Channel and continuation

`planar_channel.jl` runs a length-2, height-1 channel at unit density and viscosity 0.1.
The inlet profile is `u = 6y(1-y)`, mean velocity 1, with no-slip walls, zero-gradient
outlet velocity and fixed outlet pressure. Two disconnected blocks meet at x=0.8,
with 8 and 9 interface faces and unequal normal cell distances. Pressure descriptors
are constructed from momentum descriptors to share geometry. The first 20 iterations
are followed by continuation on the **same** `PlanarSimpleState`, preserving individual
corrected overlap fluxes.

Parameters to `run_channel`/`channel_case` include `na`, `nb`, `nx` (streamwise cells
per block), `interface_x`, `nu`, `relaxU`, `relaxp`, `initialU`, `initialp`, `iterations`
and `convergence`. For example, after including `planar_channel.jl`:

```julia
case = run_channel(na=6, nb=19, nx=6, interface_x=0.65, relaxU=0.7, relaxp=0.3)
@show case.metrics case.continued.converged
```

Metrics use corrected physical-face fluxes for continuity and boundary balance,
without adding overlap contributions a second time. Interface balance is the sum of
the two outward patch fluxes. Velocity error is the volume-weighted relative L2 error
of `(ux,uy)` against `(6y(1-y),0)` at cell centres.

Pressure drop is a volume-weighted cross-section average at the **first and last
cell-centre columns** (reported as `sections`), not a pressure maximum or a boundary
extrapolation. Its fully developed reference is `12nu*(xright-xleft)`. This is an
analytical comparison for the channel profile; finite-domain inlet/outlet conditions
and interface trace approximations may perturb the discrete flow. Outlet cell-centre
pressure is not expected to be zero just because the boundary pressure is zero.

Representative default results (Float64): 54 cumulative iterations, velocity error
0.0128, relative pressure-drop error 0.0209, integrated continuity below 1e-12,
interface balance zero to roundoff. Iteration counts and last digits are not guarantees.

## Refinement study

`channel_refinement.jl` runs three refinements with `(na,nb,nx) = (4k,5k,3k)` for
`k = 1,2,4`, plus ordinary conforming SIMPLE references with `(ny,nx) = (5k,6k)`.
The conforming mesh is a single uniform block, so it is a comparable resolution
reference, not an identical cell layout. Both use Upwind convection, Linear diffusion,
Gauss gradients, relaxation `(0.7,0.3)` and no transpose-stress correction. A separate
6:19 face-count mismatch uses x=0.65.

Representative relative errors:

| k | Nonconforming velocity | Conforming velocity | Nonconforming pressure drop | Conforming pressure drop |
|---|---:|---:|---:|---:|
| 1 | 0.0349 | 0.0272 | 0.00323 | 0.000283 |
| 2 | 0.0103 | 0.00757 | 0.0128 | 0.0120 |
| 4 | 0.00249 | 0.00184 | 0.00803 | 0.00906 |

Nonconforming observed velocity orders are approximately 1.76 and 2.05. **Pressure-drop
error is nonmonotonic**, with observed orders approximately -1.99 and 0.68. Coarse-grid
cancellation, changing interior sampling planes and finite-domain boundary effects
preclude a clean pressure-order claim. The tests require decreasing velocity errors,
bounded pressure errors and independent conservation; they do not require second-order
accuracy. Upwind convection and missing tangential/nonorthogonal reconstruction remain
limitations. The strong-mismatch case has velocity error 0.0136 and pressure-drop error
0.0307, despite continuity at roundoff: conservation and accuracy are distinct checks.

## Robustness findings and scope

`test/unit/Solvers/test_planar_simple_robustness.jl` checks initial-state variations,
relaxation variations, signed Couette flow with opposing overlaps, pressure-offset
invariance, and recovery after validation and mid-iteration failures.

The restricted solver reconstructs its pressure predictor from the **physical,
pre-relaxation momentum equation**. It stores physical diagonal D, relaxed diagonal
Dt and old velocity Uold; after pressure-source removal it computes
`H0 = (Dt*Hv - (Dt-D)*Uold)/D` and mobility `V/D`. This removes the entire
relaxation/diagonal-dominance augmentation, not just a factor of alpha. Positive
identical x/y physical diagonals are required. Both ordinary and overlap face
predictors, pressure assembly/correction and cell velocity correction use this
physical predictor/mobility. Relaxation factors must lie in (0,1].

Different initial states, momentum relaxations 0.5/0.7/0.8 and pressure relaxations
0.2/0.3/0.4/0.5 now converge to agreeing fields and individual/ordinary fluxes within
2e-6 absolute/relative test tolerances on the tested channel. This is validation
for the restricted cases, not a general stability guarantee. Sparse-row tests
independently reassemble the unrelaxed momentum equation, include interface
neighbours and exercise diagonal-dominance augmentation at alpha=1.

Ordinary SIMPLE is unchanged and still shows relaxation-sensitive converged
channel solutions; the conforming results above are a diagnostic resolution
baseline using relaxation 0.7, not the same projection algorithm. Compared with
the previous relaxed predictor, velocity errors shift slightly (finest 0.00248
to 0.00249), pressure errors improve in these cases, and the refinement trend is
preserved. No claim of universal spatial-error improvement is made.

The solver is restricted to steady static planar CPU Mesh2, unit density, positive
constant viscosity, laminar/isothermal physics, Upwind momentum, Linear diffusion and
unlimited Gauss gradients. No mesh motion/ALE, PISO, GPU, MPI, SIMPLEC, file restart,
output/postprocessing or additional sources. Rebuild state after geometry/configuration
changes. Continuation is invalidated by every failed call; retry rebuilds fluxes from
current cell fields rather than reusing stale corrected overlaps.

The runnable channel and larger study are checked in isolated modules by
`test/cases/test_planar_verification_examples.jl`; the robustness tests belong to the
unit tier. Tight linear solves keep iteration errors below the reported spatial errors.
