# Static planar-interface verification

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
