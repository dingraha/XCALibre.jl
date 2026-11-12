# Restricted static circular SIMPLE

Run from the package root, in an environment containing XCALibre dependencies:

```sh
julia --threads=4 --project=<environment> examples/interface_verification/circular_couette.jl
```

`circular_setup.jl` builds independently segmented multilayer polygonal annuli,
without mesh files, output or background work. Physical radii stay fixed at
0.5/1/1.5 while radial and angular resolution increase. The shared radius-one
interface uses exact arc surfaces and corrected cell volumes; inner/outer walls
remain polygonal chords. No mesh moves: prescribed rotating-wall velocity is not
sliding-motion/ALE validation.

## API and scope

`CircularSimpleState(momentum_geometry, velocity, pressure, velocity_boundary_values,
pressure_boundary_values, viscosity; ...)` is a separate array-based state, not
ordinary Physics/run! integration. Inputs are copied. Cell velocities use Cartesian
xy `SVector{3}` point samples, with exactly zero z. Pressure is scalar point data.
Boundary values follow the geometry's exterior face order. Config and geometry
arrays are read-only by contract. Only velocity/pressure may be changed between
calls. A new state is required after geometry, connectivity, boundary or config
changes; the snapshot does not track live mesh mutations.

Restrictions: static CPU Mesh2, unit density, positive constant viscosity, Upwind
convection, componentwise corrected viscous diffusion, fixed boundary data. No
sources, transient terms, transpose stress, turbulence, callbacks, output/restart,
SIMPLEC, motion or ALE. Velocity Dirichlet requires pressure homogeneous Neumann;
velocity homogeneous Neumann requires pressure Dirichlet. Neumann velocity backflow
is rejected. Each component needs a velocity anchor, and each pressure-unanchored
component exactly one reference, supplied as `reference=Dict(cell=>pressure)`.
Component predictor compatibility is checked before projection. Zero-gradient
velocity is not an impermeable wall; prescribe zero wall velocity.

`circular_simple!(state; iterations=100)` returns residual history, iterations this
call, total iterations and converged. Iteration exhaustion returns false, not an
exception. Inner pressure projection exhaustion throws and invalidates continuation.
No failed outer step publishes new fields/fluxes; earlier successful steps remain.
Unchanged fields retain individual corrected overlap fluxes across calls, enabling
exact split-call continuation. Changed fields or failed calls reset from current
cell values, not stale overlap fluxes. Failure invalidation occurs before argument
validation. Do not mutate state face/overlap continuation arrays.

## Coupling sequence

1. Freeze momentum using current velocity, pressure and retained corrected fluxes.
2. Solve relaxed momentum, recover physical H0 and mobility Vcurved/D, undoing
   full diagonal-dominance augmentation and its old-velocity source.
3. Save low-order interpolated H0 predictor fluxes and converge pressure deferred
   sweeps with fixed mobility. Require both continuity and refreshed operator residual.
4. Retain fluxes corrected with unrelaxed projected pressure. Relax cell pressure;
   shift unanchored components uniformly to their prescribed reference levels.
5. Correct cell velocity with H0-rD*GaussGradient(relaxed pressure), using the
   reconstructed pressure-area chain and corrected volumes.
6. Reassemble physical momentum at updated velocity, relaxed pressure and corrected
   convection fluxes. Independently gather continuity. Require refreshed momentum,
   pressure residual, continuity AND velocity/pressure/face/overlap changes below
   their absolute tolerances. Frozen linear solves alone do not imply convergence.

Default momentum/continuity/change/inner-pressure tolerances are 1e-9/1e-10/1e-9/
1e-11. Momentum residual has integrated momentum-flux units; continuity and pressure
residual have integrated volume-flow units. Change tolerance bounds velocity,
pressure and flux changes in their respective units: configure consciously for
scaled cases. These are restricted verification defaults, not nondimensional
production convergence criteria. Physical predictor interpolation remains low-order,
not affine-exact curved velocity integration. Corrected flux, not interpolated cell
velocity, drives the next convection operator.

## Couette verification and limitations

For inner angular speed omega=0.1 and nu=1, compare against circular-domain
u_theta=A*r+B/r, A=-omega*a^2/(b^2-a^2), B=omega*a^2*b^2/(b^2-a^2).
Integrating dp/dr=u_theta^2/r gives p=A^2*r^2/2+2*A*B*log(r)-B^2/(2*r^2)+C.
Pressure is compared at a shared cell reference gauge. Velocity error is corrected-
volume-weighted relative L2 at existing cell points; pressure error is max absolute.

Default na=16/nb=19 and three radial layers per zone converges in 46 iterations,
with velocity error about 0.07469 and pressure error 0.001216. Tested U/p relaxation
pairs (0.5,0.2), (0.7,0.3), (0.8,0.5) agree within 1e-7 in fields and individual
fluxes. This is evidence for these cases, not a universal stability guarantee.

Fixed-domain na=12k/nb=15k and 2k radial layers per zone, k=1/2/4:

| k | Iterations | Velocity relative L2 | Pressure max absolute |
|---|---:|---:|---:|
| 1 | 42 | 0.13413 | 0.0014164 |
| 2 | 78 | 0.03645 | 0.0011912 |
| 4 | 273 | 0.009374 | 0.0012015 |

Velocity decreases roughly quadratically in this study. Pressure is NOT converging
monotonically. Crucially, the supported homogeneous wall pressure Neumann condition
omits the exact radial dp/dr of rotating Couette flow. Along with wall faceting and
low-order predictor interpolation, this prevents interpreting these pressure errors
as an order study of the exact continuum problem. Supporting momentum-consistent
nonzero wall pressure derivatives is future work; do not weaken convergence checks
to conceal this limitation. Frozen alignments, gauge invariance, split continuation,
changed-field resets, pressure-sweep exhaustion and failure recovery are tested.
Constant Cartesian flow also verifies prescribed and mixed no-backflow outlet
boundaries. None of these tests represents moving meshes or GCL validation.

Existing ordinary and planar solvers are unchanged. Before broadening this path,
address wall pressure consistency, larger/nonorthogonal cases and pressure accuracy;
actual sliding motion, conservative transfer and ALE/GCL remain separate milestones.
