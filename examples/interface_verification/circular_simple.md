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
SIMPLEC, motion or ALE. Velocity Dirichlet requires pressure homogeneous Neumann or `:normal_gradient`;
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
Pressure is compared at a shared cell reference gauge (equivalently, errors in
pressure differences relative to that cell). Velocity error is corrected-volume-
weighted relative L2 at existing cell points; pressure error is max absolute.
`pressure_gradient` reports max absolute reconstructed Gauss-gradient error.

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
as an order study of the exact continuum problem. The prescribed-gradient verification below now permits nonzero wall derivatives;
an automatically momentum-derived general wall condition remains future work.
Do not weaken convergence checks to conceal this limitation. Frozen alignments, gauge invariance, split continuation,
changed-field resets, pressure-sweep exhaustion and failure recovery are tested.
Constant Cartesian flow also verifies prescribed and mixed no-backflow outlet
boundaries. None of these tests represents moving meshes or GCL validation.

Existing ordinary and planar solvers are unchanged. Before broadening this path,
address automatic wall pressure consistency, larger/nonorthogonal cases and pressure accuracy;
actual sliding motion, conservative transfer and ALE/GCL remain separate milestones.

## Prescribed total-pressure derivatives

Pressure geometry now also accepts `:normal_gradient`. Its boundary value is the
prescribed outward-normal derivative at the ordinary chord-face centre, not a
pressure value. Existing `:neumann` remains exactly homogeneous and rejects
nonzero data. The boundary-aware LSQ RHS uses g_n*n. The Gauss pressure trace uses
owner tangential extrapolation plus g_n times the normal owner-to-face distance.
The integrated pressure flux is -r_owner*face_area*g_n, stored identically in
assembly and recovery. Projection compatibility includes this fixed contribution.

In SIMPLE, prescribed velocity is paired with this pressure condition. The saved
pressure-free wall predictor is q_wall+r_owner*face_area*g_n. Pressure recovery
subtracts the same term, retaining prescribed wall flow without leakage. Mobility
is frozen through each inner projection; boundary data are fixed state snapshots.
External callers of the standalone projection must supply their own consistent
predictor. This is a prescribed total-pressure condition, not a pressure-correction
condition and not yet an automatically derived wall-normal momentum model.

`circular_couette_case(wall_pressure=:analytical)` supplies
u_theta(r)^2/r * dot(e_r,n_wall) at actual chord centres with actual outward
normals. The inner derivative is negative. The exact circular stationary outer
wall derivative is zero, but its chord-centre sampled derivative need not be
exactly zero. This sampling and polygon wall geometry remain approximations.
Default `wall_pressure=:homogeneous` preserves the previous verification case.

Run the side-by-side study:

```sh
julia --threads=4 --project=<environment> examples/interface_verification/circular_wall_pressure.jl
```

Same fixed-domain refinement as above:

| k | Homogeneous pressure error | Analytical-gradient pressure error | Analytical-gradient max gradient error |
|---|---:|---:|---:|
| 1 | 0.0014164 | 0.0012358 | 0.0049449 |
| 2 | 0.0011912 | 0.0011007 | 0.0084286 |
| 4 | 0.0012015 | 0.0011531 | 0.0164290 |

Velocity errors and iteration counts are nearly unchanged. Analytical derivatives
reduce pressure-difference errors by about 13%, 8% and 4%, but do NOT remove the
plateau. Max Gauss-gradient errors grow under refinement in both modes. This is
an unresolved accuracy issue even though momentum/pressure/continuity residuals
converge tightly. No pressure or gradient convergence claim is made. The wall BC
mismatch was therefore not the sole or dominant fine-grid error in this study.
Investigating the physical-predictor/pressure-gradient fixed point, boundary
localisation of gradient errors and a consistent reconstructed predictor is the
next diagnostic task, separate from implementing automatic wall derivatives.

Tests cover affine LSQ and Gauss reproduction, variable-mobility assembly/recovery,
nonzero derivative compatibility (including an incompatible predictor alone),
mixed Dirichlet/gradient boundaries, wall-flow preservation, gauge/relaxation
agreement, continuation and failure reset. These establish discrete consistency,
not general curved-flow accuracy or stability.
