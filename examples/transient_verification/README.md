# Transient verification cases

Small transient problems that check the time discretisation: the time-order scripts run one case with
`Euler` and with `CrankNicolson` at a sequence of halved time steps, prints the error at the end
time and the observed order log₂(eᵢ/eᵢ₊₁), and leaves them in `results = (; dts, errors, orders)`.
They are verification cases (known solution or self-convergence, a measured order of accuracy)
rather than physical benchmarks, hence the folder name. The meshes are built in the scripts
with `MeshBuilder2D`; all cases run on the CPU in well under a minute each (most of it
compilation) and write no output files.

Run one with, e.g., `julia --project=<env with XCALibre> heat_decay.jl`.

| Script | Solver | What it checks | Reference | Euler | CrankNicolson |
|---|---|---|---|---|---|
| `heat_decay.jl` | `laplace!` | Time operator on a scalar; also `CrankNicolson(0.9)` (off-centred, first order) | exact semi-discrete solution 1 + cos(πx)exp(λₕt) | 1.0 | 2.0 |
| `shear_wave_piso.jl` | `piso!` | PISO stage structure (predictor, correctors) | exact semi-discrete solution sin(2πy)exp(λₕνt) | 1.0 | 2.0 |
| `shear_wave_les.jl` | `piso!`, `LES{Smagorinsky}` | time level of the eddy viscosity | CN at Δt/16 | 1.0 | 2.1 |
| `oscillating_wall.jl` | `piso!` | time-dependent boundary values (`DirichletFunction`) | CN at Δt/16 | 1.0 | 2.0 |
| `taylor_green_advected.jl` | `piso!` | convecting flux and pressure coupling | CN at Δt/16 | 0.9 | 1.8–1.6 |
| `taylor_green_analytical.jl` | `piso!` | ordinary periodic analytical U/p baseline and retained-flux conservation | continuum Taylor–Green | measured separately | not studied |
| `acoustic_wave_cpiso.jl` | `cpiso!`, `Compressible`, `SensibleEnthalpy` | compressible time terms, `outer_loops` | CN at Δt/32 | see below | 2.2–2.4 |

Orders are those of the finest pairs of time steps. Notes:

- λₕ is the eigenvalue of the discrete Laplacian for the mode, so the errors of the first two cases
  are purely temporal. The other cases have no simple discrete solution and are measured against
  a run with a 16 (32) times smaller time step on the same mesh.
- `shear_wave_les.jl` uses Smagorinsky because WALE gives ν_t = 0 in pure shear.
- `taylor_green_advected.jl`: on 24² cells the CN order falls from 1.8 to 1.6 at the smaller
  time steps; the part responsible shrinks with the mesh (h²Δt, from the Rhie-Chow face flux).
- `acoustic_wave_cpiso.jl`: CrankNicolson needs `outer_loops = 8` so the density of the time terms
  matches the state equation at tⁿ⁺¹; Euler runs with one pass. The Euler error of this case does
  not decrease with Δt (amplitude and phase of the wave are off by an amount independent of Δt).

Variable time-step patterns need an internal post-processing hook to prescribe the steps;
`test/unit/Discretise/test_crank_nicolson.jl` covers them.

`heat_decay.jl` is run by the test suite (`test/cases/test_transient_verification_examples.jl`);
the shear wave of `shear_wave_piso.jl` is tested in `test/unit/Discretise/test_crank_nicolson.jl`.

## Analytical Taylor–Green baseline (no interface)

`taylor_green_analytical.jl` is a separate analytical accuracy harness, not a change to the
advected vortex's temporal self-convergence test. Run it directly for a small demonstration,
or include it to use `TaylorGreenAnalytical.run_case(; n, dt, tend, inner_loops)` and its
`exact`, `errors`, `flux_checks` helpers. Larger refinement runs are not automatic on include.

Defaults: unit periodic square, unit velocity scale/density, ν=0.01, t=0.2, Euler,
Linear convection, Gauss gradient, two PISO correctors, componentwise viscosity
(`transpose_stress=false` explicitly, without changing solver defaults). Velocity and pressure
are initialized analytically at cell point locations. The mesh is stationary and conforming:
**this is not a sliding-interface or ALE test**.

Results contain volume-weighted RMS/relative and maximum U/p errors with one common
volume-weighted pressure-error gauge, retained flux, independent original-cell continuity,
paired periodic flux cancellation, and setup/run timings with compilation counters.
The run timing includes the solver's internal setup and all timesteps; it is not a pure
per-iteration kernel measurement. Compilation counters are diagnostic and are not a precise
additive wall-time partition. Repeated `run!` calls are not used as timestep observations.

The compact automated case in `test/cases/test_taylor_green_analytical.jl` checks oracle
identities, gauge invariance, small-grid accuracy and conservative flux. It does not assert a
convergence order. In the initial 16²/32²/64² study at fixed dt=0.0005, both U/p errors decreased;
finest-grid dt halving changed p RMS by about 8%. Velocity temporal differences were not in a
clean asymptotic regime. Do not claim second-order space or first-order time from these runs.
Final nonlinear transient momentum closure is not independently certified by this harness;
PISO's returned linear residual histories are not a substitute for that check.

An Upwind control is needed before direct comparison with the current restricted circular
momentum operator. Periodic/transient circular coupling and moving GCL/time transfer remain
separate implementation milestones.
