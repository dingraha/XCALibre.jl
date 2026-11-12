# Geometry-only circular interface

`CircularInterface` is a static CPU Mesh2 geometry prototype, **not a curved
SIMPLE boundary condition or a moving-mesh solver**. The existing planar solver
and descriptors remain restricted to straight interfaces.

```julia
coupling = construct_circular_interface(mesh, :interface_a, :interface_b; centre=(0.0, 0.0, 0.0), radius=1.0)
face_balance = zeros(length(mesh.faces))
cell_balance = zeros(length(mesh.cells))
interface_face_flux!(face_balance, coupling, overlap_fluxes)
interface_cell_flux!(cell_balance, coupling, overlap_fluxes)
```

The two patches must each cover a complete circle once, with nodes on the same
specified circle in an xy plane. Each polygonal boundary chord spans less than
pi radians. Unequal/nonuniform segmentation is supported, as are independently
built frozen angular alignments. There is no radial projection or weight
renormalisation. Gaps, duplicate coverage, inconsistent mesh metrics/orientation,
degenerate faces and semicircle ambiguity are rejected. Absolute tolerance has
length units; relative tolerance is scaled by radius. Contacts or overlap pieces
no larger than tolerance are omitted; accumulated coverage is validated afterward.

## Stored metrics

For angular overlap [a,b], radius R and circle centre O:

- `measure`: arc length R(b-a), not the chord length or area-vector magnitude.
- `area_vector`: integral of the circular normal over the arc, oriented out of A.
- `centre`: arc-length centroid O + R*sinc((b-a)/2)*radial_midpoint (using sin(x)/x).
- `midpoint`: O + R*radial_midpoint, on the circle.
- `area_moment`: exact integral of x*n' ds, a 3x3 matrix in the shared Cartesian frame.
- `theta_lo`, `theta_hi`: interval bounds in [0,2pi]. A fixed seam may split a
  single face pair into two entries; retain both individual overlap fluxes.

The integrated arc normal equals the chord area vector between its endpoints.
Consequently each face's signed overlap vectors reconstruct its original mesh
area vector, even though the two polygonal boundaries do not coincide. Deterministic
face/cell gathers preserve equal-and-opposite integrated overlap fluxes. A constant
Cartesian velocity dotted with overlap area vectors reproduces original face
fluxes and preserves closed-cell area balance. These facts do **not** establish
accuracy for varying fields, curved-cell volumes or a curved pressure projection.

The arc moment differs from chord-midpoint times chord-area-vector. Replacing
chord surfaces with arcs changes geometric moments and enclosed volume. This
constructor deliberately leaves mesh coordinates, face metrics and polygonal
cell volumes unchanged; it must not be treated as a curved-volume mesh conversion.
The separate metric snapshot below corrects volumes without modifying the mesh.
Nor is one scalar physical-face trace sufficient in general to reproduce the
sum of overlap pressure times differently directed overlap area vectors.

The snapshot and gradient primitive below establish a curved volume/moment and
pressure-area integration treatment. The standalone cell-pressure reconstruction
and constant-coefficient corrected-flux primitives below extend this treatment.
The standalone pressure operator below adds exterior/ordinary trace handling,
variable mobility and deferred-correction projection. Coupled momentum and
solver-wide curved-volume integration remain separate work.
All solver volume consumers must use
the same geometry; this is not yet connected to source assembly or projection. In particular `area_vector/measure` is NOT
a unit normal, so existing planar distance/conductance formulas cannot be reused
unchanged.

## Curved metrics and pressure-area gradient primitive

```julia
metrics = construct_circular_metrics(mesh, coupling)
pressure_area = fill(zero(SVector{3,Float64}), length(coupling.measure))
gradient = fill(zero(SVector{3,Float64}), length(mesh.cells))
circular_pressure_area!(pressure_area, metrics.coupling, 2.0, (0.7, -0.3, 0.0))
face_pressure = [2.0 + dot(SVector(0.7, -0.3, 0.0), x-metrics.coupling.origin) for x in mesh.face_centre]
circular_gauss_gradient!(gradient, metrics, face_pressure, pressure_area)
```

Use `StaticArrays` and `LinearAlgebra` for this illustrative affine-field example.
No cell values or boundary descriptors are interpolated by these primitives.

`CircularMetrics` snapshots both the coupling and ordinary face geometry. It
stores original polygon volumes, corrected volumes, signed volume corrections,
and surface-moment corrections. For each ORIGINAL interface face, spanning d:

`delta_volume = sign * R^2 * (d - sin(d))/2`.

The same change in its pressure-area moment is `delta_volume * diag(1,1,0)`.
The correction must be computed once per complete face, not separately per
overlap: overlap chord segments do not collectively reproduce the original
face's chord moment. This matters particularly at the angular seam.

Construction revalidates coupling with its original construction tolerances,
checks ordinary chord geometry, closed cell vector sums, original polygon moment/
volume identities, curved moment identities and positive finite corrected volumes.
Metric-check tolerances are separate from the coupling's topology tolerances.
Mesh arrays and existing solvers remain unchanged; cell centroids are not corrected.

For the affine field `p(x) = p0 + dot(g,x-reference)`, exact overlap integrals are
`p0*S + (M-reference*S')'*g`. `circular_pressure_area!` computes these shared
A-oriented vectors. It is an analytical/reconstructed-affine primitive, not a
production pressure interpolation scheme. `p(arc centroid)*S` generally misses
the position/normal correlation and is not affine-exact.

`circular_gauss_gradient!` integrates pressure traces on ordinary straight faces,
replaces interface-face contributions with the supplied individual overlap
pressure-area integrals, and divides by corrected volume. Interface entries in
the ordinary face-pressure array are ignored (even NaN is allowed there). All
cells are overwritten; accumulation order is deterministic. Both numerical
primitives validate inputs and compute into temporary output before copying,
and reject output aliasing inputs/geometry. Snapshot arrays are read-only by
contract; rebuild rather than mutating them.

`test_circular_metrics.jl` validates affine x/y gradients, constant pressures,
pressure-offset invariance, equal-and-opposite integrals, translated circles,
numeric types, frozen alignments, volume/moment corrections and failure atomicity.
It deliberately demonstrates the error from dividing exact arc pressure-area
integrals by uncorrected polygon volumes. These are integration tests with exact
manufactured traces, not a claim of accuracy for interpolated cell pressure or
an integrated circular SIMPLE solver.

## Cell reconstruction and constant-coefficient corrected flux

```julia
reconstruction = construct_circular_reconstruction(mesh, coupling)
flux_geometry = construct_circular_flux_geometry(reconstruction)
cell_gradient = fill(zero(SVector{3,Float64}), length(mesh.cells))
circular_reconstruct!(cell_gradient, reconstruction, cell_pressure, exterior_pressure)
circular_reconstructed_pressure_area!(pressure_area, flux_geometry, cell_pressure, cell_gradient)
terms = circular_diffusion_terms(flux_geometry, cell_gradient, 0.1)
overlap_flux = zeros(length(coupling.measure))
circular_diffusion_flux!(overlap_flux, terms, cell_pressure)
```

These remain standalone CPU primitives, not boundary descriptors or a solver.
Cell values are POINT samples at existing mesh cell centres, not curved-cell
averages. Exterior pressures follow `reconstruction.exterior_faces` order and
are point samples at `exterior_points` (ordinary exterior chord centres). No
boundary conditions are evaluated. Internal neighbours and interface donor cells
enter inverse-distance-squared 2D LSQ; distinct donor IDs are deduplicated so the
fixed angular seam does not change stencil weights. Rank deficiency or excessive
normal-matrix condition number is rejected without regularisation. The default
`condition_limit` is 1e6. All snapshot arrays are read-only by contract.

Flux geometry uses S = integrated overlap area vector, d = xB-xA and
`beta = dot(S,S)/dot(S,d)`, `T = S-beta*d`. It requires alignment cosine above
`min_alignment` (default 0.05), plus resolved positive owner distances to the
plane through the arc midpoint normal to S. These distances define bounded
weights. No clipping is performed. Some coarse polygonal fixtures have valid
surface coverage but fail owner-plane bracketing: geometry validity does not
imply admissibility for this discretisation.

The shared pressure trace is the weighted blend of the two owner affine
polynomials, referenced at the arc midpoint. Its integral uses exact arc moments;
it is gauge/constant preserving and affine-exact when owner gradients are exact.
There is no limiter or guarantee of boundedness for general data.

For positive finite CONSTANT diffusivity/mobility gamma, `CircularDiffusionTerms`
freezes `G=gamma*beta` and `E=-gamma*dot(g_shared,T)`, with an independent copy
of flux geometry. Recovery uses `q=G*(phiA-phiB)+E`, oriented out of A. A future
-Laplacian assembly must use the overlap block `G*[1 -1; -1 1]` and explicit
RHS additions `(-E,+E)` on (A,B). The same frozen terms must be used for assembly
and recovery; do not refresh gradients in between. The coefficient block is
symmetric positive semidefinite, but this does not establish stability or
monotonicity of a deferred-correction solve. No equation assembly or solve is
installed by these primitives. The separate standalone pressure operator below
adds variable coefficient interpolation, assembly and pressure projection.

All numerical outputs are failure-atomic and reject aliasing input/snapshot
arrays. Tests independently assemble implicit blocks/explicit RHS and compare
with recovered cell flux sums. They verify cell-sample reconstruction, affine
pressure integrals and corrected fluxes, gauge invariance, donor deduplication,
positive coefficients, conservation, bad stencils/alignment and failure atomicity.
A logarithmic radial field with simultaneous angular/radial refinement gives
relative integrated overlap-flux errors approximately 0.0748, 0.0174, 0.00408.
This is a primitive manufactured-flux study, not convergence of a coupled solver.
Ordinary exterior face traces in those primitive Gauss chain tests are still
analytical; the pressure operator below also tests the complete reconstructed chain.

## Standalone static pressure projection

`construct_circular_pressure_geometry(mesh, coupling, kinds)` builds an independent
CPU snapshot of ordinary internal/exterior faces and shared circular overlaps.
`kinds` covers every noninterface exterior face in ascending face-ID order (the
same order as `CircularReconstruction.exterior_faces`), using only `:dirichlet`
or `:neumann`. Boundary values are scalar chord-face-centre point values for
Dirichlet; Neumann entries MUST be zero. There are no boundary callbacks,
ordinary solver descriptors, momentum equations, SIMPLE, or ALE support.

```julia
r = construct_circular_reconstruction(mesh, coupling)
kinds = fill(:dirichlet, length(r.exterior_faces))
h = construct_circular_pressure_geometry(mesh, coupling, kinds)
# boundary_values: one prescribed pressure at each r.exterior_points entry
# mobility: positive scalar cell values, e.g. corrected volume / physical diagonal
result = circular_pressure_project(h, mobility, boundary_values,
    predictor_faces, predictor_overlaps; max_sweeps=200, tolerance=1e-10)
@assert result.converged
```

`predictor_faces` has global face length, `predictor_overlaps` has overlap length.
Physical interface face entries of `predictor_faces` are ignored (NaN allowed):
overlaps supply the interface flux, and result face entries are signed diagnostic
sums. Predictor arrays are never mutated. Each sweep overwrites corrected fluxes
from this saved predictor; repeated calls do not accumulate pressure corrections.

Boundary-aware LSQ replaces exterior pressure-difference rows on Neumann faces
by unit-normal derivative rows, with zero RHS and unit weight. This constrains
only the normal derivative, unlike copying owner pressure to a displaced face.
Normal matrices are revalidated without regularisation. Construction also
validates the original all-sample reconstruction stencil, so both stencils must
pass the chosen condition limit. Ordinary internal traces blend the two affine
owner polynomials at the face centre. Neumann traces tangentially extrapolate the
owner reconstruction; Dirichlet traces use prescribed point values. Circular
traces integrate shared affine polynomials with exact moments. The complete
`circular_pressure_gradient!` chain uses corrected volumes, without changing mesh
cell centres or interpreting cell pressures as curved-cell averages.

Each ordinary internal face and overlap has the validated beta/T split described
above. Positive owner mobility is interpolated using series resistance:
`r_f=(dA+dB)/(dA/rA+dB/rB)`. Its stable scaled implementation avoids overflowing
reciprocals for extreme ratios. Boundary Dirichlet faces use owner mobility;
Neumann pressure flux is exactly zero, irrespective of tangential reconstruction.
`circular_pressure_terms(h, gradients, mobility, boundary_values)` freezes all
implicit/explicit terms, boundary data and independent geometry. Its ungauged
sparse integrated negative-Laplacian matrix satisfies
`matrix*p + offset == balance(pressure_flux)`.
`circular_pressure_flux(terms, p)` recovers ordinary and overlap fluxes from the
SAME terms. Pressure-flux balance is gathered once per face/overlap; divergence
is balance divided by corrected volume. No volumetric source API is provided.

The projection solves `balance(qH+qp)=0` using direct CPU sparse solves. Each sweep
reconstructs, freezes terms, assembles, solves and recovers fluxes. It then
reconstructs again at the new pressure to evaluate the refreshed full-operator
residual. BOTH recovered continuity and refreshed residual must be below
`tolerance`, an absolute integrated-flux tolerance. Exhaustion returns
`converged=false` with final pressure/fluxes, frozen `terms` and residual history.
There is no automatic damping or unconditional convergence guarantee for the
explicit tangential correction. Continuity of a frozen solve alone is not enough.

All-Neumann components require a reference dictionary, e.g. `reference=Dict(1=>0.0)`
for one connected unanchored domain. Require exactly one reference per unanchored
component; reject references on Dirichlet-anchored components. Compatibility is
checked on EACH component using its own predictor-flux scale, not global
cancellation. Symmetric row/column elimination fixes gauges, while convergence
and flux diagnostics evaluate original ungauged equations. Default compatibility
tolerance is `1000eps(T)` times max(component scale,1); imbalance beyond it is
rejected rather than hidden in the reference equation. Homogeneous pressure
Neumann boundaries cannot correct an unbalanced exterior predictor.

Tests cover full reconstructed affine gradients (including an affine field tangent
to a Neumann face), harmonic-mobility positivity, sparse assembly/recovery,
manufactured discrete variable-mobility projections, gauge invariance, incompatible
predictors, frozen-solve versus refreshed convergence and failure-atomic gradient
outputs. Component checks and gauge elimination additionally have independent
synthetic disconnected-graph tests. Radial log-pressure solves with simultaneous
angular/radial refinement give maximum cell-pressure errors approximately
0.0656, 0.0106, 0.00226. This is a standalone scalar pressure study, not a coupled
flow convergence result. Discontinuous mobility tests establish conservation,
operator consistency and orthogonal series resistance only: cross-interface LSQ
can smear physical gradient jumps, so general curved/skewed discontinuous-field
accuracy is NOT claimed.

The next coupled-flow milestone requires momentum assembly, physical-diagonal
mobility, consistent corrected volumes in all consumers, cell velocity correction
and explicit lifecycle/continuation rules. Existing planar/ordinary solvers remain
unchanged; do not pass these snapshots into their pressure paths.

## Rebuild lifecycle

Construct new coupling objects after any geometry/ID change. Existing objects
are snapshots with mutable array storage; do not mutate those arrays directly.
The geometry constructor does not detect later mesh mutation. There is no automatic
continuation/rebuild integration or old-overlap transfer in this increment.

Future solver integration must build new shared descriptors, extended sparse
equations, preconditioners/workspaces and predictor/coefficient storage after an
alignment change, invalidating all old geometry-bound continuation. Do not copy
fluxes by overlap index, even when counts happen to agree. Conservative transfer
and time-level geometry belong to a later ALE/discrete-GCL increment. Frozen
alignment tests are not simulations of motion.

Tests in `test/unit/InterfaceCoupling/test_circular_interface.jl` cover analytic
metrics and quadrature, both-side vector reconstruction, arbitrary signed scalar
and vector fluxes, closed-cell constant-velocity balances, multiple alignments,
wraparound, endpoint/order invariance, Float32/Float64 and Int32/Int64, invalid
geometry and isolated reconstruction. Test fixtures use independently segmented
polygonal annuli, without mesh files.
