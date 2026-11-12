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
pressure-area integration treatment. Before solver integration we still need
cell-pressure reconstruction, corrected sample locations/centroids as appropriate,
and diffusion/mobility and skewness review. All solver volume consumers must use
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
