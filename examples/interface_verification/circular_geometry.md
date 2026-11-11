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
Nor is one scalar physical-face trace sufficient in general to reproduce the
sum of overlap pressure times differently directed overlap area vectors.

Before solver integration we must choose consistent cell volume/moment and
pressure-gradient treatment, review diffusion/mobility and skewness, and use an
interface-aware pressure-area gather that replaces the ordinary interface-face
contribution (never counts it twice). In particular `area_vector/measure` is NOT
a unit normal, so existing planar distance/conductance formulas cannot be reused
unchanged.

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
