# Frozen NCC source/provenance ledger

Production route selected by project owner: independently implement numerical method under MPL-2.0. B2b-1 implementation authorized separately; test execution/registration, stage2 algebra and distribution review are not authorized by that increment.

Authorship: AI coding assistant acting for project owner, current implementation session. Current author has inspected OpenFOAM Foundation dev fed424ace4164a7685a5035d43fff7ba03f76a0e and reference-derived GPL validation. **Not clean-room work. No legal clearance or independent-authorship certification.** Owner-selected route is not a qualified legal review. Reviewer/name/date/distribution approval: pending. No retrospective relabelling of GPL files.

New implementation follows the B2a numerical/data specification and B2b explicit facade design. No Foundation/validation source pasted or mechanically translated into these files; no GPL validation imports, includes or runtime dependency. New record schema, validation flow, ownership/snapshot design and typed guard containment authored for this narrow API. The current author cannot certify non-derivation solely by assertion; source review before distribution remains required.

| File | Origin/function |
|---|---|
| src/NCC/NCC.jl | New namespace/include/import registration following existing XCALibre module conventions |
| src/NCC/types.jl | New typed schema/owned handles/private identity tokens; no stock AbstractMesh/field inheritance |
| src/NCC/geometry.jl | New schema validation and deterministic pair/gather derivation against documented geometric predicates |
| src/NCC/context.jl | New owned scalar/mobility/q/physical trace inputs and deep snapshot data; coefficient formula from mathematical specification |
| src/NCC/guards.jl | New handle-specific rejection methods against existing MPL XCALibre entrypoints |
| src/XCALibre.jl | One include of new namespace; existing MPL module unchanged otherwise |
| test/unit/NCC/test_contract.jl | Newly authored 3-cell algebra inputs/malformed variations/no-mutation checks, no copied fixture/export/helper tests |
| docs/src/ncc_frozen_scalar.md, docs/src/ncc_provenance.md | New API limitations/origin disclosure, not Foundation documentation translation |

Numerical formulas: cell/image delta, area-vector norm, positive projected distances, normal-distance interpolation and explicitly selected reciprocal-distance/0.05 capped projected coefficient. These documented mathematical behaviours target Foundation fidelity; general fallback remains excluded. Citation points: Foundation surfaceInterpolation/nonConformalCoupled patch numerical policy; root xcalibre-foundation-ncc-implementation-map.md. Distinguish algorithm behaviour from source expression; reviewed legal acceptability remains unresolved.

Existing package APIs inspected: XCALibre module include order, ModelFramework equation/CSR construction, field layouts, named BC assignment, solver dispatch and writer initialization. Stage1 imports existing modules solely for typed rejection. No stock assembly/correction/recovery modifications, external-code dependencies or solver enablement. Passed B1 GPL validation files/results unchanged.

Before stage2: owner/API/source review of new code and exact next manifest. Before distribution: qualified review of new implementation origin, licences/notices and any external GPL validation/test bundle. If expressive copying/translation concern is identified, stop/review rather than rename/reformat it. No claim source separation alone removes obligations for combined distributions.
