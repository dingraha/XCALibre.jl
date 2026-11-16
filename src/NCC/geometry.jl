# SPDX-License-Identifier: MPL-2.0
finite3(x) = all(isfinite, x)
geometry_close(a, b) = maximum(abs, a-b) <= 1e-11 * max(1.0, maximum(abs, a), maximum(abs, b))

function validate_geometry(input::NCCGeometryInput)
    # Publish only the fully validated copy, never caller-owned arrays.
    s = deepcopy(input)
    require_ncc(s.schema == 1, "schema must be 1")
    nc, nf, np = length(s.cells), length(s.faces), length(s.patches)
    require_ncc(nc > 0, "empty_cells")
    require_ncc(max(nc, nf, np) < typemax(Int32), "index capacity")
    require_ncc([c.id for c in s.cells] == collect(1:nc), "cell_ids")
    require_ncc([p.id for p in s.patches] == collect(1:np), "patch_ids")
    require_ncc(allunique([p.name for p in s.patches]), "patch_names")
    for c in s.cells
        require_ncc(finite3(c.centre), "nonfinite_C")
        require_ncc(isfinite(c.volume) && c.volume > 0, "nonpositive_V")
    end
    keymap = Dict(f.key => i for (i,f) in enumerate(s.faces))
    require_ncc(length(keymap) == nf, "face_keys")
    for f in s.faces
        k = f.key
        require_ncc(k.class in (:internal,:boundary) && k.local_id >= 1, "face_keys")
        require_ncc(k.class === :internal ? k.patch === nothing : k.patch !== nothing && 1 <= k.patch <= np, "face patch")
    end
    covered = NCCFaceKey[]
    for p in s.patches
        require_ncc(p.kind in (:physical, :residual, :coupled, :empty, :error), "patch kind")
        require_ncc(finite3(p.translation), "nonfinite translation")
        require_ncc(all(k -> k.class === :boundary && k.patch == p.id, p.keys), "patch_coverage")
        require_ncc([k.local_id for k in p.keys] == collect(1:length(p.keys)), "patch_coverage")
        require_ncc(all(k -> haskey(keymap,k), p.keys), "patch_coverage")
        p.kind in (:empty,:error) && require_ncc(isempty(p.keys), "empty/error runtime faces")
        if p.kind === :coupled
            require_ncc(p.partner !== nothing && 1 <= p.partner <= np && p.partner != p.id, "partner_patch")
            other = s.patches[p.partner]
            require_ncc(other.kind === :coupled && other.partner == p.id, "partner_patch")
            require_ncc(geometry_close(p.translation, -other.translation), "translation_inverse")
        else
            require_ncc(p.partner === nothing && p.translation == zero(Vec3), "noncoupled patch metadata")
        end
        append!(covered,p.keys)
    end
    boundary_keys = [f.key for f in s.faces if f.key.class === :boundary]
    require_ncc(length(covered) == length(boundary_keys) && allunique(covered) && Set(covered) == Set(boundary_keys), "patch_coverage")
    internal_keys = [f.key for f in s.faces if f.key.class === :internal]
    require_ncc(sort([k.local_id for k in internal_keys]) == collect(1:length(internal_keys)), "internal face IDs")
    if s.physical !== nothing
        require_ncc(all(finite3,s.physical.points), "physical points")
        for polygon in s.physical.polygons
            require_ncc(length(polygon) >= 3 && allunique(polygon) && all(i -> 1 <= i <= length(s.physical.points),polygon), "physical polygons")
        end
    end
    paired = zeros(Int32,nf)
    for (a,b) in s.pairs
        require_ncc(haskey(keymap,a) && haskey(keymap,b) && a != b, "pair_missing")
        ia, ib = keymap[a], keymap[b]
        require_ncc(paired[ia] == 0 && paired[ib] == 0, "pair_reciprocity")
        require_ncc(a.class === :boundary && b.class === :boundary, "pair class")
        require_ncc(s.patches[a.patch].kind === :coupled && s.patches[a.patch].partner == b.patch, "partner_patch")
        paired[ia], paired[ib] = Int32(ib), Int32(ia)
    end
    areas = zeros(nf); normals = fill(zero(Vec3),nf); deltas = fill(zero(Vec3),nf)
    distances = zeros(nf); weights = ones(nf); correction_weights = ones(nf)
    gathers = [Tuple{Int32,Int8}[] for _ in 1:nc]
    for (i,f) in enumerate(s.faces)
        require_ncc(f.key.class in (:internal,:boundary) && f.key.local_id >= 1, "face_keys")
        require_ncc(1 <= f.owner <= nc, "owner")
        require_ncc(f.physical_face > 0 && (s.physical === nothing || f.physical_face <= length(s.physical.polygons)), "physical_provenance")
        require_ncc(finite3(f.Sf), "nonfinite_Sf")
        require_ncc(finite3(f.Cf), "nonfinite_Cf")
        A = norm(f.Sf)
        require_ncc(isfinite(A) && A > 0, "zero_area")
        n = f.Sf/A
        Co = s.cells[f.owner].centre
        linked = f.key.class === :internal || paired[i] != 0
        if f.key.class === :internal
            require_ncc(f.key.patch === nothing, "internal patch")
            require_ncc(f.neighbour !== nothing && 1 <= f.neighbour <= nc, "neighbour")
            require_ncc(f.neighbour != f.owner, "internal_self")
        else
            require_ncc(f.key.patch !== nothing && 1 <= f.key.patch <= np, "face patch")
            p = s.patches[f.key.patch]
            require_ncc((p.kind === :coupled) == (paired[i] != 0), "pair_missing")
            if paired[i] == 0
                require_ncc(f.neighbour === nothing, "noncoupled neighbour")
            else
                other = s.faces[paired[i]]
                require_ncc(f.neighbour !== nothing && 1 <= f.neighbour <= nc && f.neighbour == other.owner && other.neighbour == f.owner, "partner_cells")
                require_ncc(geometry_close(f.Sf,-other.Sf), "paired_Sf")
                require_ncc(geometry_close(f.Cf,other.Cf+p.translation), "paired_Cf")
            end
        end
        if linked
            t = f.key.class === :internal ? zero(Vec3) : s.patches[f.key.patch].translation
            d = s.cells[f.neighbour].centre + t - Co
            require_ncc(finite3(d) && isfinite(norm(d)) && norm(d) > 0, "zero_delta")
            do_ = dot(n,f.Cf-Co); dn = dot(n,d-(f.Cf-Co))
            require_ncc(isfinite(do_) && do_ > 0, "owner_projection")
            require_ncc(isfinite(dn) && dn > 0, "neighbour_projection")
            weights[i] = dn/(do_+dn)
            correction_weights[i] = dot(n,d-(f.Cf-Co))/dot(n,d)
            require_ncc(isfinite(weights[i]) && 0 < weights[i] < 1 && isfinite(correction_weights[i]), "projected weights")
        else
            d = f.Cf-Co
            require_ncc(finite3(d) && isfinite(norm(d)) && norm(d) > 0, "physical_zero_delta")
        end
        areas[i], normals[i], deltas[i], distances[i] = A,n,d,norm(d)
        push!(gathers[f.owner],(Int32(i),Int8(1)))
        f.key.class === :internal && push!(gathers[f.neighbour],(Int32(i),Int8(-1)))
    end
    # Runtime order remains the caller's explicit key order. No boundary-first mesh is exported.
    (; input=s, keymap, paired, areas, normals, deltas, distances, weights, correction_weights, gathers)
end
