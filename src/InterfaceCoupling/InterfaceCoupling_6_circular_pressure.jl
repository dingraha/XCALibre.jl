export CircularPressureGeometry, construct_circular_pressure_geometry
export circular_pressure_reconstruct!, circular_pressure_gradient!
export CircularPressureTerms, circular_pressure_terms, circular_pressure_flux
export circular_pressure_project

"""Static CPU pressure snapshot. `boundary_kind` follows `exterior_faces` order:
`:dirichlet` (face-centre point value) or `:neumann` (homogeneous normal derivative).
All geometry arrays are read-only by contract. Cell pressures are point samples,
not curved-cell averages. Rebuild after geometry/IDs or boundary kinds change.
No momentum, SIMPLE, ALE or automatic mesh change detection."""
struct CircularPressureGeometry{R,F,V,W,A,B,C}
    reconstruction::R
    overlap_geometry::F
    edge_a::A
    edge_b::A
    edge_face::A # ordinary face ID, or zero for circular overlap
    edge_overlap::A # overlap ID, or zero for ordinary internal face
    edge_area::V
    edge_point::V
    beta::W
    remainder::V
    weight_a::W
    exterior_faces::A
    boundary_kind::B
    boundary_beta::W
    boundary_remainder::V
    inverse::C
    components::Vector{Vector{Int}}
end

function _pressure_components(n,a,b)
    rows = [Int[] for _ in 1:n]
    for (i,j) in zip(a,b)
        push!(rows[i],j); push!(rows[j],i)
    end
    seen = falses(n); components = Vector{Int}[]
    for i in 1:n
        seen[i] && continue
        queue = [i]; seen[i] = true
        for j in queue
            for k in rows[j]
                if !seen[k]
                    push!(queue,k); seen[k] = true
                end
            end
        end
        push!(components,sort!(queue))
    end
    components
end

function _pressure_split(S,d,xA,x;min_alignment,radius)
    ns,nd = norm(S),norm(d)
    dot(S,d) > min_alignment*ns*nd && ns > 0 && nd > 0 ||
        _reconstruction_error("unsuitable ordinary face alignment")
    da = dot(x-xA,S/ns); db = dot(d,S/ns)-da
    min(da,db) > 100eps(eltype(S))*max(radius,nd) ||
        _reconstruction_error("ordinary face owner points do not bracket face")
    beta = dot(S,S)/dot(S,d)
    beta,S-beta*d,db/(da+db)
end

"""
    construct_circular_pressure_geometry(mesh, coupling, boundary_kind;
                                        min_alignment=0.05, condition_limit=1e6)

Validate all ordinary/overlap splits and replace exterior pressure-sample LSQ
rows by normal-derivative rows on homogeneous Neumann faces. Neumann rows are
unit-normal rows with unit weight; pressure-difference rows use inverse distance
squared weights. Reject singular/ill-conditioned stencils, without regularisation.
Boundary kinds must cover every ordinary exterior face, in ascending face-ID order.
"""
function construct_circular_pressure_geometry(mesh::Mesh2,c::CircularInterface,kinds::AbstractVector;
    min_alignment=0.05,condition_limit=1e6)
    _circular_input(kinds)
    all(k -> k in (:dirichlet,:neumann),kinds) || _reconstruction_error("unsupported pressure boundary kind")
    # The general primitive validates its sample stencil too. The boundary-aware
    # normal matrix is independently validated below, after replacing Neumann rows.
    r = construct_circular_reconstruction(mesh,c;condition_limit)
    f = construct_circular_flux_geometry(r;min_alignment)
    length(kinds) == length(r.exterior_faces) || throw(DimensionMismatch("incorrect pressure boundary coverage"))
    T,I = eltype(f.beta),eltype(c.cell_a)
    a,b,face,overlap = I[],I[],I[],I[]
    area,point,rem = SVector{3,T}[],SVector{3,T}[],SVector{3,T}[]
    beta,weights = T[],T[]
    m = r.metrics
    for j in eachindex(m.owner_a)
        u,v = m.owner_a[j],m.owner_b[j]
        u == v && continue
        S,x = m.face_area_vector[j],m.face_centre[j]
        B,R,w = _pressure_split(S,r.points[v]-r.points[u],r.points[u],x;
            min_alignment,radius=c.radius)
        push!(a,u);push!(b,v);push!(face,I(j));push!(overlap,zero(I))
        push!(area,S);push!(point,x);push!(rem,R);push!(beta,B);push!(weights,w)
    end
    for k in eachindex(c.measure)
        push!(a,c.cell_a[k]);push!(b,c.cell_b[k]);push!(face,zero(I));push!(overlap,I(k))
        push!(area,c.area_vector[k]);push!(point,c.midpoint[k]);push!(rem,f.remainder[k])
        push!(beta,f.beta[k]);push!(weights,f.weight_a[k])
    end
    bB,bR = T[],SVector{3,T}[]
    for j in r.exterior_faces
        u = m.owner_a[j]; S = m.face_area_vector[j]; d = m.face_centre[j]-r.points[u]
        dot(S,d) > min_alignment*norm(S)*norm(d) || _reconstruction_error("unsuitable exterior face alignment")
        B = dot(S,S)/dot(S,d)
        push!(bB,B);push!(bR,S-B*d)
    end
    inverse = SMatrix{2,2,T,4}[]
    for i in eachindex(r.points)
        G = zero(SMatrix{2,2,T,4})
        for j in r.neighbours[i]
            d = SVector{2,T}((r.points[j]-r.points[i])[1:2])
            G += d*d'/dot(d,d)
        end
        for j in r.exterior_rows[i]
            v = kinds[j] == :neumann ? normalize(m.face_area_vector[r.exterior_faces[j]]) :
                r.exterior_points[j]-r.points[i]
            d = SVector{2,T}(v[1],v[2])
            G += d*d'/dot(d,d)
        end
        push!(inverse,_circular_inverse(G,condition_limit,i))
    end
    CircularPressureGeometry(r,f,a,b,face,overlap,area,point,beta,rem,weights,
        copy(r.exterior_faces),collect(kinds),bB,bR,inverse,_pressure_components(length(r.points),a,b))
end

function _pressure_data(h,p,values)
    _circular_input(p,values)
    length(p) == length(h.reconstruction.points) && length(values) == length(h.exterior_faces) ||
        throw(DimensionMismatch("incorrect pressure or boundary data length"))
    all(isfinite,p) && all(isfinite,values) || _reconstruction_error("nonfinite pressure/boundary data")
    all(j -> h.boundary_kind[j] == :dirichlet || iszero(values[j]),eachindex(values)) ||
        _reconstruction_error("Neumann boundary values must be exactly zero")
end

"""Failure-atomic boundary-aware LSQ. Dirichlet values follow exterior_faces;
Neumann entries must be zero and constrain the NORMAL derivative, not tangential
variation or a displaced pressure difference."""
function circular_pressure_reconstruct!(out::AbstractVector,h::CircularPressureGeometry,p::AbstractVector,values::AbstractVector)
    _pressure_data(h,p,values)
    r = h.reconstruction
    _circular_output(out,length(p),true,h,p,values)
    proposed = similar(out); T = eltype(h.beta)
    for i in eachindex(p)
        rhs = zero(SVector{2,T})
        for j in r.neighbours[i]
            d = SVector{2,T}((r.points[j]-r.points[i])[1:2])
            rhs += (p[j]-p[i])*d/dot(d,d)
        end
        for j in r.exterior_rows[i]
            h.boundary_kind[j] == :neumann && continue # zero derivative RHS
            d = SVector{2,T}((r.exterior_points[j]-r.points[i])[1:2])
            rhs += (values[j]-p[i])*d/dot(d,d)
        end
        g = h.inverse[i]*rhs
        proposed[i] = SVector(g[1],g[2],zero(T))
    end
    all(x -> all(isfinite,x),proposed) || _reconstruction_error("nonfinite pressure reconstruction")
    copyto!(out,proposed)
    out
end

"""
    circular_pressure_gradient!(out, geometry, pressure, boundary_values)

Complete cell-data -> LSQ -> reconstructed ordinary/circular traces -> Gauss chain.
Dirichlet traces are supplied chord-centre values; Neumann traces tangentially
extrapolate owner LSQ gradients. Ordinary internal traces blend owner affine
polynomials. Circular pressure-area integrals use exact moments. Divide only by
corrected snapshot volumes. Does not imply LSQ and Gauss agree for general fields.
"""
function circular_pressure_gradient!(out::AbstractVector,h::CircularPressureGeometry,p::AbstractVector,values::AbstractVector)
    _pressure_data(h,p,values)
    _circular_output(out,length(p),true,h,p,values)
    r = h.reconstruction; T = eltype(h.beta)
    g = fill(zero(SVector{3,T}),length(p))
    circular_pressure_reconstruct!(g,h,p,values)
    traces = fill(T(NaN),length(r.metrics.owner_a))
    for k in eachindex(h.edge_a)
        j = h.edge_face[k]; iszero(j) && continue
        a,b,w,x = h.edge_a[k],h.edge_b[k],h.weight_a[k],h.edge_point[k]
        traces[j] = w*(p[a]+dot(g[a],x-r.points[a]))+(1-w)*(p[b]+dot(g[b],x-r.points[b]))
    end
    for (k,j) in enumerate(h.exterior_faces)
        a = r.metrics.owner_a[j]
        if h.boundary_kind[k] == :dirichlet
            traces[j] = values[k]
        else
            n = normalize(r.metrics.face_area_vector[j]); d = r.metrics.face_centre[j]-r.points[a]
            traces[j] = p[a]+dot(g[a],d-dot(d,n)*n)
        end
    end
    integrated = fill(zero(SVector{3,T}),length(r.metrics.coupling.measure))
    circular_reconstructed_pressure_area!(integrated,h.overlap_geometry,p,g)
    circular_gauss_gradient!(out,r.metrics,traces,integrated)
end

"""Frozen variable-mobility pressure terms. `matrix` is the ungauged integrated
negative Laplacian. `offset` is the balance of explicit fluxes and prescribed
Dirichlet terms. Thus matrix*p+offset == recovered pressure-flux balance.
Snapshot storage is read-only by contract. Use identical terms for solve/recovery."""
struct CircularPressureTerms{H,V,A}
    geometry::H
    mobility::V
    conductance::V
    correction::V
    boundary_conductance::V
    boundary_correction::V
    boundary_values::V
    matrix::A
    offset::V
end

# Stable harmonic mean: weight_a weights pressure interpolation, whereas
# resistance weights are distance fractions (1-weight_a, weight_a).
function _pressure_harmonic(a,b,w)
    scale = min(a,b)
    scale/((1-w)*(scale/a)+w*(scale/b))
end

"""
    circular_pressure_terms(geometry, gradients, cell_mobility, boundary_values)

Positive scalar owner mobility, distance-weighted harmonic shared edge mobility.
Exterior Dirichlet mobility is owner mobility; homogeneous Neumann flux is zero.
Freeze gradients, mobility and boundary values together. Cross-interface LSQ is
not a discontinuous-coefficient gradient-jump reconstruction: general jump-field
accuracy on curved/skewed meshes is NOT guaranteed. No variable-density flux.
"""
function circular_pressure_terms(h::CircularPressureGeometry,g::AbstractVector,mob::AbstractVector,values::AbstractVector)
    n = length(h.reconstruction.points); T = eltype(h.beta)
    _pressure_data(h,zeros(T,n),values)
    _circular_samples(h.overlap_geometry,zeros(T,n),g)
    _circular_input(mob)
    length(mob) == n || throw(DimensionMismatch("incorrect mobility length"))
    all(x -> isfinite(x) && x > 0,mob) || _reconstruction_error("positive finite cell mobility required")
    mobility,G,E = similar(h.beta),similar(h.beta),similar(h.beta)
    bg,be = similar(h.boundary_beta),similar(h.boundary_beta)
    rows,cols = Int[],Int[]; entries = T[]; offset=zeros(T,n)
    for k in eachindex(G)
        a,b,w = h.edge_a[k],h.edge_b[k],h.weight_a[k]
        mobility[k] = _pressure_harmonic(mob[a],mob[b],w)
        G[k] = mobility[k]*h.beta[k]
        E[k] = -mobility[k]*dot(w*g[a]+(1-w)*g[b],h.remainder[k])
        append!(rows,(a,a,b,b));append!(cols,(a,b,a,b));append!(entries,(G[k],-G[k],-G[k],G[k]))
        offset[a]+=E[k];offset[b]-=E[k]
    end
    for (k,j) in enumerate(h.exterior_faces)
        a=h.reconstruction.metrics.owner_a[j]
        if h.boundary_kind[k] == :dirichlet
            bg[k]=mob[a]*h.boundary_beta[k]
            be[k]=-mob[a]*dot(g[a],h.boundary_remainder[k])
            push!(rows,a);push!(cols,a);push!(entries,bg[k])
            offset[a]+=be[k]-bg[k]*values[k]
        else
            bg[k]=zero(T);be[k]=zero(T)
        end
    end
    all(x -> isfinite(x) && x > 0,G) && all(x -> isfinite(x) && x > 0,mobility) &&
        all(isfinite,E) && all(isfinite,bg) && all(isfinite,be) && all(isfinite,offset) ||
        _reconstruction_error("invalid pressure term conversion")
    all(k -> h.boundary_kind[k] == :neumann || bg[k] > 0,eachindex(bg)) || _reconstruction_error("invalid boundary conductance")
    A=sparse(rows,cols,entries,n,n)
    all(isfinite,nonzeros(A)) || _reconstruction_error("nonfinite assembled matrix")
    stored_values=T.(values)
    all(isfinite,stored_values) || _reconstruction_error("nonfinite boundary value conversion")
    CircularPressureTerms(deepcopy(h),mobility,G,E,bg,be,stored_values,A,offset)
end

function _pressure_balance(h,faces,overlaps)
    m=h.reconstruction.metrics; c=m.coupling
    length(faces)==length(m.owner_a) && length(overlaps)==length(c.measure) ||
        throw(DimensionMismatch("incorrect face/overlap flux length"))
    all(j -> m.interface_face[j] || isfinite(faces[j]),eachindex(faces)) && all(isfinite,overlaps) ||
        _reconstruction_error("nonfinite flux inputs")
    balance=zeros(eltype(h.beta),length(m.volume))
    interface_cell_flux!(balance,c,overlaps)
    for j in eachindex(faces)
        m.interface_face[j] && continue
        a,b=m.owner_a[j],m.owner_b[j]
        balance[a]+=faces[j]
        a==b || (balance[b]-=faces[j])
    end
    all(isfinite,balance) || _reconstruction_error("nonfinite flux balance")
    balance
end

"""Return pressure-only fluxes `(faces, overlaps, balance, divergence)` from the
SAME frozen terms used in assembly. Physical interface face entries are diagnostic
signed sums, never counted again in balance. Divergence uses corrected volumes."""
function circular_pressure_flux(t::CircularPressureTerms,p::AbstractVector)
    h=t.geometry; m=h.reconstruction.metrics
    _pressure_data(h,p,t.boundary_values)
    faces=zeros(eltype(h.beta),length(m.owner_a)); overlaps=zeros(eltype(h.beta),length(m.coupling.measure))
    for k in eachindex(h.edge_a)
        q=t.conductance[k]*(p[h.edge_a[k]]-p[h.edge_b[k]])+t.correction[k]
        j=h.edge_face[k]
        iszero(j) ? (overlaps[h.edge_overlap[k]]=q) : (faces[j]=q)
    end
    for (k,j) in enumerate(h.exterior_faces)
        a=m.owner_a[j]
        faces[j]=t.boundary_conductance[k]*(p[a]-t.boundary_values[k])+t.boundary_correction[k]
    end
    interface_face_flux!(faces,m.coupling,overlaps)
    all(isfinite,faces) && all(isfinite,overlaps) || _reconstruction_error("nonfinite recovered pressure flux")
    balance=_pressure_balance(h,faces,overlaps)
    divergence=balance./m.volume
    all(isfinite,divergence) || _reconstruction_error("nonfinite pressure divergence")
    (;faces,overlaps,balance,divergence)
end

function _pressure_gauges(h,predictor_balance,reference,scale,tolerance)
    n=length(predictor_balance)
    refs=Dict{Int,eltype(h.beta)}()
    for (i,v) in pairs(reference)
        i isa Integer && 1<=i<=n && isfinite(v) || _reconstruction_error("invalid pressure reference")
        refs[Int(i)]=v
        isfinite(refs[Int(i)]) || _reconstruction_error("nonfinite reference conversion")
    end
    anchored=falses(n)
    for (k,j) in enumerate(h.exterior_faces)
        h.boundary_kind[k]==:dirichlet && (anchored[h.reconstruction.metrics.owner_a[j]]=true)
    end
    for cells in h.components
        ids=[i for i in cells if haskey(refs,i)]
        if any(anchored[cells])
            isempty(ids) || _reconstruction_error("reference supplied for Dirichlet-anchored component")
        else
            length(ids)==1 || _reconstruction_error("one reference required per unanchored component")
            local_scale=scale isa AbstractVector ? sum(scale[cells]) : scale
            isfinite(local_scale) || _reconstruction_error("nonfinite component predictor scale")
            abs(sum(predictor_balance[cells])) <= tolerance*max(local_scale,one(local_scale)) ||
                _reconstruction_error("incompatible all-Neumann predictor component")
        end
    end
    refs
end

function _pressure_solve(matrix,rhs,refs)
    A=copy(matrix); b=copy(rhs)
    # Eliminate all reference columns simultaneously on the ungauged matrix.
    for (i,v) in refs
        b .-= matrix[:,i]*v
    end
    for (i,v) in refs
        A[:,i].=0; A[i,:].=0; A[i,i]=1; b[i]=v
    end
    isempty(refs) || dropzeros!(A)
    A\b
end

"""
    circular_pressure_project(geometry, mobility, boundary_values,
        predictor_faces, predictor_overlaps; initial=zeros(n), reference=Dict(),
        max_sweeps=100, tolerance=1e-10, compatibility_tolerance=1000eps(T))

Standalone static unit-density projection, direct CPU sparse solves; no SIMPLE.
Solve integrated balance(qH+qp)=0 with boundary-aware deferred corrections.
Each sweep freezes terms, solves, recovers FROM SAVED predictor, then reconstructs
and checks the refreshed full operator residual. Require BOTH recovered continuity
and refreshed residual <= tolerance (absolute integrated flux units). Return
converged=false on iteration exhaustion, never silently accept a frozen solve.
All-Neumann compatibility/reference checked per connected component; symmetric
row/column elimination fixes gauges but diagnostics use original ungauged balance.
Interface entries of predictor_faces are ignored (may be NaN); overlaps are
mandatory. Neumann pressure cannot change exterior predictor fluxes.
No inputs are mutated; result includes final frozen terms for reproducible recovery,
pressure, corrected faces/overlaps, divergence, residual history and convergence.
No cell velocity correction or variable-density support. Arrays read-only by contract.
"""
function circular_pressure_project(h::CircularPressureGeometry,mob::AbstractVector,values::AbstractVector,
    predictor_faces::AbstractVector,predictor_overlaps::AbstractVector;
    initial=zeros(eltype(h.beta),length(h.reconstruction.points)),reference=Dict{Int,Float64}(),
    max_sweeps=100,tolerance=1e-10,compatibility_tolerance=1000eps(eltype(h.beta)))
    max_sweeps isa Integer && max_sweeps>0 || _reconstruction_error("positive sweep count required")
    isfinite(tolerance) && tolerance>0 && isfinite(compatibility_tolerance) && compatibility_tolerance>=0 ||
        _reconstruction_error("invalid projection tolerances")
    _pressure_data(h,initial,values)
    _circular_input(predictor_faces,predictor_overlaps)
    pred=_pressure_balance(h,predictor_faces,predictor_overlaps)
    m=h.reconstruction.metrics
    scale=zeros(eltype(h.beta),length(pred))
    for j in eachindex(predictor_faces)
        m.interface_face[j] && continue
        a,b=m.owner_a[j],m.owner_b[j]
        scale[a]+=abs(predictor_faces[j])
        a==b || (scale[b]+=abs(predictor_faces[j]))
    end
    for k in eachindex(predictor_overlaps)
        scale[m.coupling.cell_a[k]]+=abs(predictor_overlaps[k])
        scale[m.coupling.cell_b[k]]+=abs(predictor_overlaps[k])
    end
    all(isfinite,scale) || _reconstruction_error("nonfinite predictor scale")
    refs=_pressure_gauges(h,pred,reference,scale,compatibility_tolerance)
    T=eltype(h.beta); p=T.(initial); g=fill(zero(SVector{3,T}),length(p))
    history=NamedTuple{(:continuity,:refreshed),Tuple{T,T}}[]
    terms=nothing; corrected=nothing; converged=false
    for sweep in 1:max_sweeps
        circular_pressure_reconstruct!(g,h,p,values)
        terms=circular_pressure_terms(h,g,mob,values)
        p=_pressure_solve(terms.matrix,-pred-terms.offset,refs)
        all(isfinite,p) || _reconstruction_error("nonfinite solved pressure")
        flux=circular_pressure_flux(terms,p)
        faces=T.(predictor_faces); overlaps=T.(predictor_overlaps)+flux.overlaps
        for j in eachindex(faces)
            h.reconstruction.metrics.interface_face[j] || (faces[j]+=flux.faces[j])
        end
        interface_face_flux!(faces,h.reconstruction.metrics.coupling,overlaps)
        balance=_pressure_balance(h,faces,overlaps)
        divergence=balance./h.reconstruction.metrics.volume
        all(isfinite,divergence) || _reconstruction_error("nonfinite corrected divergence")
        corrected=(;faces,overlaps,balance,divergence)
        circular_pressure_reconstruct!(g,h,p,values)
        refreshed=circular_pressure_terms(h,g,mob,values)
        residual=refreshed.matrix*p+refreshed.offset+pred
        all(isfinite,residual) || _reconstruction_error("nonfinite refreshed residual")
        continuity,full=maximum(abs,balance),maximum(abs,residual)
        push!(history,(continuity=continuity,refreshed=full))
        if continuity<=tolerance && full<=tolerance
            converged=true
            break
        end
    end
    (;pressure=p,corrected...,terms,history,iterations=length(history),converged)
end
