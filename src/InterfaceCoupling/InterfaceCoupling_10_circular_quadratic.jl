export PeriodicCircularQuadratic, construct_periodic_circular_quadratic
export periodic_circular_quadratic_flux

"""Read-only local point-sample recovery map in scaled xy coordinates."""
struct _PeriodicQuadraticStencil
    donors::Vector{Int}
    displacements::Vector{SVector{3,Float64}}
    scale::Float64
    recovery::Matrix{Float64}
    condition::Float64
end

"""Opt-in frozen scalar quadratic negative-Laplacian snapshot.
`edge_matrix` maps point values to shared integrated outward edge flux;
`matrix` maps the same values to integrated cell balances (NOT balance/volume).
All arrays, including the owned deep-copied geometry, are read-only by contract.
Rebuild after geometry/connectivity/periodic-pair changes. No live mutation tracking.

Restricted to static CPU Float64 filled unit-square geometry with a circular split
and unit-axis periodic translations. Not a pressure/momentum/transient/ALE API.
Nonsymmetric full reconstructed operator, NOT the old two-point matrix plus an
explicit correction. No implicit solve, gauge modification or stabilisation.
"""
struct PeriodicCircularQuadratic{H}
    geometry::H
    stencils::Vector{_PeriodicQuadraticStencil}
    edge_matrix::SparseMatrixCSC{Float64,Int}
    matrix::SparseMatrixCSC{Float64,Int}
    coefficient::Float64
end

_quadratic_basis(d) = SVector(d[1],d[2],d[1]^2/2,d[1]*d[2],d[2]^2/2)

function _periodic_quadratic_stencils(h,condition_limit,max_rows)
    stencils=_PeriodicQuadraticStencil[]
    for i in eachindex(h.points)
        seen=Set([(i,0,0)]);frontier=[(i,0,0)]
        for _ in 1:2
            next=Tuple{Int,Int,Int}[]
            for (u,sx,sy) in frontier, (v,d) in h.rows[u]
                shift=d+h.points[u]-h.points[v]
                ix,iy=round(Int,shift[1]),round(Int,shift[2])
                norm(shift-SVector(Float64(ix),Float64(iy),0.))<=1e-12 ||
                    _reconstruction_error("quadratic stencil requires unit-square periodic images")
                key=(Int(v),sx+ix,sy+iy)
                if !(key in seen);push!(seen,key);push!(next,key);end
            end
            frontier=next
        end
        delete!(seen,(i,0,0));keys=sort!(collect(seen))
        5<=length(keys)<=max_rows || _reconstruction_error("quadratic support bound failed at cell $i")
        donors=[k[1] for k in keys]
        displacements=[h.points[j]+SVector(Float64(sx),Float64(sy),0.)-h.points[i] for (j,sx,sy) in keys]
        all(d -> all(isfinite,d) && norm(d)>0,displacements) || _reconstruction_error("invalid quadratic donor displacement")
        scale=sqrt(sum(sum(abs2,d) for d in displacements)/length(displacements))
        isfinite(scale) && scale>0 || _reconstruction_error("invalid quadratic scale")
        weights=[scale/norm(d) for d in displacements]
        design=reduce(vcat,[permutedims(collect(_quadratic_basis(d/scale))) for d in displacements])
        F=svd(design.*weights)
        length(F.S)==5 && minimum(F.S)>1e-12*maximum(F.S) || _reconstruction_error("rank deficient quadratic stencil at cell $i")
        condition=maximum(F.S)/minimum(F.S)
        isfinite(condition) && condition<=condition_limit || _reconstruction_error("quadratic condition limit failed at cell $i")
        recovery=(F.V*Diagonal(1 ./ F.S)*F.U').*permutedims(weights)
        all(isfinite,recovery) || _reconstruction_error("nonfinite quadratic recovery")
        push!(stencils,_PeriodicQuadraticStencil(donors,displacements,scale,recovery,condition))
    end
    stencils
end

function _quadratic_edge_functional(h,k,owner,scale,is_a)
    F=h.point[k];S=h.area[k]
    position=h.points[owner]+(is_a ? zero(h.shift[k]) : h.shift[k]);d=F-position
    moment=iszero(h.face_a[k]) ? h.metrics.coupling.area_moment[h.overlap[k]]-F*S' : zero(SMatrix{3,3,Float64,9})
    [-S[1]/scale,-S[2]/scale,
     -(d[1]*S[1]+moment[1,1])/scale^2,
     -(d[2]*S[1]+d[1]*S[2]+moment[1,2]+moment[2,1])/scale^2,
     -(d[2]*S[2]+moment[2,2])/scale^2]
end

"""
    construct_periodic_circular_quadratic(h; coefficient=1.0, condition_limit=100.0, max_rows=80)

Deep-copy `PeriodicCircularGeometry`, construct fixed TWO-hop quadratic point-sample
stencils (inverse-distance weights, scaled xy coordinates), then assemble sparse
shared-flux and integrated balance maps. Five nonconstant coefficients; no adaptive
enlargement, regularisation, clipping, dropped modes or fallback. Donor identity
includes periodic image, not only cell ID. Limits may only tighten the qualified
bounds (condition<=100, rows<=80); rejected construction leaves input unchanged.

Scope: Float64 CPU, square [0,1]², coplanar cells, all exterior faces paired with
unit x/y translations. Positive constant finite scalar coefficient. Quadratic
reconstruction does not imply arbitrary-mesh stability; evidence covers the tested
filled-square/circle controls only. Snapshot storage includes duplicated geometry.
"""
function construct_periodic_circular_quadratic(h::PeriodicCircularGeometry;
    coefficient::Real=1.0,condition_limit::Real=100.0,max_rows::Integer=80)
    eltype(h.beta)===Float64 || _reconstruction_error("quadratic API requires Float64 geometry")
    isfinite(coefficient) && coefficient>0 || _reconstruction_error("positive finite quadratic coefficient required")
    gamma=Float64(coefficient)
    isfinite(gamma) && gamma>0 || _reconstruction_error("unrepresentable quadratic coefficient")
    isfinite(condition_limit) && 1<=condition_limit<=100 || _reconstruction_error("quadratic condition limit must lie in [1,100]")
    5<=max_rows<=80 || _reconstruction_error("quadratic row limit must lie in [5,80]")
    # Validate the documented square, rather than silently accepting arbitrary
    # periodic lattices after integer rounding of image shifts.
    all(p -> all(isfinite,p) && 0<p[1]<1 && 0<p[2]<1,h.points) || _reconstruction_error("quadratic points must lie inside unit square")
    c=h.metrics.coupling
    all(p -> abs(p[3]-c.origin[3])<=1e-12,h.points) || _reconstruction_error("quadratic points must be coplanar")
    sides=falses(4)
    for k in eachindex(h.beta)
        iszero(h.face_b[k]) && continue
        s=h.shift[k];F=h.point[k]
        abs(s[3])<=1e-12 && ((abs(abs(s[1])-1)<=1e-12 && abs(s[2])<=1e-12) ||
            (abs(abs(s[2])-1)<=1e-12 && abs(s[1])<=1e-12)) || _reconstruction_error("unit-axis periodic shift required")
        for (j,point) in ((h.face_a[k],F),(h.face_b[k],F-s))
            side=abs(point[1])<=1e-12 ? 1 : abs(point[1]-1)<=1e-12 ? 2 : abs(point[2])<=1e-12 ? 3 : abs(point[2]-1)<=1e-12 ? 4 : 0
            side!=0 || _reconstruction_error("periodic face must lie on unit-square exterior")
            normal=h.area[k]
            transverse=side<=2 ? normal[2] : normal[1]
            abs(transverse)<=1e-12*norm(normal) || _reconstruction_error("periodic face not aligned with square side")
            sides[side]=true
        end
    end
    all(sides) || _reconstruction_error("all four square sides must be paired")
    frozen=deepcopy(h)
    stencils=_periodic_quadratic_stencils(frozen,Float64(condition_limit),Int(max_rows))
    rows=Int[];cols=Int[];values=Float64[];n=length(frozen.points);ne=length(frozen.beta)
    for k in 1:ne
        a,b=frozen.edge_a[k],frozen.edge_b[k]
        for (i,w,is_a) in ((a,frozen.weight[k],true),(b,1-frozen.weight[k],false))
            stencil=stencils[i]
            row=(gamma*w).*(stencil.recovery'*_quadratic_edge_functional(frozen,k,i,stencil.scale,is_a))
            for (j,x) in zip(stencil.donors,row);push!(rows,k);push!(cols,j);push!(values,x);end
            push!(rows,k);push!(cols,i);push!(values,-sum(row))
        end
    end
    E=sparse(rows,cols,values,ne,n)
    incidence=sparse(vcat(Int.(frozen.edge_a),Int.(frozen.edge_b)),vcat(collect(1:ne),collect(1:ne)),vcat(ones(ne),-ones(ne)),n,ne)
    A=incidence*E
    all(isfinite,nonzeros(E)) && all(isfinite,nonzeros(A)) || _reconstruction_error("nonfinite quadratic assembled operator")
    PeriodicCircularQuadratic(frozen,stencils,E,A,gamma)
end

"""Recover shared edge flux from the SAME frozen quadratic snapshot as assembly.
Return integrated balances, volume-divided divergence, accumulated circular patch
fluxes and exactly opposite periodic diagnostic face entries. No solve or source.
Input must be finite point values with correct length. All returned arrays owned.
"""
function periodic_circular_quadratic_flux(t::PeriodicCircularQuadratic,p::AbstractVector)
    _circular_input(p);h=t.geometry
    length(p)==length(h.points) || throw(DimensionMismatch("incorrect quadratic sample length"))
    all(isfinite,p) || _reconstruction_error("nonfinite quadratic samples")
    q=t.edge_matrix*p
    balance=zeros(Float64,length(p));faces=zeros(Float64,length(h.metrics.owner_a));overlaps=zeros(Float64,length(h.metrics.coupling.measure))
    for k in eachindex(q)
        balance[h.edge_a[k]]+=q[k];balance[h.edge_b[k]]-=q[k]
        if iszero(h.face_a[k]);overlaps[h.overlap[k]]=q[k]
        else
            faces[h.face_a[k]]=q[k]
            iszero(h.face_b[k]) || (faces[h.face_b[k]]=-q[k])
        end
    end
    interface_face_flux!(faces,h.metrics.coupling,overlaps)
    divergence=balance./h.metrics.volume
    all(isfinite,q) && all(isfinite,divergence) || _reconstruction_error("nonfinite quadratic recovered flux")
    (;edges=q,faces,overlaps,balance,divergence)
end
