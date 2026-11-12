export CircularMomentumGeometry, construct_circular_momentum_geometry
export circular_velocity_reconstruct, CircularMomentumTerms, circular_momentum_terms
export circular_momentum_flux, circular_momentum_solve, circular_momentum_predictor_flux

"""Independent static CPU momentum snapshot. Pressure and velocity share the same
validated geometry but have separate boundary-aware reconstruction matrices.
Velocity kinds follow exterior_faces: :dirichlet or homogeneous :neumann.
Cartesian x/y, zero z, unit density; arrays read-only by contract. No solver
continuation, SIMPLE, motion, body forces, transpose stress or transient terms."""
struct CircularMomentumGeometry{H}
    pressure::H
    velocity::H
end

"""
    construct_circular_momentum_geometry(pressure_geometry, velocity_kinds;
                                        condition_limit=1e6)

Copy a validated pressure snapshot without rebuilding geometry with potentially
different tolerances. Recompute velocity LSQ normal matrices with its OWN boundary
kinds. Neumann rows constrain normal derivatives, not displaced pressure samples.
All ordinary exterior faces must be covered, in pressure_geometry.exterior_faces
order. Rebuild after geometry/ID/kind changes; snapshots do not track live meshes.
"""
function construct_circular_momentum_geometry(h::CircularPressureGeometry,kinds::AbstractVector;
    condition_limit=1e6)
    _circular_input(kinds)
    length(kinds)==length(h.exterior_faces) || throw(DimensionMismatch("incorrect velocity boundary coverage"))
    all(k -> k in (:dirichlet,:neumann),kinds) || _reconstruction_error("unsupported velocity boundary kind")
    isfinite(condition_limit) && condition_limit>=1 || _reconstruction_error("invalid condition limit")
    hp=deepcopy(h); r=hp.reconstruction; T=eltype(hp.beta)
    inverse=SMatrix{2,2,T,4}[]
    for i in eachindex(r.points)
        G=zero(SMatrix{2,2,T,4})
        for j in r.neighbours[i]
            d=SVector{2,T}((r.points[j]-r.points[i])[1:2])
            G+=d*d'/dot(d,d)
        end
        for j in r.exterior_rows[i]
            v=kinds[j]==:neumann ? normalize(r.metrics.face_area_vector[r.exterior_faces[j]]) :
                r.exterior_points[j]-r.points[i]
            d=SVector{2,T}(v[1],v[2]); G+=d*d'/dot(d,d)
        end
        push!(inverse,_circular_inverse(G,condition_limit,i))
    end
    fields=map(fieldnames(typeof(hp))) do name
        name==:boundary_kind ? collect(kinds) : name==:inverse ? inverse : deepcopy(getproperty(hp,name))
    end
    hv=CircularPressureGeometry(fields...)
    CircularMomentumGeometry(hp,hv)
end

function _momentum_vectors(x,n,label)
    _circular_input(x)
    length(x)==n || throw(DimensionMismatch("incorrect $label length"))
    eltype(x)<:SVector{3,<:AbstractFloat} && all(v -> all(isfinite,v) && iszero(v[3]),x) ||
        _reconstruction_error("$label must contain finite Cartesian xy SVector{3} values")
end
function _momentum_values(m,U,values)
    h=m.velocity
    _momentum_vectors(U,length(h.reconstruction.points),"velocity")
    _momentum_vectors(values,length(h.exterior_faces),"velocity boundary data")
    all(j -> h.boundary_kind[j]==:dirichlet || iszero(values[j]),eachindex(values)) ||
        _reconstruction_error("Neumann velocity boundary entries must be zero")
end

"""Return x/y component gradient arrays (each SVector{3}, zero derivative.z).
Homogeneous normal-gradient velocity data must be zero vectors. No mutation of
inputs; cell velocities are point samples at existing mesh centres."""
function circular_velocity_reconstruct(m::CircularMomentumGeometry,U::AbstractVector,values::AbstractVector)
    _momentum_values(m,U,values)
    T=eltype(m.velocity.beta)
    ntuple(2) do j
        out=fill(zero(SVector{3,T}),length(U))
        circular_pressure_reconstruct!(out,m.velocity,getindex.(U,j),getindex.(values,j))
        out
    end
end

"""Frozen Upwind + corrected constant-viscosity momentum operator. `matrix` is
common to x/y; `offset` is the integrated convective/viscous boundary and explicit
flux balance. `pressure_force` is -Vcurved*reconstructed_Gauss_gradient(p), saved
explicitly, not found by source index. Physical equation: matrix*U+offset=force.
All arrays read-only by contract. Convection retains every signed overlap flux.
Neumann velocity inflow is rejected; zero-gradient is NOT an impermeable wall."""
struct CircularMomentumTerms{M,A,V,S,F,G}
    geometry::M
    matrix::A
    offset::V
    pressure_force::V
    boundary_values::V
    viscosity::S
    faces::F
    overlaps::F
    diffusion::G
end

function _momentum_balance(h,faces,overlaps)
    m=h.reconstruction.metrics; c=m.coupling
    balance=fill(zero(eltype(faces)),length(m.volume))
    interface_cell_flux!(balance,c,overlaps)
    for j in eachindex(faces)
        m.interface_face[j] && continue
        a,b=m.owner_a[j],m.owner_b[j]
        balance[a]+=faces[j]
        a==b || (balance[b]-=faces[j])
    end
    all(v -> all(isfinite,v),balance) || _reconstruction_error("nonfinite momentum balance")
    balance
end

function _momentum_freeze(m,U,values,faces,overlaps,nu,force)
    _momentum_values(m,U,values)
    h=m.velocity; T=eltype(h.beta); n=length(U)
    _circular_input(faces,overlaps)
    _pressure_balance(h,faces,overlaps) # dimensions/finite noninterface flux checks
    isfinite(nu) && nu>0 || _reconstruction_error("positive finite constant viscosity required")
    _momentum_vectors(force,n,"pressure force")
    gradients=circular_velocity_reconstruct(m,U,values)
    diffusion=ntuple(j -> circular_pressure_terms(h,gradients[j],fill(nu,n),getindex.(values,j)),2)
    diffusion[1].matrix==diffusion[2].matrix || _reconstruction_error("x/y implicit diffusion differs")
    A=copy(diffusion[1].matrix)
    offset=[SVector(diffusion[1].offset[i],diffusion[2].offset[i],zero(T)) for i in 1:n]
    rows,cols=Int[],Int[]; entries=T[]
    for k in eachindex(h.edge_a)
        a,b=h.edge_a[k],h.edge_b[k]
        j=h.edge_face[k]; q=iszero(j) ? overlaps[h.edge_overlap[k]] : faces[j]
        ca,cb=max(q,zero(q)),min(q,zero(q))
        append!(rows,(a,a,b,b));append!(cols,(a,b,a,b));append!(entries,(ca,cb,-ca,-cb))
    end
    for (k,j) in enumerate(h.exterior_faces)
        a=h.reconstruction.metrics.owner_a[j]; q=faces[j]
        h.boundary_kind[k]==:neumann && q<0 && _reconstruction_error("Neumann velocity inflow at face $j")
        # Upwind exterior: prescribed inflow, owner outflow, including Dirichlet.
        if q<0
            offset[a]+=q*values[k]
        else
            push!(rows,a);push!(cols,a);push!(entries,q)
        end
    end
    A+=sparse(rows,cols,entries,n,n)
    D=diag(A)
    all(x -> isfinite(x) && x>0,D) || _reconstruction_error("positive finite physical momentum diagonals required")
    all(isfinite,nonzeros(A)) && all(v -> all(isfinite,v),offset) || _reconstruction_error("nonfinite momentum assembly")
    storedvalues=SVector{3,T}.(values); storedforce=SVector{3,T}.(force)
    storedfaces=T.(faces); storedoverlaps=T.(overlaps)
    # Ignore diagnostic interface-face entries; rebuild them from individual fluxes.
    interface_face_flux!(storedfaces,h.reconstruction.metrics.coupling,storedoverlaps)
    all(v -> all(isfinite,v),storedvalues) && all(v -> all(isfinite,v),storedforce) &&
        all(isfinite,storedfaces) && all(isfinite,storedoverlaps) || _reconstruction_error("nonfinite momentum snapshot conversion")
    CircularMomentumTerms(deepcopy(m),A,offset,storedforce,storedvalues,T(nu),storedfaces,storedoverlaps,diffusion)
end

"""
    circular_momentum_terms(momentum_geometry, velocity, velocity_boundary_values,
        pressure, pressure_boundary_values, faces, overlaps, viscosity)

Freeze convection fluxes, reconstructed component viscous corrections and pressure
force together. Physical interface face entries are ignored; individual overlaps
are mandatory. Explicit flux corrections use existing velocity for reconstruction.
No body forces or volumetric source API. Pressure force uses the complete shared
pressure-area chain and corrected volume, never polygonal source-removal helpers.
"""
function circular_momentum_terms(m::CircularMomentumGeometry,U::AbstractVector,values::AbstractVector,
    p::AbstractVector,pvalues::AbstractVector,faces::AbstractVector,overlaps::AbstractVector,nu::Real)
    T=eltype(m.pressure.beta)
    gradient=fill(zero(SVector{3,T}),length(m.pressure.reconstruction.points))
    circular_pressure_gradient!(gradient,m.pressure,p,pvalues)
    force=[-m.pressure.reconstruction.metrics.volume[i]*gradient[i] for i in eachindex(gradient)]
    _momentum_freeze(m,U,values,faces,overlaps,nu,force)
end

"""Recover frozen vector convection+viscous fluxes, A-outward on every edge.
Includes no pressure force in flux arrays. `balance == matrix*U+offset`
componentwise; physical residual is balance-pressure_force. Diagnostic interface
face sums do not enter balance a second time. No caller arrays are mutated."""
function circular_momentum_flux(t::CircularMomentumTerms,U::AbstractVector)
    m=t.geometry; h=m.velocity
    _momentum_values(m,U,t.boundary_values)
    T=eltype(h.beta)
    viscous=ntuple(j -> circular_pressure_flux(t.diffusion[j],getindex.(U,j)),2)
    faces=[SVector(viscous[1].faces[k],viscous[2].faces[k],zero(T)) for k in eachindex(t.faces)]
    overlaps=[SVector(viscous[1].overlaps[k],viscous[2].overlaps[k],zero(T)) for k in eachindex(t.overlaps)]
    for k in eachindex(h.edge_a)
        a,b=h.edge_a[k],h.edge_b[k]; j=h.edge_face[k]
        q=iszero(j) ? t.overlaps[h.edge_overlap[k]] : t.faces[j]
        convection=q*(q>=0 ? U[a] : U[b])
        iszero(j) ? (overlaps[h.edge_overlap[k]]+=convection) : (faces[j]+=convection)
    end
    for (k,j) in enumerate(h.exterior_faces)
        q=t.faces[j]; a=h.reconstruction.metrics.owner_a[j]
        faces[j]+=q*(q>=0 ? U[a] : t.boundary_values[k])
    end
    interface_face_flux!(faces,h.reconstruction.metrics.coupling,overlaps)
    all(v -> all(isfinite,v),faces) && all(v -> all(isfinite,v),overlaps) || _reconstruction_error("nonfinite momentum flux")
    balance=_momentum_balance(h,faces,overlaps)
    residual=balance-t.pressure_force
    all(v -> all(isfinite,v),residual) || _reconstruction_error("nonfinite momentum residual")
    (;faces,overlaps,balance,residual)
end

"""
    circular_momentum_predictor_flux(momentum_geometry, H0, velocity_boundary_values)

Low-order geometric blend of velocity-like H0 dotted with integrated S, NOT an
exact affine curved velocity integral. Never multiply by mobility again. Exterior
Dirichlet flux uses prescribed velocity; Neumann uses owner H0 and rejects inflow.
Return ordinary faces and individual overlaps (interface face entries are sums).
This is NOT corrected continuity flux; a coupled loop must retain pressure-corrected
fluxes for subsequent convection rather than reconstructing from cell velocity.
"""
function circular_momentum_predictor_flux(m::CircularMomentumGeometry,H0::AbstractVector,values::AbstractVector)
    _momentum_values(m,H0,values)
    h=m.velocity; T=eltype(h.beta)
    faces=zeros(T,length(h.reconstruction.metrics.owner_a))
    overlaps=zeros(T,length(h.reconstruction.metrics.coupling.measure))
    for k in eachindex(h.edge_a)
        a,b,w=h.edge_a[k],h.edge_b[k],h.weight_a[k]
        q=dot(w*H0[a]+(1-w)*H0[b],h.edge_area[k])
        j=h.edge_face[k]
        iszero(j) ? (overlaps[h.edge_overlap[k]]=q) : (faces[j]=q)
    end
    for (k,j) in enumerate(h.exterior_faces)
        a=h.reconstruction.metrics.owner_a[j]
        q=dot(h.boundary_kind[k]==:dirichlet ? values[k] : H0[a],h.reconstruction.metrics.face_area_vector[j])
        h.boundary_kind[k]==:neumann && q<0 && _reconstruction_error("Neumann predictor inflow at face $j")
        faces[j]=q
    end
    interface_face_flux!(faces,h.reconstruction.metrics.coupling,overlaps)
    all(isfinite,faces) && all(isfinite,overlaps) || _reconstruction_error("nonfinite momentum predictor flux")
    (;faces,overlaps)
end

"""
    circular_momentum_solve(terms, old_velocity; relaxation=0.7)

ONE frozen relaxed direct CPU sparse solve, not a deferred momentum iteration.
Require a prescribed velocity boundary in each component (no singular all-Neumann
solve). Save D=diag(A), Dt=max(abs(D),sum(abs(offdiagonal)))/relaxation and augment
RHS by (Dt-D)*oldU. Undo complete augmentation/source in pressure-free H/aP to
recover H0, and rD=Vcurved/D. Return relaxed velocity, physical/relaxed diagonals,
H0, mobility, frozen linear residual, physical and refreshed momentum residuals.
These residuals are integrated units; one frozen solve is NOT convergence of the
physical corrected operator. H0 equivalence across relaxation is at identical
velocity/terms, not a claim that different relaxed solves give identical velocity.
Predictor interpolation is an explicit separate call (may reject Neumann inflow).
No inputs mutated; nonfinite/singular solve throws, no persistent state to corrupt.
"""
function circular_momentum_solve(t::CircularMomentumTerms,old::AbstractVector;relaxation=0.7)
    _momentum_values(t.geometry,old,t.boundary_values)
    isfinite(relaxation) && 0<relaxation<=1 || _reconstruction_error("momentum relaxation must lie in (0,1]")
    h=t.geometry.velocity; n=length(old); T=eltype(h.beta)
    anchored=falses(n)
    for (k,j) in enumerate(h.exterior_faces)
        h.boundary_kind[k]==:dirichlet && (anchored[h.reconstruction.metrics.owner_a[j]]=true)
    end
    all(cells -> any(anchored[cells]),h.components) || _reconstruction_error("prescribed velocity required per component")
    A=t.matrix; D=Vector(diag(A))
    all(x -> isfinite(x) && x>0,D) || _reconstruction_error("invalid physical diagonal")
    # Sum off-diagonal magnitudes directly, avoiding cancellation against D.
    rows=zeros(T,n)
    for j in 1:n, entry in nzrange(A,j)
        i=rowvals(A)[entry]
        i==j || (rows[i]+=abs(nonzeros(A)[entry]))
    end
    Dt=max.(abs.(D),rows)./relaxation
    all(x -> isfinite(x) && x>0,Dt) || _reconstruction_error("invalid relaxed diagonal")
    relaxed=A+spdiagm(0=>Dt-D)
    U=fill(zero(SVector{3,T}),n); H=similar(U)
    frozen=similar(U)
    for component in 1:2
        b=getindex.(t.pressure_force-t.offset,component)+(Dt-D).*getindex.(old,component)
        u=relaxed\b
        # Remove saved pressure force exactly, rather than assuming source index 1.
        off=A*u-D.*u
        Hv=(b-getindex.(t.pressure_force,component)-off)./Dt
        h0=(Dt.*Hv-(Dt-D).*getindex.(old,component))./D
        res=relaxed*u-b
        for i in 1:n
            U[i]=setindex(U[i],u[i],component)
            H[i]=setindex(component==1 ? zero(SVector{3,T}) : H[i],h0[i],component)
            frozen[i]=setindex(component==1 ? zero(SVector{3,T}) : frozen[i],res[i],component)
        end
    end
    mobility=h.reconstruction.metrics.volume./D
    all(x -> isfinite(x) && x>0,mobility) && all(v -> all(isfinite,v),U) &&
        all(v -> all(isfinite,v),H) && all(v -> all(isfinite,v),frozen) || _reconstruction_error("nonfinite momentum solve/predictor")
    physical=circular_momentum_flux(t,U).residual
    refreshed=_momentum_freeze(t.geometry,U,t.boundary_values,t.faces,t.overlaps,t.viscosity,t.pressure_force)
    full=circular_momentum_flux(refreshed,U).residual
    (;velocity=U,H0=H,mobility,physical_diagonal=D,relaxed_diagonal=Dt,
        frozen_residual=frozen,physical_residual=physical,refreshed_residual=full)
end
