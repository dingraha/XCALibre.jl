export circular_momentum_reconstructed_predictor_flux

"""
    circular_momentum_reconstructed_predictor_flux(m, H0, pressure_velocity, values)

Shared static CPU predictor: reconstruct the provisional velocity
`H0-pressure_velocity` with velocity boundary-aware LSQ; retain LOW-ORDER
interpolation of the supplied pressure-velocity contribution. In SIMPLE the
latter is `rD*GaussGradient(p_old)` at the current physical mobility, so the
provisional velocity is consistent with the physical cell correction, not the
old source velocity. No second mobility multiplication.

Ordinary internal traces blend owner affine polynomials. Circular overlaps use
exact area moments componentwise, one A-outward integrated flux shared with B.
Dirichlet exterior flux remains prescribed; Neumann exterior uses owner H0 and
rejects inflow, as in the low-order predictor. Prescribed total-pressure derivative
compensation remains the caller's responsibility. No limiter, ALE or general
stability guarantee. Inputs are not mutated; returned arrays own their storage.
"""
function circular_momentum_reconstructed_predictor_flux(m::CircularMomentumGeometry,
    H0::AbstractVector,pressure_velocity::AbstractVector,values::AbstractVector)
    _momentum_values(m,H0,values)
    _momentum_vectors(pressure_velocity,length(H0),"pressure velocity")
    provisional=H0-pressure_velocity
    gradients=circular_velocity_reconstruct(m,provisional,values)
    h=m.velocity;r=h.reconstruction;T=eltype(h.beta)
    # Retains the original exterior treatment and its backflow validation.
    result=circular_momentum_predictor_flux(m,H0,values)
    for e in eachindex(h.edge_a)
        j=h.edge_face[e];iszero(j) && continue
        a,b,w,x=h.edge_a[e],h.edge_b[e],h.weight_a[e],h.edge_point[e]
        ua=provisional[a]+SVector(dot(gradients[1][a],x-r.points[a]),dot(gradients[2][a],x-r.points[a]),zero(T))
        ub=provisional[b]+SVector(dot(gradients[1][b],x-r.points[b]),dot(gradients[2][b],x-r.points[b]),zero(T))
        trace=w*ua+(1-w)*ub+w*pressure_velocity[a]+(1-w)*pressure_velocity[b]
        result.faces[j]=dot(trace,h.edge_area[e])
    end
    integrated=ntuple(2) do d
        out=fill(zero(SVector{3,T}),length(result.overlaps))
        circular_reconstructed_pressure_area!(out,h.overlap_geometry,getindex.(provisional,d),gradients[d])
        out
    end
    for k in eachindex(result.overlaps)
        c,w=r.metrics.coupling,h.overlap_geometry.weight_a[k]
        contribution=w*pressure_velocity[c.cell_a[k]]+(1-w)*pressure_velocity[c.cell_b[k]]
        result.overlaps[k]=integrated[1][k][1]+integrated[2][k][2]+dot(contribution,c.area_vector[k])
    end
    interface_face_flux!(result.faces,r.metrics.coupling,result.overlaps)
    all(isfinite,result.faces) && all(isfinite,result.overlaps) || _reconstruction_error("nonfinite reconstructed predictor flux")
    result
end
