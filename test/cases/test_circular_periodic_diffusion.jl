module CircularPeriodicDiffusionTests
using Test, XCALibre, LinearAlgebra
include(pkgdir(XCALibre,"examples","interface_verification","circular_periodic_diffusion.jl"))
const D=CircularPeriodicDiffusion

# Numerical certificates only: proposed accuracy acceptance is evaluated in the
# retained study, not silently converted to a passing/relaxed accuracy test.
@testset "Manufactured periodic diffusion integration certificates" begin
    for (tag,phase) in ((:matched,0.),(:nonmatching,0.),(:nonmatching,.13),(:nonmatching,1.2),(:stitched,0.)), scale in (1,2,4)
        q=D.CS.build(na=16scale,nb=(tag==:nonmatching ? 20 : 16)*scale,nra=3scale,nrb=3scale,phase=phase,stitched=tag==:stitched)
        r=D.evaluate(q);c=r.certificates
        @test c.conservation<1e-12
        @test c.volume_boundary<1e-8
        @test c.volume_quadrature<1e-8
        @test c.boundary_quadrature<1e-8
        @test c.volume_measure<1e-12
        if tag!=:stitched
            @test c.identity<1e-12
        else
            @test isnan(c.identity) # never report a nonexistent matrix certificate
        end
        @test all(isfinite,r.error) && all(isfinite,r.edge_error)
        @test abs(sum(r.reference))<1e-12 # no source compatibility projection
        ms=Dict(m.region=>m for m in r.metrics)
        @test sum(ms[k].count for k in ("circle","seam","bulk"))==length(r.p)
        @test sum(ms[k].volume for k in ("circle","seam","bulk"))≈1. atol=1e-13
        @test ms["all"].rms>0 # accuracy is nontrivial
        # Independent boundary oracle sign/periodicity and gamma=1 PDE identity.
        p=D.field.(r.h.points)
        @test maximum(abs,D.source.(r.h.points) .- (8pi^2) .* p)<1e-13
        println("diffusion certificates tag=$tag phase=$phase scale=$scale $c")
    end
    @test_throws ArgumentError D.evaluate(D.CS.build(L=2.,radius=.5,core_half=.2))
end
end
