using Test
using XCALibre

# Isolated module so example helpers cannot collide with other test files.
module TaylorGreenAnalyticalTests
using Test, XCALibre
include(pkgdir(XCALibre, "examples", "transient_verification", "taylor_green_analytical.jl"))
const TG = TaylorGreenAnalytical

@testset "Taylor–Green analytical oracle" begin
    k, nu, t = 2pi, 0.01, 0.2
    a = exp(-2nu*k^2*t)
    for (x,y) in ((0.13,0.27), (0.41,0.68), (0.77,0.91))
        q = TG.exact(x,y,t)
        ux, uy = a*k*cos(k*x)*cos(k*y), -a*k*sin(k*x)*sin(k*y)
        vx, vy = a*k*sin(k*x)*sin(k*y), -a*k*cos(k*x)*cos(k*y)
        px, py = -a^2*k/2*sin(2k*x), -a^2*k/2*sin(2k*y)
        @test abs(ux+vy) < 2e-14
        @test abs(q.ux*ux+q.uy*uy+px) < 2e-14
        @test abs(q.ux*vx+q.uy*vy+py) < 2e-14
        @test TG.exact(x+1,y,t).ux ≈ q.ux atol=2e-14
        @test TG.exact(x,y+1,t).p ≈ q.p atol=2e-14
        h = 1e-5
        derivative = (TG.exact(x,y,t+h).ux-TG.exact(x,y,t-h).ux)/(2h)
        @test derivative ≈ -2nu*k^2*q.ux atol=2e-10
    end
    target = [TG.exact(x,y,t) for (x,y) in ((0.13,0.27),(0.41,0.68),(0.77,0.91))]
    u, v, p = getproperty.(target,:ux), getproperty.(target,:uy), getproperty.(target,:p)
    e = TG.errors(u,v,p .+ 17.,target,[1.,2.,3.])
    @test e.u_rms == 0
    @test e.p_max < 5e-15
    @test e.gauge ≈ 17.
    @test_throws ArgumentError TG.run_case(dt=0.003, tend=0.2)
end

@testset "Ordinary periodic analytical Taylor–Green" begin
    # Gates chosen before execution: small-grid accuracy only, not an order claim.
    r = TG.run_case(n=16, dt=0.001, tend=0.05)
    @test r.finite
    @test all(iszero, r.uz)
    @test r.error.u_relative < 0.02
    @test r.error.p_relative < 0.15
    @test maximum(r.pressure)-minimum(r.pressure) > 0.1
    @test r.conservation.continuity_scaled < 1e-9
    @test r.conservation.divergence_max < 1e-8
    @test r.conservation.pair_max < 1e-11
    @test abs(r.conservation.global_balance) < 1e-11
    TG.print_result(r)
end
end
