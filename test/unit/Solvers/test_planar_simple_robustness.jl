using XCALibre
using Test
using StaticArrays
using Accessors
using LinearAlgebra

# Shared shipped setup, isolated from the test namespace and without executing examples.
module PlanarVerificationSetup
using XCALibre
include(joinpath(pkgdir(XCALibre),"examples/interface_verification/channel_setup.jl"))
end

@testset "Planar SIMPLE relaxation and initial-state robustness" begin
    baseline = nothing
    for (u,p,relaxU,relaxp) in ((0.7,0.,0.7,0.3),(0.2,2.,0.7,0.3),(0.2,2.,0.5,0.2),(1.1,-1.,0.8,0.4))
        model,cfg = PlanarVerificationSetup.channel_case(na=6,nb=11,nx=4,interface_x=0.65,
            initialU=u,initialp=p,relaxU=relaxU,relaxp=relaxp)
        state = PlanarSimpleState(model,cfg;continuity=1e-9)
        r = planar_simple!(state)
        metrics = PlanarVerificationSetup.channel_metrics(model,cfg,state.work.transport.values;transport=state.work.transport)
        @test r.converged
        @test metrics.continuity < 1e-9
        @test abs(metrics.interface_balance) < 1e-12
        @test abs(metrics.boundary_balance) < 1e-10
        @test metrics.unequal_distances
        if baseline === nothing
            baseline = (copy(model.momentum.U.x.values),copy(model.momentum.U.y.values),copy(model.momentum.p.values),copy(state.work.transport.overlaps))
        else
            for (a,b) in zip((model.momentum.U.x.values,model.momentum.U.y.values,model.momentum.p.values,state.work.transport.overlaps),baseline)
                if relaxU == 0.7 && relaxp == 0.3
                    @test a ≈ b atol=2e-6 rtol=2e-6
                else
                    # Characterise, not conceal, under-relaxed predictor-flux sensitivity.
                    @test maximum(abs,a-b) < 0.02
                end
            end
        end
        @test metrics.velocity_error < 0.06
        @test metrics.pressure_error < 0.08
    end
end

function signed_couette(; offset=0.,fail=Ref(false),convergence=1e-8)
    model,cfg = PlanarVerificationSetup.channel_case(na=5,nb=8,nx=3,interface_x=0.8,convergence=convergence)
    pair = construct_planar_transport(model.domain,:interface_a,:interface_b)
    pressure = construct_planar_pressure(pair)
    # Callback is valid at construction time, but can fail after momentum work starts.
    profile(x,t,i) = SVector(2x[2]-1,0.,fail[] && t > 0 ? 1. : 0.)
    boundaries = assign((U=[pair...,DirichletFunction(:inlet,profile),DirichletFunction(:outlet,profile),DirichletFunction(:walls,profile)],
        p=[pressure...,Zerogradient(:inlet),Zerogradient(:outlet),Zerogradient(:walls)]);region=model.domain)
    @reset cfg.boundaries = boundaries
    initialise!(model.momentum.U,[0.,0.,0.]); initialise!(model.momentum.p,offset)
    PlanarSimpleState(model,cfg;pref=offset,continuity=1e-9)
end

@testset "Planar SIMPLE signed overlaps and pressure gauge" begin
    reference = nothing
    for offset in (0.,3.)
        state = signed_couette(;offset)
        r = planar_simple!(state)
        m,w,cfg = state.model,state.work,state.config
        @test r.converged
        @test r.continuity[end] < 1e-9
        exact = [2x[2]-1 for x in m.domain.cell_centre]
        @test maximum(abs,m.momentum.U.x.values-exact) < 0.08
        @test minimum(w.transport.overlaps) < -0.01
        @test maximum(w.transport.overlaps) > 0.01
        @test sum(abs,w.transport.overlaps) > 0.4
        @test abs(sum(w.transport.overlaps)) < 1e-10
        c = w.transport.coupling
        qa = sum(w.transport.values[m.domain.boundaries[c.patch_a].IDs_range])
        qb = sum(w.transport.values[m.domain.boundaries[c.patch_b].IDs_range])
        @test abs(qa+qb) < 1e-12
        if reference === nothing
            reference = (copy(m.momentum.U.x.values),copy(m.momentum.U.y.values),copy(m.momentum.p.values),copy(w.transport.overlaps))
        else
            @test m.momentum.U.x.values ≈ reference[1] atol=2e-6 rtol=2e-6
            @test m.momentum.U.y.values ≈ reference[2] atol=2e-6 rtol=2e-6
            @test m.momentum.p.values .- m.momentum.p.values[1] ≈ reference[3] .- reference[3][1] atol=2e-6 rtol=2e-6
            @test w.transport.overlaps ≈ reference[4] atol=2e-6 rtol=2e-6
        end
    end
end

function copy_cell_fields!(dest,src)
    for (a,b) in zip((dest.momentum.U.x,dest.momentum.U.y,dest.momentum.U.z,dest.momentum.p),
        (src.momentum.U.x,src.momentum.U.y,src.momentum.U.z,src.momentum.p))
        copyto!(a.values,b.values)
    end
end

@testset "Planar SIMPLE failed-call invalidation and recovery" begin
    state = signed_couette(convergence=0.)
    planar_simple!(state;iterations=5)
    @test state.work.valid[]
    # Unchanged finite fields, but invalid request: even validation must invalidate.
    @test_throws ArgumentError planar_simple!(state;iterations=0)
    @test !state.work.valid[]
    fresh = signed_couette(convergence=0.)
    copy_cell_fields!(fresh.model,state.model)
    a = planar_simple!(state;iterations=6)
    b = planar_simple!(fresh;iterations=6)
    @test state.work.iteration[] == 6
    @test a.Ux == b.Ux
    @test state.work.transport.overlaps == fresh.work.transport.overlaps

    # Input-validation failure on a formerly valid state.
    state.model.momentum.U.z.values[1] = 1.
    @test_throws ArgumentError planar_simple!(state;iterations=1)
    @test !state.work.valid[]
    state.model.momentum.U.z.values[1] = 0.
    fresh = signed_couette(convergence=0.)
    copy_cell_fields!(fresh.model,state.model)
    a = planar_simple!(state;iterations=3)
    b = planar_simple!(fresh;iterations=3)
    @test a.Ux == b.Ux
    @test state.work.transport.overlaps == fresh.work.transport.overlaps

    flag = Ref(false)
    state = signed_couette(fail=flag,convergence=0.)
    planar_simple!(state;iterations=5)
    before = copy(state.model.momentum.U.x.values)
    flag[] = true
    @test_throws ArgumentError planar_simple!(state;iterations=2)
    @test !state.work.valid[]
    @test state.model.momentum.U.x.values != before # momentum work has happened
    flag[] = false
    fresh = signed_couette(convergence=0.)
    copy_cell_fields!(fresh.model,state.model)
    a = planar_simple!(state;iterations=6)
    b = planar_simple!(fresh;iterations=6)
    @test state.work.iteration[] == 6
    @test a.Ux == b.Ux
    @test a.p_absolute == b.p_absolute
    @test state.work.transport.overlaps == fresh.work.transport.overlaps
    @test state.model.momentum.p.values == fresh.model.momentum.p.values
end
