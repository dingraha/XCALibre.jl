using XCALibre
using Test
using LinearAlgebra
using Accessors

module PhysicalPredictorSetup
using XCALibre
include(pkgdir(XCALibre,"examples/interface_verification/channel_setup.jl"))
end

@testset "Physical planar momentum predictor validation" begin
    model,cfg = PhysicalPredictorSetup.channel_case()
    for alpha in (0.0,1.1,NaN)
        for name in (:U,:p)
            bad = cfg
            if name === :U
                @reset bad.solvers.U.relax = alpha
            else
                @reset bad.solvers.p.relax = alpha
            end
            @test_throws ArgumentError PlanarSimpleState(model,bad)
        end
    end
end

@testset "Physical planar momentum predictor sparse-row equivalence" begin
    for alpha in (0.5,0.8,1.0)
        model,cfg = PhysicalPredictorSetup.channel_case(na=4,nb=7,nx=3,relaxU=alpha)
        s = PlanarSimpleState(model,cfg)
        w,U,p = s.work,model.momentum.U,model.momentum.p
        # Deliberately divergence-producing initial transport exercises diagonal
        # dominance augmentation even at alpha=1, and opposing overlaps.
        q = [isodd(k) ? 0.3 : -0.1 for k in eachindex(w.transport.overlaps)]
        update_interface_flux!(w.transport,q)
        for i in eachindex(p.values)
            x,y = model.domain.cell_centre[i]
            p.values[i] = x + 0.2y
            U.x.values[i] = 0.4+0.1y
            U.y.values[i] = 0.03x
        end
        grad!(w.gradp,w.pf,p,cfg.boundaries.p,1,cfg)
        XCALibre.Solvers._planar_solve_momentum!(w,U,cfg,1)
        remove_pressure_source!(w.Ueqn,w.gradp,cfg)
        H!(w.Hv,U,w.Ueqn,cfg)
        relaxed_h = copy(w.Hv.x.values)
        XCALibre.Solvers._planar_physical_predictor!(w)
        expected = (w.relaxed_diagonal .* relaxed_h .- (w.relaxed_diagonal-w.physical_diagonal).*w.oldU.x.values)./w.physical_diagonal
        @test w.Hv.x.values ≈ expected atol=1e-13
        @test w.rD.values ≈ model.domain.cell_volume ./ w.physical_diagonal atol=1e-14
        @test any(w.relaxed_diagonal .> w.physical_diagonal)
        if alpha == 1
            @test any(abs.(w.relaxed_diagonal-w.physical_diagonal) .> 1e-10)
        end
        # Independent unrelaxed reassembly with the unchanged frozen flux.
        eq = w.Ueqn
        discretise!(eq,U,cfg;time=1)
        for (dir,field,h,bname) in ((XDir(),U.x,w.Hv.x,:bx),(YDir(),U.y,w.Hv.y,:by))
            update_equation!(eq,cfg)
            apply_boundary_conditions!(eq,cfg.boundaries.U,dir,1,cfg)
            A = Matrix(eq.equation.A.parent)
            b = copy(getproperty(eq.equation,bname))
            g = dir isa XDir ? w.gradp.result.x.values : w.gradp.result.y.values
            b .+= model.domain.cell_volume .* g # remove -Source(gradp)
            d = diag(A)
            reference = (b - A*field.values + d.*field.values)./d
            @test d == w.physical_diagonal
            @test h.values ≈ reference atol=1e-12 rtol=1e-12
            c = w.transport.coupling
            @test all(A[a,bb] != 0 for (a,bb) in zip(c.cell_a,c.cell_b))
        end
    end
end
