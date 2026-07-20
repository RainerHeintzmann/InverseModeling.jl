@testset "Optimizers" begin

    # ── project! in sgd_optimize (via optimize_model with Optimisers rule) ──
    @testset "project! callback" begin
        # well-conditioned diagonal system: A = I, x_true known
        A = Float32.([2 0 0 0; 0 2 0 0; 0 0 2 0; 0 0 0 2])
        x_true = Float32[3.0, 0.0, 1.5, 0.5]
        y = A * x_true
        fwd_model(g) = A * g(:x)
        start_val = (x=zeros(Float32, 4),)

        res, trace = optimize_model(
            start_val, fwd_model, y, loss_gaussian;
            optimizer=Optimisers.Descent(0.05f0),
            iterations=500,
            project! = p -> p.x .= max.(p.x, 0),
            store_trace=true)

        @test all(res.x .>= 0)
        @test isapprox(res.x, x_true, rtol=0.1)
    end

    # ── rl_optimizer via low-level optimize_model ──
    @testset "rl_optimizer" begin
        A = Float32.([2 0; 0 2])
        x_true = Float32[1.0, 0.0]
        y = A * x_true
        loss_fkt(p) = sum(abs2, A * p - y)
        start = Float32[0.1, 0.1]

        result = InverseModeling.optimize_model(
            loss_fkt, start;
            optimize=rl_optimizer(0.2f0; project! = p -> p .= max.(p, 0)),
            iterations=1000)

        @test all(result.minimizer .>= 0)
        @test result.minimizer[1] > 0.5
        @test result.minimizer[2] < 0.1
    end

    # ── dm_optimizer via low-level optimize_model ──
    @testset "dm_optimizer" begin
        # identity projections: x should stay unchanged
        P1!(p) = nothing
        P2!(p) = nothing

        start = ComponentVector(x=[-1.0, 2.0, 3.0])
        loss_fkt(p) = sum(abs2, p.x)

        result = InverseModeling.optimize_model(
            loss_fkt, start;
            optimize=dm_optimizer(1.0f0; P1! = P1!, P2! = P2!),
            iterations=5)

        @test isapprox(result.minimizer.x, [-1.0, 2.0, 3.0])
    end

end
