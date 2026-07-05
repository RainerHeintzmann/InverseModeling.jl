using FiniteDifferences
@testset "Noise Model Gaussian"  begin

    @test loss_gaussian(1,0,0) == 1
    @test loss_gaussian(0,0,0) == 0
    @test loss_gaussian(0,1,0) == 1
    @test loss_gaussian(0,1,1) == 1

    @test loss_poisson(1,0,0) == Inf
    @test loss_poisson(0,0,0) == 0
    @test loss_poisson(0,1,0) == 1
    @test loss_poisson(-1,0,0) == -Inf
    # @test loss_poisson(0,-1,0) == 0

    @test loss_poisson_pos(1,0,0) == Inf
    @test loss_poisson_pos(0,0,0) == 0
    @test loss_poisson_pos(-1,0,0) == 0
    @test loss_poisson_pos(0,-1,0) == 0
    @test loss_poisson_pos(0,1,0) == 1

    @test loss_anscombe(1,0,0) == 1.0
    @test loss_anscombe(0,0,0) == 0
    @test loss_anscombe(0,1,0) == 1

    @test loss_anscombe_pos(1,0,0) == 1.0
    @test loss_anscombe_pos(0,0,0) == 0
    @test loss_anscombe_pos(0,1,0) == 1
    @test loss_anscombe_pos(-1,0,0) == 0
    @test loss_anscombe_pos(0,-1,0) == 0
    @test loss_anscombe_pos(0,1,0) == 1

    val = 10.0
    tst_val = (tst1=val, )
    tst_start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->x(:tst1), tst_val);

    valn = -10.0
    tst_valn = (tst1=valn, )
    tst_start_valsn, fixed_valsn, forwardn, backwardn, get_fit_resultsn = create_forward((x)->x(:tst1), tst_valn);

    measured = 10.0
    measuredn = -10.0
    myloss = loss(measured, forward)

    @test myloss(tst_start_vals) == 0
    myloss = loss(measured, forward, loss_gaussian)
    @test myloss(tst_start_vals) == 0
    myloss = loss(measured, forward, loss_poisson)
    @test myloss(tst_start_vals) ≈ -val*log(val) + val 
    bg = 3
    myloss = loss(measured, forward, loss_poisson, bg)
    @test myloss(tst_start_vals) ≈ -(val+bg)*log(val+bg) + val+bg 

    myloss = loss(measured, forwardn, loss_poisson)
    @test_throws DomainError myloss(tst_start_valsn)
    # @test myloss(tst_start_valsn)

    myloss = loss(measured, forward, loss_poisson_pos)
    @test myloss(tst_start_vals) ≈ -val*log(val) + val 

    myloss = loss(measuredn, forwardn, loss_poisson_pos)
    @test myloss(tst_start_valsn) == 0

    myloss = loss(measured, forward, loss_anscombe)
    @test myloss(tst_start_vals) ≈ 0.0 


    start_vals = tst_start_vals
    optim_res = InverseModeling.optimize_model(myloss, start_vals, iterations=100);

    # @show optim_res = Optim.optimize(loss(measured, forward), start_vals, LBFGS())
    bare, res = get_fit_results(optim_res)
    fit = forward(bare);
    @test isapprox(fit, measured, atol=1e-4)

    start_vals[:tst1] = start_vals[:tst1] + 1
    optim_res = InverseModeling.optimize_model(myloss, start_vals, iterations=100);
    # @show optim_res = Optim.optimize(loss(measured, forward), start_vals, LBFGS())
    bare, res = get_fit_results(optim_res)
    fit = forward(bare);
    @test isapprox(fit, measured, atol=5e-3)

end
