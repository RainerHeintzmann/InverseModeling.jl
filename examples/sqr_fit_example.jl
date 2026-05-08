using InverseModeling
start_val = (σ=Positive([8.0, 25.0]),
        μ=Fixed([1.0, 3.0]) )

times = -10:10
fit_fkt(params)::Vector{Float64} = (times.*params(:σ)[1]).^2 + (times.*params(:σ)[2]).^3 .+ times .* params(:μ)[1]

start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fit_fkt, start_val)
meas = forward(start_vals)
meas .+= 5e5*randn(size(meas))
# distort the ground truth parameters a little
start_vals.σ += [0.5,0.2]


@show optim_res = optimize_model(loss(meas, forward), start_vals)

bare, fit_res = get_fit_results(optim_res)
fit = forward(bare)

using Plots # , Optim
plot(times, forward(start_vals), label="start values", xlabel="x", ylabel="f(x)")
scatter!(times, meas, label="measurement")
plot!(times, fit, label="fit")
