# collect other optimizers in addition to what Optim.jl provides
export steepest_decent_optimizer, sgd_optimize

global_G = []


"""
    gradient_optimize(od, start_vals, optimizer, optim_options)

an optimizer implementing steepest descent without linesearch

# arguments
+ `α` : overrelaxation constant (always multiplied)
+ `P!`: precoditioner, modifies the gradient, receives `vals` and `gradients` as argument
+ `norm_vars`: applies a normalization to each `gradients` part using the `norm_fct` applied to the corresponding `vals` arguments.  
            if variables are given ([var1:, var2:, ...]) you can also give separate `α` values as (var1= 0.8f0, var2= 0.5f0)
            A gradient part is devided by the product of the normed result of all other current values in analogy to the ePIE algorithm
+ `verbose`: if true, the loss values per iteration will be printed
"""
function steepest_decent_optimizer(α=1f0; P! = nothing, norm_vars=[], num_batches=1, norm_fct=maximum, verbose=false)
    myP! = P!
    function steepest_decent_optimize(od!, start_vals, optimizer, optim_options)
        iterations = optim_options.iterations;
        vals = copy(start_vals);
        G = copy(start_vals);
        losses=[];
        for n=1:iterations
            # perm = randperm()
            for b in 1:num_batches
                od!.df(G, vals)
                if (verbose)
                    loss = od!.f(vals)
                    println("iteration: $n, loss: $(loss)")
                    push!(losses, (value=loss,))
                end
                for myvar in norm_vars
                    myalpha = α
                    if (isa(α, NamedTuple)) && (haskey(α, myvar))
                        myalpha = α[myvar]
                    end
                    alpha_var = myalpha
                    for other_var in norm_vars
                        if (other_var !== myvar)
                            alpha_var /= norm_fct(abs2.(vals[other_var]))
                        end
                    end
                    # if (verbose)
                    #     println("alpha[$(myvar)]=$(alpha_var)")
                    # end
                    G[myvar] .*=  alpha_var
                end

                if !isnothing(myP!)
                    myP!(vals, G)
                end
                if (isempty(norm_vars))
                    # push!(global_G, Float32(α) .*  G[:screen])
                    # println("pushed empty G2")
                    vals .-= Float32(α) .* G
                else
                    # push!(global_G, G[:screen])
                    # println("pushed G2")
                    vals .-= G # alpha was already accounted for
                end
            end # batches
        end # iterations
        return (minimizer=vals, trace=losses)
    end
    return steepest_decent_optimize
end


"""
    sgd_optimize(forward_fn, data, loss_type, batch_dim, batch_size, start_vals;
                 opt_rule=Descent(0.01), iterations=100, shuffle=true, bg=0, verbose=false)

Stochastic gradient descent with mini-batches. The forward model is called
with `batch_dim`, `batch_idx`, and `n_total` keyword arguments so it only
computes the current batch's output. The loss is then computed between the
batched forward output and the corresponding data slice.

Batch-varying parameters must have `ndims(param) >= batch_dim` and
`size(param, batch_dim) == size(data, batch_dim)`. They are automatically
sliced along `batch_dim` inside the forward model.
"""
function sgd_optimize(forward_fn, data, loss_type, batch_dim, batch_size, start_vals;
                       opt_rule=Optimisers.Descent(0.01), iterations=100,
                       shuffle=true, bg=zero(eltype(data)), verbose=false)
    n_total = size(data, batch_dim)
    fit_params = copy(start_vals)
    opt_state = Optimisers.setup(opt_rule, fit_params)
    losses = Float32[]

    for epoch in 1:iterations
        order = shuffle ? randperm(n_total) : 1:n_total
        for batch_start in 1:batch_size:n_total
            batch_end = min(batch_start + batch_size - 1, n_total)
            batch_idx = order[batch_start:batch_end]
            data_batch = selectdim(data, batch_dim, batch_idx)

            val, pb = Zygote.pullback(fit_params) do p
                fwd_batch = forward_fn(p; batch_dim, batch_idx, n_total)
                loss_type(data_batch, fwd_batch, bg)
            end
            grad = pb(one(eltype(val)))[1]

            opt_state, fit_params = Optimisers.update(opt_state, fit_params, grad)
            push!(losses, val)

            if verbose
                println("epoch $epoch, batch $(batch_start:batch_end), loss $val")
            end
        end
    end

    return (minimizer=fit_params, trace=[(value=l,) for l in losses])
end


## ToDo:  RL-Algorithms, Implement Preconditioners, Holmes&Liu Scheme, Overrelaxition tables
# directional derivative?
# alternating projections
# regularizers for sum contraint, positivity, equality
# warning for non-normalized data
