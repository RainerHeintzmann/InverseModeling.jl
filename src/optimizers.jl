# collect other optimizers in addition to what Optim.jl provides
export steepest_decent_optimizer

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


## ToDo:  RL-Algorithms, Implement Preconditioners, Holmes&Liu Scheme, Overrelaxition tables
# directional derivative?
# alternating projections
# regularizers for sum contraint, positivity, equality
# warning for non-normalized data
# support for batching
