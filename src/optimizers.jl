# collect other optimizers in addition to what Optim.jl provides
export steepest_decent_optimizer, sgd_optimize, rl_optimizer, dm_optimizer
export schedule_const, schedule_directional_hessian, schedule_table, schedule_exp
export batch_plan

# global_G = []  # for debugging

using Zygote
using LinearAlgebra


"""
    schedule_const()

returns a schedule function which always returns the alpha supplied to it, which means that the constant alpha is used.

# arguments of all schedule functions are:
function schedule(α, i, N, f, x, d, fx=nothing, g=nothing)
+ `i`: current iteration number (starting with 1)
+ `N`: total number of iterations
+ `f`: loss function to optimize
+ `x`: current position
+ `d`: current search direction
+ `fx`: f(x) pre-computed by the optimizer
+ `g`: gradient at x, pre-computed by the optimizer

"""
function schedule_const()
    function schedule(α, i, N, f, x, d, fx=nothing, g=nothing)
        return α
    end
    return schedule;
end

"""
     schedule_table(vectorfront=nothing, starting_ones=20, remaining_const=10, myconst=1f0)

returns a schedule function which picks the learning rates (overrelaxtion constants) from a fixed `vectorfront` cyclically.
Only the last `remaining_const` iterations the value of `myconst` is used.
# arguments
+ tablefront: a vector to cyclically pick the factors from. By default [1,2,1,16] is used.
+ startingones: for the `startingones` first iterations `myconst` is used
+ remainingones: for the training `remainingones` iterations `myconst` is used.

# arguments of all schedule functions are:
function schedule(α, i, N, f, x, d, fx=nothing, g=nothing)
+ `i`: current iteration number (starting with 1)
+ `N`: total number of iterations
+ `f`: loss function to optimize
+ `x`: current position
+ `d`: current search direction
+ `fx`: f(x) pre-computed by the optimizer
+ `g`: gradient at x, pre-computed by the optimizer

"""
function schedule_table(vectorfront=nothing, starting_ones=20, remaining_ones=10, myconst=1f0)
    vectorfront = (isnothing(vectorfront)) ? [1f0,2f0,1f0,16f0] : vectorfront
    function schedule(α, i, N, f, x, d, fx=nothing, g=nothing)
        i = i -starting_ones
        myalpha = (i > N - remaining_ones || i < 1) ? α * myconst : α * vectorfront[1+mod(i-1, length(vectorfront))]
        # @show myalpha
        return myalpha
    end
    return schedule;
end

"""
    schedule_exp(coeff=1.01f0)

returns a schedule function which is exponentially growing or shrinking over the iteration number
"""
function schedule_exp(coeff=1.01f0)
    myalpha = 1f0
    function schedule(α, i, N, f, x, d, fx=nothing, g=nothing)
        myalpha = (i <= 1) ? 1f0 : myalpha * coeff
        # @show myalpha
        return α * myalpha
    end
    return schedule;
end


# 1. The directional second derivative helper (Forward-over-Reverse)
"""
    alpha_directional_hessian(f, myreg=0, default_α=0.01f0)

returns a schedule that calculates the directional 2nd derivative of a loss function `f` along a given search direction `d`
by using automatic differentiation in mixed forward/reverse mode.
From this curvature it estimates the step length as norm(g)^2 / (gᵀ * H * g + myreg)
where `myreg` is an optional regularization term, which limits step lengths by adding a parabolic potential.

# arguments
+ `f`: loss function to optimize
+ `x`: current position
+ `d`: current search direction
"""
function schedule_directional_hessian(myreg=0, default_α=0.01f0; ε=1.0f-3)
    buf = nothing
    function schedule(α, i, N, f, x, minusd, fx, g)
        # Pre-allocate perturbation buffer once to avoid CuArray intermediates.
        if buf === nothing || length(buf) != length(x)
            buf = copy(x)
        end
        # In-place perturbation: write to raw CuArray storage (fully fused
        # broadcast, no intermediate arrays).  For flat arrays / NamedTuples
        # the else branch uses ComponentVector-level broadcasting.
        if x isa ComponentArrays.ComponentVector
            parent(buf) .= parent(x) .- ε .* parent(minusd)
        else
            @. buf = x - ε * minusd
        end
        # Directional derivative ∇f·d.
        # For steepest descent g = -d, so real(dot(g, d)) = -||d||².
        # When g is not provided, compute from d alone (avoids copy(G) allocation).
        dir_deriv = -real(LinearAlgebra.dot(g, minusd))
        d_H_d = 2 * (f(buf) - fx - ε * dir_deriv) / (ε^2)
        # ||d||² via BLAS dot (no intermediate reduction array).
        sqnorm_d = sum(abs2.(minusd))  # real(LinearAlgebra.dot(minusd,minusd)) # 
        α = (d_H_d <= 0) ? default_α : sqnorm_d / (d_H_d + myreg)
        return α
    end
    return schedule
end

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
+ `schedule`: a function which yields a learning rate in dependence of various input parameters. See `schedule_const()` for details.
"""
function steepest_decent_optimizer(α=1f0; P! = nothing, norm_vars=[], schedule=schedule_const(), num_batches=1, norm_fct=maximum, verbose=false)
    myP! = P!
    function steepest_decent_optimize(od!, start_vals, optimizer, optim_options, loss_fct=nothing)
        iterations = optim_options.iterations;
        vals = copy(start_vals);
        G = copy(start_vals);
        losses=[];
        lossfct = (loss_fct !== nothing) ? loss_fct : (vals) -> od!.f(vals)
        for n=1:iterations
            for b in 1:num_batches
                if loss_fct !== nothing
                    loss, grads = Zygote.withgradient(loss_fct, vals)
                    G .= grads[1]
                else
                    od!.df(G, vals)
                    loss = lossfct(vals)
                end
                if (verbose)
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
                    alpha_vars = schedule(alpha_var, n, iterations, lossfct, vals, G, loss, G)
                    G[myvar] .*=  alpha_var
                end

                if !isnothing(myP!)
                    myP!(vals, G)
                end
                if (isempty(norm_vars))
                    myalpha = schedule(α, n, iterations, lossfct, vals, G, loss, G)
                    vals .-= Float32(myalpha) .* G
                else
                    myalpha =  schedule(1f0, n, iterations, lossfct, vals, G, loss, nothing)
                    vals .-= Float32(myalpha) .* G
                end
            end # batches
        end # iterations
        return (minimizer=vals, trace=losses)
    end
    return steepest_decent_optimize
end

"""
    batch_plan(plan_to_batch, data_size, example_arr, batch_dim=ndims(batch_arg), batch_size=size(batch_arg, batchdim))    

a helper function that allows to mimic plan behaviour for batched data processing. It potentially allocates two plans instead of one.

# arguments
+ plan_to_batch: The function used for plan creation which has accept a single argument of size batch_arg
+ data_size: the full datasize (before batching)
+ example_arr: an example array to infer the datatype of the plan from and allocate buffers 
+ batch_dim=ndims(batch_arg)
+ batch_size=size(batch_arg, batchdim)
"""
function batch_plan(plan_to_batch, data_size, example_arr, batch_dim=ndims(batch_arg), batch_size=size(batch_arg, batchdim))    
    batched_data_sz = ntuple((d)-> (d==batch_dim) ? batch_size : data_size[d], length(data_size))
    dummy = similar(example_arr, batched_data_sz)
    b_plan = plan_to_batch(dummy)
    dummy = nothing
    remaining_size = mod(data_size[batch_dim], batch_size)
    b_rem_plan = (remaining_size == 0) ? b_plan : begin
            rem_data_sz = ntuple((d)-> (d==batch_dim) ? remaining_size : data_size[d], length(data_size))
            dummy2 = similar(example_arr, rem_data_sz)
            plan_to_batch(dummy2)
        end
    function myplan(batch_data)
        if (size(batch_data, batch_dim) == batch_size)
            return b_plan(batch_data)
        else
            return b_rem_plan(batch_data)
        end
    end
    return myplan
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
                       shuffle=true, bg=zero(eltype(data)), verbose=false,
                       project! = nothing)
    n_total = size(data, batch_dim)
    fit_params = copy(start_vals)
    opt_state = Optimisers.setup(opt_rule, fit_params)
    losses = []
    # batch_losses = Float32[ceil(n_total÷batch_size)]
    # batch_losses .= NaN

    for epoch in 1:iterations
        order = shuffle ? randperm(n_total) : 1:n_total
        total_loss = 0f0
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
            if project! !== nothing
                project!(fit_params)
            end
            total_loss += val

            if verbose
                println("epoch $epoch, batch $(batch_start:batch_end), loss $val")
            end
        end
        push!(losses, (value=total_loss,))
    end

    return (minimizer=fit_params, trace=losses)
end


"""
    rl_optimizer(α=1.0; project! = nothing, norm_factor=nothing, verbose=false)

Projected gradient version of the Richardson-Lucy algorithm.
Each iteration computes the standard gradient, applies an RL-style
preconditioner (scaling each gradient component by the corresponding
parameter value divided by `norm_factor`), then applies the projection.

# Arguments
- `α`: step size (default 1.0)
- `project!`: optional in-place projection `project!(params)` called after each update
- `norm_factor`: NamedTuple or ComponentArray of per-parameter normalization factors
                 (corresponds to Hᵀ1 in the classical RL derivation).
                 When `nothing`, only the current parameter value scaling is applied.
- `verbose`: print loss per iteration

# Usage
```julia
optimize_model(start_val, fwd, data, loss;
    optimize=rl_optimizer(1.0; project! = p -> p .= max.(p, 0)))
```
"""
function rl_optimizer(α=1.0; project! = nothing, norm_factor=nothing, verbose=false)
    function rl_optimize(od!, start_vals, optimizer, optim_options, loss_fct=nothing)
        iterations = optim_options.iterations
        vals = copy(start_vals)
        G = copy(start_vals)
        losses = []
        for n = 1:iterations
            if loss_fct !== nothing
                loss, grads = Zygote.withgradient(loss_fct, vals)
                G .= grads[1]
            else
                od!.df(G, vals)
                loss = od!.f(vals)
            end
            if verbose
                println("iteration: $n, loss: $(loss)")
                push!(losses, (value=loss,))
            end
            # RL-style preconditioner: scale gradient by current estimate
            for key in keys(vals)
                gk = G[key]
                vk = vals[key]
                αf = Float32(α)
                if gk isa AbstractArray
                    if norm_factor !== nothing
                        G[key] .= αf .* vk .* gk ./ norm_factor[key]
                    else
                        G[key] .= αf .* vk .* gk
                    end
                else
                    if norm_factor !== nothing
                        G[key] = αf * vk * gk / norm_factor[key]
                    else
                        G[key] = αf * vk * gk
                    end
                end
            end
            vals .-= G
            if project! !== nothing
                project!(vals)
            end
        end
        return (minimizer=vals, trace=losses)
    end
    return rl_optimize
end


"""
    dm_optimizer(β=1.0; P1!, P2!, verbose=false)

Difference Map optimizer.
Each iteration computes:
    x' = x + β * (P₁(2·P₂(x) - x) - P₂(x))
where P₁ and P₂ are user-supplied in-place projection functions
that modify their argument.  The loss is evaluated for diagnostics
but is not used by the DM update itself.

# Arguments
- `β`: DM relaxation parameter (default 1.0)
- `P1!`: in-place projection `P1!(params)` — the first constraint set
- `P2!`: in-place projection `P2!(params)` — the second constraint set
- `verbose`: print loss per iteration

# Usage
```julia
P1!(p) = (p .= max.(p, 0))
P2!(p) = (p .= clamp.(p, 0, 1))
optimize_model(start_val, fwd, data, loss;
    optimize=dm_optimizer(0.8; P1!, P2!))
```
"""
function dm_optimizer(β=1.0; P1!, P2!, verbose=false)
    function dm_optimize(od!, start_vals, optimizer, optim_options, loss_fct=nothing)
        iterations = optim_options.iterations
        vals = copy(start_vals)
        losses = []
        for n = 1:iterations
            if verbose
                loss = od!.f(vals)
                println("iteration: $n, loss: $(loss)")
                push!(losses, (value=loss,))
            end
            # DM update: x' = x + β*(P₁(2·P₂(x) - x) - P₂(x))
            p2_vals = deepcopy(vals)
            P2!(p2_vals)
            p1_in = deepcopy(vals)
            for key in keys(p1_in)
                p1_in[key] .= 2 .* p2_vals[key] .- vals[key]
            end
            P1!(p1_in)
            for key in keys(vals)
                vals[key] .= vals[key] .+ Float32(β) .* (p1_in[key] .- p2_vals[key])
            end
        end
        return (minimizer=vals, trace=losses)
    end
    return dm_optimize
end


## ToDo:  RL-Algorithms, Implement Preconditioners, Holmes&Liu Scheme, Overrelaxition tables
# directional derivative?
# alternating projections
# regularizers for sum contraint, positivity, equality
# warning for non-normalized data

"""
    gradient_descent_so!(f, x_init; maxiter=100, tol=1e-5)
# --- Test it out ---
# A classic banana-like valley function
loss(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2 # solution is [1.0, 1.0]

x0 = [-1.2, 1.0]
x_opt = gradient_descent_so!(loss, x0)
println("Found minimum at: x_opt, loss=loss(x_opt))")
"""
# 2. Custom Gradient Descent with Exact Second-Order Step Size
# function gradient_descent_so!(f, x_init; maxiter=100, tol=1e-5, verbose=true, myreg=0)
#     x = copy(x_init)
    
#     for i in 1:maxiter
#         # Compute gradient (first derivative)
#         g = Zygote.gradient(f, x)[1]
        
#         # Check convergence
#         if norm(g) < tol
#             println("Converged in $i iterations.")
#             return x
#         end
        
#         # Search direction for Gradient Descent is steepest descent
#         d = -g 
        
#         # Compute a regularized step using the second derivative along direction d: dᵀ * H * d
#         α = alpha_directional_hessian(f, x, d)
        
#         # Take the step
#         x .+= α .* d
#         if (verbose)
#             println("iteration: $(i), alpha=$(α), grad: $(d), step: $(x), loss: $(loss(x))")
#         end
#     end
    
#     println("Reached max iterations.")
#     return x
# end

