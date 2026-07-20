export create_forward, sim_forward, loss, optimize_model

# split a NamedTuple with Fit and NonFit types into two 
# ComponentArrays
"""
    prepare_fit(vals, dtype=Float64)

this function is called before a fit is started.
#arguments
+ `vals`:   values to compare the data with
# `dtype`:  currently unused

#returns
a tuple of 
`fit_params`    : the (variable) fit parameters
`fixed_params`  : the fixed parameters with which the model is called but they are not optimized for
`get_fit_results`: a function that retrieves a tuple `(bare, params)` of the preforwardmodel fit result and a named tuple of version when called with the result of `optim()`
                `bare` is the raw result withoút the pre_forward_model being applied. This can be plugged into `foward`
                `param` is a named tuple of the result according to the model that the user specified.  
`stripped_params``: a version of params, with the fixed parameters stripped from all modifications
"""
function prepare_fit(vals, dtype=Float64)  # 
    fit_dict = Dict() #ComponentArray{dtype}()
    non_fit_dict = Dict() #ComponentArray{dtype}()
    stripped_params = Dict() # removed all modifications from Fixed values
    for (key, val) in zip(keys(vals), vals)        
        if is_fixed(val) # isa Fixed
            non_fit_dict[key] = get_val(val) # all other modifiers are ignored. 
            stripped_params[key] = Fixed(get_val(val))
        else
            fit_dict[key] = get_inv_val(val)
            stripped_params[key] = val
        end
    end
    fit_named_tuple = construct_named_tuple(fit_dict)
    non_fit_named_tuple = construct_named_tuple(non_fit_dict)
    stripped_params = construct_named_tuple(stripped_params)
    
    fit_params = ComponentArray(fit_named_tuple) # the optim routine cannot deal with tuples. ComponentArray{dtype} does NOT work for CUDA!
    fixed_params = ComponentArray(non_fit_named_tuple) # ComponentArray{dtype} does NOT work for CUDA!

    function get_fit_results(res)
        # g(id) = get_val(getindex(fit_params, id), id, fit_params, fixed_params) 
        bare = res.minimizer # Optim.minimizer(res)
        all_keys = keys(bare)
        # The line below may apply pre-forward-models to the fit results. This is not necessary for the fixed params
        fwd = NamedTuple{all_keys}(collect(get_fwd_val(vals[id], id, bare, fixed_params) for id in keys(bare)))
        fwd = merge(fwd, fixed_params)
        return bare, fwd
    end

    return fit_params, fixed_params, get_fit_results, stripped_params
end

"""
    create_forward(fwd, params)

creates a forward model given a model function `fwd` and a set of parameters `param`.
The properties such as `Positive` or `Normalize` of the modified `params` are baked into the model
#returns
a tuple of
`fit_params`    : a collection of the parameters to fit
`fixed_params`  : a collection of the fixed parameters exluded from the fitting, but provided to the model
`forward`       : the forward model
`backward`      : the adjoint model
`get_fit_results`: a function that retrieves the fit result for the result of optim
"""
function create_forward(fwd::Function, params, dtype=Float32) # 
    fit_params, fixed_params, get_fit_results, stripped_params = prepare_fit(params, dtype) #

    # can be called with a NamedTuple or a ComponentArray. This will call the fwd function, 
    # which itself needs to access its one argument by function calls with the ids
    # fwd(g) which accesses the parameters via g(:myparamname)
    function forward(fit_params; batch_dim=nothing, batch_idx=nothing, n_total=nothing)
        function g(id)
            val = get_fwd_val(stripped_params[id], id, fit_params, fixed_params)
            if batch_dim !== nothing && ndims(val) >= batch_dim && size(val, batch_dim) == n_total
                idxs = ntuple(d -> d == batch_dim ? batch_idx : Colon(), ndims(val))
                val = val[idxs...]
            end
            return val
        end
        return fwd(g)
    end

    function backward(vals)
        all_keys = keys(vals)
        NamedTuple{all_keys}(collect(get_fwd_val(vals, id, vals, fixed_params) for id in keys(vals)))
    end
    
    return fit_params, fixed_params, forward, backward, get_fit_results
end

"""
    sim_forward(fwd, params)

creates a model with a set of parameters and runs the forward method `fwd` to obtain the result.
This is useful for a simulation. No noise is applied, but can be applied afterwards.
"""
function sim_forward(fwd, params)
    vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd, params)
    return forward(vals);
end

"""
    loss(data, forward, loss_gaussian)

returns a loss function given a forward model `forward` with some measured data `data`. The noise_model is specified by `my_norm`.
The returned function needs to be called with parameters to be given to the forward model as arguments. 
"""
function loss(data, forward, my_norm = loss_gaussian, bg=eltype(data)(0))
    return (params) -> my_norm(data, forward(params), bg)
end

"""
    optimize_model(loss_fct, start_vals; iterations=100, optimizer=LBFGS(), kwargs...)

performs the optimization of the model parameters by calling Optim.optimize() and returns the result.
Other options such as `store_trace=true` can be provided and will be passed to `Optim.Options`.

#arguments
+ `loss_fct`        : the loss function to optimize
+ `start_vals`      : the set of parameters over which the optimization is performed
+ `iterations`      : number of iterations to perform (default: 100). This is provided via the `Optim.Options` stucture.
+ `optimizer=LBFGS()`: the optimizer to use

#returns
the result as provided by Optim.optimize()
"""
function optimize_model(loss_fct::Function, start_vals; iterations=100, optimizer=LBFGS(), optimize=Optim.optimize, kwargs...)
    optim_options = Optim.Options(;iterations=iterations, kwargs...)

    function fg!(F, G, vec)
        if !isnothing(G)
            val_pb = Zygote.pullback(loss_fct, vec)
            G .= val_pb[2](one(eltype(vec)))[1]
            if !isnothing(F)
                return val_pb[1]
            end
        else
            return loss_fct(vec)
        end
    end
    od = OnceDifferentiable(Optim.NLSolversBase.only_fg!(fg!), start_vals)
    if optimize === Optim.optimize
        optim_res = optimize(od, start_vals, optimizer, optim_options)
    else
        optim_res = optimize(od, start_vals, optimizer, optim_options, loss_fct)
    end
    optim_res
end

"""
    optimize_model(start_val::Tuple, fwd_model::Function, meas, loss_type=loss_gaussian; iterations=100, optimizer=LBFGS(), store_trace=true, kwargs...)

performs the optimization of the model parameters by calling Optim.optimize() and returns the result.

#arguments
+ `start_val`: the set of parameters over which the optimization is performed
+ `fwd_model`: the model which is optimized.
+ `meas`: measurement data to be compared to the forward projection
+ `loss_type`: the type of the loss function to use. Default: `loss_gaussian`
+ `iterations`      : number of iterations to perform (default: 100). This is provided via the `Optim.Options` stucture.
+ `optimizer=LBFGS()`: the optimizer to use

#returns
the result is a Tuple of `res` and the trace of the loss function value. `res` is a `ComponentArray` with all the results after applying the pre-forward part of the algorithm.
This includes the values marked as `Fixed()`.
if the argument ``store_trace=false` is provided no trace will be returned.

#See also:
The other (low-level) version of `optimize_model` with the loss function as the first argument.
"""
function optimize_model(start_val::NamedTuple, fwd_model::Function, meas, loss_type=loss_gaussian; iterations=100, optimizer=LBFGS(), store_trace=true, batch_size=nothing, batch_dim=ndims(meas), bg=eltype(meas)(0), learning_rate=1f0, shuffle=true, verbose=false, project! = nothing, kwargs...)
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd_model, start_val);

    if optimizer isa Optimisers.AbstractRule || (batch_size !== nothing && batch_size < size(meas, batch_dim))
        # ── SGD / Optimisers.jl path ──
        sgd_batch_size = batch_size === nothing ? size(meas, batch_dim) : min(batch_size, size(meas, batch_dim))
        opt_rule = optimizer isa Optimisers.AbstractRule ? optimizer : Optimisers.Descent(Float32(learning_rate))
        optim_res = sgd_optimize(forward, meas, loss_type, batch_dim, sgd_batch_size, start_vals;
                                 opt_rule=opt_rule, iterations=iterations,
                                 shuffle=shuffle, bg=bg, verbose=verbose, project! = project!)
        bare, res = get_fit_results(optim_res)
        if store_trace
            return res, [t.value for t in optim_res.trace]
        else
            return res
        end
    else
        # ── full-batch optimization ──
        if optimizer isa Function
            optim_res = InverseModeling.optimize_model(loss(meas, forward, loss_type, bg), start_vals; iterations=iterations, optimize=optimizer, store_trace=store_trace, kwargs...)
        else
            optim_res = InverseModeling.optimize_model(loss(meas, forward, loss_type, bg), start_vals; iterations=iterations, optimizer=optimizer, store_trace=store_trace, kwargs...)
        end
        bare, res = get_fit_results(optim_res)
        if store_trace
            return res, [t.value for t in optim_res.trace][2:end]
        else
            return res
        end
    end
end


function get_loss(start_val::NamedTuple, fwd_model::Function, meas, loss_type=loss_gaussian)
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd_model, start_val);
    loss(meas, forward, loss_type)(start_vals)
end

