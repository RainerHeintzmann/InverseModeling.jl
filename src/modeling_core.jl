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

    function get_fit_results(res::ComponentVector)
        # @show typeof(res)
        # g(id) = get_val(getindex(fit_params, id), id, fit_params, fixed_params) 
        bare = res; # Optim.minimizer(res)
        all_keys = keys(bare)
        # The line below may apply pre-forward-models to the fit results. This is not necessary for the fixed params
        fwd = NamedTuple{all_keys}(collect(get_fwd_val(vals[id], id, bare, fixed_params) for id in keys(bare)))
        fwd = merge(fwd, fixed_params)
        return bare, fwd
    end
    """
        get_fit_results(res::Optim.MultivariateOptimizationResults)

    this version of `get_fit_results` is applied to the result of optim and first extracts
    the result data via `Optim.minimizer(res)` and then calles the internally defined get_fit_results function.

    """
    function get_fit_results(res::Optim.MultivariateOptimizationResults)
        get_fit_results(Optim.minimizer(res))
    end

    return fit_params, fixed_params, get_fit_results, stripped_params
end



"""
    create_forward(fwd, params, dtype=Float32)

creates a forward model given a model function `fwd` and a set of parameters `param`.
The properties such as `Positive` or `Normalize` of the modified `params` are baked into the model

Can be called with a `NamedTuple` or a `ComponentArray`. This will call the fwd function, 
which itself needs to access its one argument by function calls with the ids
fwd(g) which accesses the parameters via g(:myparamname)

# returns
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
    function forward(fit_params)
        # g is a function which receives an id and returns the corresponding paramter.
        # id corresponds to the variable names in a named tuple or ComponentArray
        function g(id) # an accessor function for the parameters
            #v = get_fwd_val(stripped_params[id], id, fit, fixed_params) 
            #@show stripped_params[id]
            #@show v
            # @show id
            #println("params[$(id)] is $(params[id])")
            #println("$(id) is $(fit[id]) is $v")
            # v
            # DO NOT PUT @time HERE! This will break Zygote.
            get_fwd_val(stripped_params[id], id, fit_params, fixed_params) 
        end
        return fwd(g) # calls fwd giving it the parameter-access function g
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
    optimize_model(loss_fkt, start_vals; iterations=100, optimizer=LBFGS(), kwargs...)

performs the optimization of the model parameters by calling Optim.optimize() and returns the result.
Other options such as `store_trace=true` can be provided and will be passed to `Optim.Options`.

#arguments
+ `loss_fkt`        : the loss function to optimize
+ `start_vals`      : the set of parameters over which the optimization is performed
+ `iterations`      : number of iterations to perform (default: 100). This is provided via the `Optim.Options` stucture.
+ `optimizer=LBFGS()`: the optimizer to use

#returns
the result as provided by Optim.optimize()
"""
function optimize_model(loss_fkt::Function, start_vals; iterations=100, optimizer=LBFGS(), kwargs...)
    optim_options = Optim.Options(;iterations=iterations, kwargs...)

    function fg!(F, G, vec)
        #val, mygrad = Zygote.withgradient(loss_fkt, vec)
        val_pb = Zygote.pullback(loss_fkt, vec);
        # println("in fg!: F:$(!isnothing(F)) G:$(!isnothing(G))")
        if !isnothing(G)
            # G .= mygrad
            # @show val_pb[2](one(eltype(vec)))[1]
            G .= val_pb[2](one(eltype(vec)))[1]
            # mutating calculations specific to g!
        end
        if !isnothing(F)
            # calculations specific to f
            return val_pb[1]
            # return val # val_pb[1]
        end
    end
    od = OnceDifferentiable(Optim.NLSolversBase.only_fg!(fg!), start_vals)
    # g!(G, vec) = G.=gradient(loss_fkt, vec)[1]

    optim_res = Optim.optimize(od, start_vals, optimizer, optim_options)
    # optim_res = Optim.optimize(loss_fkt, g!, start_vals, optimizer, optim_options)
    # optim_res = Optim.optimize(loss_fkt, start_vals, optimizer, optim_options) # ;  autodiff = :forward
    optim_res
end

"""
    optimize_model(start_val::Tuple, fwd_model::Function, loss_type=loss_gaussian; iterations=100, optimizer=LBFGS(), store_trace=true, kwargs...)

performs the optimization of the model parameters by calling Optim.optimize() and returns the result.

#arguments
+ `start_val`: the set of parameters over which the optimization is performed
+ `fwd_model`: the model which is optimized.
+ `meas`: measurement data to be compared to the forward projection
+ `loss_type`: the type of the loss function to use. Default: `loss_gaussian`
+ `iterations`      : number of iterations to perform (default: 100). This is provided via the `Optim.Options` stucture.
+ `optimizer=LBFGS()`: the optimizer to use
+ `return_fit`: if `true`, the forward projected result is returned as the second value. Default: `false`
+ `store_trace`. if `true`, the trace of the loss function is returned as the last value

# returns
the result is a Tuple of `res`, potentially a forward projected result, and the trace of the loss function value. `res` is a `ComponentArray` with all the results after applying the pre-forward part of the algorithm.
This includes the values marked as `Fixed()`.
If the argument `store_trace=false` is provided no trace will be returned.

#See also:
The other (low-level) version of `optimize_model` with the loss function as the first argument.
"""
function optimize_model(start_val::NamedTuple, fwd_model::Function, meas, loss_type=loss_gaussian; iterations=100, regularization=nothing, optimizer=LBFGS(), store_trace=true, return_fit=false, kwargs...)
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd_model, start_val);
    total_loss = loss(meas, forward, loss_type)
    if !isnothing(regularization)
        # @show typeof(dataloss)
        # dataloss = regularization
        reg_on_preforward(parameters) = regularization(get_fit_results(parameters)[2])
        total_loss = combine(total_loss, reg_on_preforward)
    end
    # return total_loss(start_vals) # regularization(get_fit_results(start_vals)[2])

    optim_res = InverseModeling.optimize_model(total_loss, start_vals; iterations=iterations, optimizer=optimizer, store_trace=store_trace, kwargs...);
    bare, res = get_fit_results(Optim.minimizer(optim_res))
    if store_trace
        if (return_fit)
            return res, forward(Optim.minimizer(optim_res)), [t.value for t in optim_res.trace][2:end]
        else
            return res, [t.value for t in optim_res.trace][2:end]
        end
    else
        if (return_fit)
            return res, forward(Optim.minimizer(optim_res))
        else
            return res
        end
    end
end

"""
    get_loss(start_val::NamedTuple, fwd_model::Function, meas, loss_type=loss_gaussian)

a helper function to easily retrieve the loss of a NamedTuple of starting values or fit results.
This function is not used inside the InverseModeling framework itself.
"""
function get_loss(start_val::NamedTuple, fwd_model::Function, meas, loss_type=loss_gaussian)
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd_model, start_val);
    loss(meas, forward, loss_type)(start_vals)
end

