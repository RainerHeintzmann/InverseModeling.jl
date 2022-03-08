export create_forward, Fixed, Positive, loss, optimize_model

# this datatyp specifies that this parameter is not part of the fit
struct Fixed{T}
    data::T
end

# this datatyp specifies that this parameter is kept positive during the fit
# this is achieved by introducing an auxiliary function whos abs2.() yields the parameter
struct Positive{T}
    data::T
end

"""
    @my_get t s

Access a NamedTuple/Struct `t` with the field `s`.
```
julia> t = (a=10, b=12.0)
(a = 10, b = 12.0)

julia> @my_get t a
10
```
"""
macro my_get(t, x)
    :($t.$x)
end


# access a NamedTuple with a symbol (eg  :a)
get_val(val::Fixed, id, fit, non_fit) = getindex(non_fit, id)
get_val(val::Positive, id, fit, non_fit) = abs2.(getindex(fit, id))
get_val(val, id, fit, non_fit) = getindex(fit, id)

# inverse opterations for the pre-forward model parts
get_inv_val(val::Fixed, id, fit, non_fit) = getindex(non_fit, id)
get_inv_val(val::Positive, id, fit, non_fit) = sqrt.(getindex(fit, id))
get_inv_val(val, id, fit, non_fit) = getindex(fit, id)

 # construct a named tuple from a dict
construct_named_tuple(d) = NamedTuple{Tuple(keys(d))}(values(d))

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
"""
function prepare_fit(vals, dtype=Float64)
    fit_dict = Dict() #ComponentArray{dtype}()
    non_fit_dict = Dict() #ComponentArray{dtype}()
    for (key, val) in zip(keys(vals), vals)        
        if val isa Fixed
            non_fit_dict[key] = val.data
        elseif val isa Positive
            fit_dict[key] = sqrt.(val.data)
        else
            fit_dict[key] = val
        end
    end
    fit_named_tuple = construct_named_tuple(fit_dict)
    non_fit_named_tuple = construct_named_tuple(non_fit_dict)
    
    fit_params = ComponentArray(fit_named_tuple)
    fixed_params = ComponentArray(non_fit_named_tuple)

    function get_fit_results(res)
        # g(id) = get_val(getindex(fit_params, id), id, fit_params, fixed_params) 
        bare = Optim.minimizer(res)
        fwd = NamedTuple{keys(bare)}(collect(get_val(bare, id, bare, fixed_params) for id in keys(bare)))
        return bare, fwd
    end

    return fit_params, fixed_params, get_fit_results
end

"""
    create_forward(fwd, params, dtype=Float64)

creates a forward model given a model function `fwd` and a set of parameters `param`.
#returns
a tuple of
`fit_params`    : a collection of the parameters to fit
`fixed_params`  : a collection of the fixed parameters exluded from the fitting, but provided to the model
`forward`       : the forward model
`backward`      : the adjoint model
`get_fit_results`: a function that retrieves the fit result for the result of optim
"""
function create_forward(fwd, params, dtype=Float64)
    fit_params, fixed_params, get_fit_results = prepare_fit(params, dtype)

    function forward(fit)
        g(id) = get_val(getindex(params, id), id, fit, fixed_params) 
        return fwd(g)
    end
    function backward(vals)
        NamedTuple{keys(vals)}(collect(get_val(vals, id, vals, fixed_params) for id in keys(vals)))
    end
    
    return fit_params, fixed_params, forward, backward, get_fit_results
end

"""
    loss(data, forward, my_norm=gaussian_norm)

returns a loss function given a forward model `forward` with some measured data `data`. The noise_model is specified by `my_norm`.
The returned function needs to be called with parameters to be given to the forward model as arguments. 
"""
function loss(data, forward, my_norm=gaussian_norm)
    return (params) -> my_norm(data, forward(params))
end

"""
    optimize_model(loss_fkt, start_vals; iterations=100, optimizer=LBFGS())

performs the optimization of the model parameters by calling Optim.optimize() and returns the result.

#arguments
+ `loss_fkt`        : the loss function to optimize
+ `start_vals`      : the set of parameters over which the optimization is performed
+ `iterations`      : number of iterations to perform (default: 100). This is provided via the `Optim.Options` stucture.
+ `optimizer=LBFGS()`: the optimizer to use

#returns
the result as provided by Optim.optimize()
"""
function optimize_model(loss_fkt, start_vals; iterations=100, optimizer=LBFGS())
    optim_options = Optim.Options(iterations=iterations)
    # g!(G,vec) = G.=gradient(loss_fkt,vec)[1]
    # @time optim_res = Optim.optimize(loss_fkt, g!, start_vals, optimizer, optim_options)
    optim_res = Optim.optimize(loss_fkt, start_vals, optimizer, optim_options) # ;  autodiff = :forward
    optim_res
end

"""
    into_mask!(vec::AbstractArray{T, 1}, mymask::AbstractArray{Bool, N}, tmp_data=zeros(eltype(vec),size(mymask))) where {T, N}

fills a vector into a mask `mymask`. This is very useful for optimizing only some pixels in an image. E.g. the inner part of a Fourier-space aperture.
#arguments
+ `vec` : the vector to fill into the mask
+ `mymask`  : a binary mask to use
+ `tmp_data` : the mask is filled into only this part of the dataset. For efficiency reasons you should provide this argument, even if this is not strictly necessary as zeros are filled in.
#returns
tmp_mask
"""
function into_mask!(vec::AbstractArray{T, 1}, mymask::AbstractArray{Bool, N}, tmp_data=zeros(eltype(vec),size(mymask))) where {T, N}
    tmp_data[mymask] = vec
    return tmp_data
end

using ChainRulesCore
# define custom adjoint for, since mutating arrays is not supported
function ChainRulesCore.rrule(::typeof(into_mask!), vec::AbstractArray{T, 1}, amask::AbstractArray{Bool, M},
    tmp_data=zeros(eltype(vec), size(amask))) where {T, M}
    Y = into_mask!(vec, amask, tmp_data)
    function into_mask_pullback(barx)
        if eltype(vec) <: Complex
            return NoTangent(), barx[amask], NoTangent(), NoTangent()
        else
            return NoTangent(), real(barx[amask]), NoTangent(), NoTangent()
        end
    end
    return Y, into_mask_pullback
end
