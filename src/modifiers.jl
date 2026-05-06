
export Fixed, Normalize, ClampSum, Positive, PositiveH
export BoundedSoftMax, BoundedSoftMaxH
export MaskOnly

"""
    abstract type Modifier end
 This type is the base-type of various pre-forward models which typically place constraints
 onto domain of parameters to optimize.
"""
abstract type Modifier end

function Base.size(d::Modifier, varargs...)
    return size(d.data, varargs...)
end

# Functions which all Modifiers should implement:
# These are the bare fallback versions:
# by default all Modifiers are not fixed, i.e. they are part of the fit:
is_fixed(val) = false
# this returns just the bare value
get_val(val) = val
get_fwd_val(val) = val
get_fwd_val(val, id, fit_params, non_fit) = get_fwd_val(fit_params[id]) # @view fit_params[id]
# inverse opterations for the pre-forward model parts
get_inv_val(val) = val
get_inv_val(val, fct) = fct(val)

# this datatyp specifies that this parameter is not part of the fit
"""
    struct Fixed{T} <: Modifier

The `Fixed` property applied to a parameter removes the parameter from the optimization framework
without modifying the forward function which can still access this parameter. It is just not optimized. 
"""
struct Fixed{T} <: Modifier
    data::T
end
# if Fixed appears anywhere in a chain of modifyers, all of them are ignored
is_fixed(val::Fixed) = true
get_val(val::Fixed) = get_val(val.data) # to strip off the properties
get_fwd_val(val::Fixed) = get_fwd_val(val.data) # to strip off the properties
get_fwd_val(val::Fixed, id, fit, non_fit) = getindex(non_fit, id) # do not apply any modifiers here.
get_inv_val(val::Fixed, fct=identity) = get_inv_val(val.data, fct)


"""  
    struct Normalize{T,F} <: Modifier
    this type specifies that this parameter is normalized before the fit 
    this is achieved by deviding by the factor in the inverse path and multiplying by it in the forward model.
    Note that by chosing a fit parameter `p` for example using `Normalize(p, maximum(p))` or `Normalize(p, mean(p))`, the fit-variable will be unitless,
    but the result will automatically be cast back to the original scale. This helps the fit to converge.
"""
struct Normalize{T,F} <: Modifier
    data::T
    factor::F
end

is_fixed(val::Normalize) = is_fixed(val.data)
get_val(val::Normalize) = get_val(val.data) # to strip off the properties
get_fwd_val(val::Normalize) = get_fwd_val(val.data) .* val.factor
get_fwd_val(val::Normalize, id, fit, non_fit) = get_fwd_val(val.data, id, fit, non_fit) .* val.factor
get_inv_val(val::Normalize, fct=identity) = get_inv_val(val.data, (x)->fct.(x)./ val.factor) 

## helper function for ClampSum
function clamp_sum(val)
    myval = get_inv_val(val)
    return myval[1:end-1] # optimize the 1D- view with the last value missing
end

""" 
    struct ClampSum{T, S, N} <: Modifier
    this contraint ensures that the value will never change its sum (optionally over predifined dimensions) during optimization. This is useful to avoid ambiguities during optimization.
    In practice the last value is simply computed by the given initial sum minus all other values.
"""
struct ClampSum{T, S, N}
    data::T
    mysum::S
    mysize::NTuple{N, Int}
end

""" 
struct ClampSum(dat::T, dims=ntuple((n)->n, ndims(dat))) where {T} <: Modifier

convenience constructor. 
# Arguments
+ `data`: data for which the sum should be clamped
+ `dims`: defines the dimensions over which to sums should be clamped. The default is all dimensions (clamps the total sum)
"""
function ClampSum(dat::T) where {T}
    val = get_val(dat)
    mysum = sum(val)
    ClampSum{T,typeof(mysum),length(size(val))}(dat, mysum, size(val))
end

is_fixed(val::ClampSum) = is_fixed(val.data)
get_val(val::ClampSum) = get_val(val.data) # to strip off the properties

# evaluated the pre-forward model modifyers
function get_fwd_val(val::ClampSum)
    myvals = get_fwd_val(val.data)
    # @show val.mysum
    # @show sum(myvals)
    reshape(vcat(myvals, val.mysum - sum(myvals)), val.mysize)
end

function get_fwd_val(val::ClampSum, id, fit, non_fit)
    myvals = get_fwd_val(val.data, id, fit, non_fit)
    reshape(vcat(myvals, val.mysum - sum(myvals)), val.mysize)
end
# begin 
#     tmp = @view fit_params[id]#  getindex(fit, id)
#     get_fwd_val(tmp)
# end
get_inv_val(val::ClampSum, fct=identity) = get_inv_val(val.data, (x)->clamp_sum(fct.(x)))

""" 
struct Positive{T} <: Modifier
    this datatype specifies that this parameter is positive during the fit
    this is achieved by introducing an auxiliary function whos abs2.() yields the parameter
    Note that the inverse operation will return the square root of the value, picking only the positive branch as the starting value.
"""
struct Positive{T} <: Modifier
    data::T
end
is_fixed(val::Positive) = is_fixed(val.data)
get_val(val::Positive) = get_val(val.data) # to strip off the properties
get_fwd_val(val::Positive) = abs2.(get_fwd_val(val.data)) 
get_fwd_val(val::Positive, id, fit, non_fit) = abs2.(get_fwd_val(val.data, id, fit, non_fit))
get_inv_val(val::Positive, fct=identity) = get_inv_val(val.data, (x)->(sqrt.(fct.(max.(x, 0)))))

# Helper functions for BoundedSoftMax:
sigmoid(x) = 1 / (1 + exp(-x))
logit(x) = log(x / (1 - x))

"""
    struct BoundedSoftMax{T} <: Modifier
    this datatype specifies that parameter is bounded between `lower` and `upper` values. 
    Achieved with sigmoid function to map the optimizer variable to the range [lower, upper].
    The inverse mapping is done by using the logit function.
"""
struct BoundedSoftMax{T} <: Modifier
    data::T
    lower::eltype(T)
    upper::eltype(T)

    function BoundedSoftMax(dat::T, lower=zero(eltype(T)), upper=one(eltype(T))) where {T}
        if lower >= upper
            throw(ArgumentError("lower bound must be smaller than upper bound"))
        end
        if any(dat .< lower) 
            throw(ArgumentError("all data must be greater than or equal to lower bound"))
        end
        if any(dat .> upper) 
            throw(ArgumentError("all data must be less than or equal to upper bound"))
        end
        return new{T}(dat, lower, upper)
    end

end

is_fixed(val::BoundedSoftMax) = is_fixed(val.data)
get_val(val::BoundedSoftMax) = get_val(val.data)
get_fwd_val(val::BoundedSoftMax) = val.lower .+ (val.upper - val.lower) .* sigmoid.(get_fwd_val(val.data))
get_fwd_val(val::BoundedSoftMax, id, fit, non_fit) = val.lower .+ (val.upper - val.lower) .* sigmoid.(get_fwd_val(val.data, id, fit, non_fit))

get_inv_val(val::BoundedSoftMax, fct=identity) = get_inv_val(val.data, (x) -> logit.((fct.(x) .- val.lower) ./ (val.upper - val.lower)))


#### Hyperbolic Versions of Positive and BoundedSoftMax:
piecewise_hyperbolic(x) = ifelse(x < 0, 1/(1-x), 1+x)
piecewise_hyperbolic_inv(x) = ifelse(x < 1, 1 - 1/x, x - 1)

"""
    struct PositiveH{T} <: Modifier
    this datatype specifies that this parameter is positive during the fit
    this is achieved by introducing an auxiliary function whos abs2.() yields the parameter
    Note that the inverse operation will return the square root of the value, picking only the positive branch as the starting value.
"""
struct PositiveH{T} <: Modifier
    data::T
end
is_fixed(val::PositiveH) = is_fixed(val.data)
get_val(val::PositiveH) = get_val(val.data) # to strip off the properties
get_fwd_val(val::PositiveH) = piecewise_hyperbolic.(get_fwd_val(val.data)) 
get_fwd_val(val::PositiveH, id, fit, non_fit) = piecewise_hyperbolic.(get_fwd_val(val.data, id, fit, non_fit))
get_inv_val(val::PositiveH, fct=identity) = get_inv_val(val.data, (x)->(piecewise_hyperbolic_inv.(fct.(max.(x, 0)))))

# Helper functions for BoundedSoftMax:
sigmoid_hyperbolic(x) = 1 / (1 + piecewise_hyperbolic(-x))
sigmoid_hyperbolic_inv(x) = piecewise_hyperbolic_inv(x / (1 - x))

"""
struct BoundedSoftMaxH{T} <: Modifier
    this datatype specifies that parameter is bounded between `lower` and `upper` values. 
    Achieved with sigmoid function to map the optimizer variable to the range [lower, upper].
    The inverse mapping is done by using the logit function.
"""
struct BoundedSoftMaxH{T} <: Modifier
    data::T
    lower::eltype(T)
    upper::eltype(T)

    function BoundedSoftMaxH(dat::T, lower=zero(eltype(T)), upper=one(eltype(T))) where {T}
        if lower >= upper
            throw(ArgumentError("lower bound must be smaller than upper bound"))
        end
        if any(dat .< lower) 
            throw(ArgumentError("all data must be greater than or equal to lower bound"))
        end
        if any(dat .> upper) 
            throw(ArgumentError("all data must be less than or equal to upper bound"))
        end
        return new{T}(dat, lower, upper)
    end
end

is_fixed(val::BoundedSoftMaxH) = is_fixed(val.data)
get_val(val::BoundedSoftMaxH) = get_val(val.data)
get_fwd_val(val::BoundedSoftMaxH) = val.lower .+ (val.upper - val.lower) .* sigmoid_hyperbolic.(get_fwd_val(val.data))
get_fwd_val(val::BoundedSoftMaxH, id, fit, non_fit) = val.lower .+ (val.upper - val.lower) .* sigmoid_hyperbolic.(get_fwd_val(val.data, id, fit, non_fit))

get_inv_val(val::BoundedSoftMaxH, fct=identity) = get_inv_val(val.data, (x) -> sigmoid_hyperbolic_inv.((fct.(x) .- val.lower) ./ (val.upper - val.lower)))


"""
    struct MaskOnly{T} <: Modifier
    this datatype specifies that parameter is bounded between `lower` and `upper` values. 
    Achieved with sigmoid function to map the optimizer variable to the range [lower, upper].
    The inverse mapping is done by using the logit function.
"""
struct MaskOnly{V, T, M} <: Modifier
    data::V
    tmp_data::T
    mask::M

    function MaskOnly(dat::T, mask::M) where {BT,D,T<:AbstractArray{BT, D}, M<:AbstractArray{Bool,D}}
        tmp_data = copy(dat)
        return new{typeof(dat[mask]), T,M}(dat[mask], tmp_data, mask)
    end
end

is_fixed(val::MaskOnly) = is_fixed(val.data)
get_val(val::MaskOnly) = get_val(val.data)
get_fwd_val(val::MaskOnly) = into_mask!(get_fwd_val(val.data), val.mask, val.tmp_data)
get_fwd_val(val::MaskOnly, id, fit, non_fit) = into_mask!(get_fwd_val(val.data, id, fit, non_fit), val.mask, val.tmp_data)

get_inv_val(val::MaskOnly, fct=identity) = get_inv_val(val.tmp_data, (x) -> fct.(x)[val.mask])
