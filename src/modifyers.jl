export Fixed, Positive, Normalize, ClampSum, BoundedSoftMax

abstract type Modificator end

# this datatyp specifies that this parameter is not part of the fit
struct Fixed{T} <: Modificator
    data::T
end

# this datatyp specifies that this parameter is kept positive during the fit
# this is achieved by introducing an auxiliary function whos abs2.() yields the parameter
struct Positive{T} <: Modificator
    data::T
end

"""  Normalize{T}
    this datatyp specifies that this parameter is normalized before the fit 
    this is achieved by deviding by the factor in the inverse path and multiplying by it in the forward model.
    Note that by chosing a fit parameter `p` for example using `Normalize(p, maximum(p))` or `Normalize(p, mean(p))`, the fit-variable will be unitless,
    but the result will automatically be cast back to the original scale. This helps the fit to converge.
"""
struct Normalize{T,F} <: Modificator
    data::T
    factor::F
end

function Base.size(d::Modificator, varargs...)
    return size(d.data, varargs...)
end

""" ClampSum{T}
    this contraint ensures that the value will never change its sum (optionally over predifined dimensions) during optimization. This is useful to avoid ambiguities during optimization.
    In practice the last value is simply computed by the given initial sum minus all other values.
"""
struct ClampSum{T, S, N}
    data::T
    mysum::S
    mysize::NTuple{N, Int}
end
""" ClampSum(dat::T, dims=ntuple((n)->n, ndims(dat))) where {T}

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

# if Fixed appears anywhere in a chain of modifyers, all of them are ignored
is_fixed(val) = false
is_fixed(val::Fixed) = true
is_fixed(val::Positive) = is_fixed(val.data)
is_fixed(val::Normalize) = is_fixed(val.data)
is_fixed(val::ClampSum) = is_fixed(val.data)

# access a NamedTuple with a symbol (eg  :a)
# this returns just the bare value
get_val(val) = val
get_val(val::Fixed) = get_val(val.data) # to strip off the properties
get_val(val::Positive) = get_val(val.data) # to strip off the properties
get_val(val::Normalize) = get_val(val.data) # to strip off the properties
get_val(val::ClampSum) = get_val(val.data) # to strip off the properties

# evaluated the pre-forward model modifyers
get_fwd_val(val) = val
get_fwd_val(val::Fixed) = get_fwd_val(val.data) # to strip off the properties
get_fwd_val(val::Positive) = abs2.(get_fwd_val(val.data)) 
get_fwd_val(val::Normalize) = get_fwd_val(val.data) .* val.factor
function get_fwd_val(val::ClampSum)
    myvals = get_fwd_val(val.data)
    # @show val.mysum
    # @show sum(myvals)
    reshape(vcat(myvals, val.mysum - sum(myvals)), val.mysize)
end

get_fwd_val(val::Fixed, id, fit, non_fit) = getindex(non_fit, id) # do not apply any modifiers here.
get_fwd_val(val::Positive, id, fit, non_fit) = abs2.(get_fwd_val(val.data, id, fit, non_fit))
get_fwd_val(val::Normalize, id, fit, non_fit) = get_fwd_val(val.data, id, fit, non_fit) .* val.factor
function get_fwd_val(val::ClampSum, id, fit, non_fit)
    myvals = get_fwd_val(val.data, id, fit, non_fit)
    # @show val.mysum
    # @show sum(myvals)
    reshape(vcat(myvals, val.mysum - sum(myvals)), val.mysize)
end
get_fwd_val(val, id, fit_params, non_fit) = get_fwd_val(fit_params[id]) # @view fit_params[id]
# begin 
#     tmp = @view fit_params[id]#  getindex(fit, id)
#     get_fwd_val(tmp)
# end

# inverse opterations for the pre-forward model parts
get_inv_val(val) = val
get_inv_val(val, fct) = fct(val)
get_inv_val(val::Fixed, fct=identity) = get_inv_val(val.data, fct)
get_inv_val(val::Positive, fct=identity) = get_inv_val(val.data, (x)->(sqrt.(fct.(max.(x, 0)))))
get_inv_val(val::Normalize, fct=identity) = get_inv_val(val.data, (x)->fct.(x)./ val.factor) 
get_inv_val(val::ClampSum, fct=identity) = get_inv_val(val.data, (x)->clamp_sum(fct.(x)))
function clamp_sum(val)
    myval = get_inv_val(val)
    return myval[1:end-1] # optimize the 1D- view with the last value missing
end

"""BoundedSoftMax{T}
    this datatyp specifies that parameter is bounded between `lower` and `upper` values. 
    Achieved with sigmoid function to map the optimizer variable to the range [lower, upper].
    The inverse mapping is done by using the logit function.
"""
struct BoundedSoftMax{T} <: Modificator
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


# Helper functions:
sigmoid(x) = 1 / (1 + exp(-x))
logit(x) = log(x / (1 - x))

is_fixed(val::BoundedSoftMax) = is_fixed(val.data)

get_val(val::BoundedSoftMax) = get_val(val.data)

get_fwd_val(val::BoundedSoftMax) = val.lower .+ (val.upper - val.lower) .* sigmoid.(get_fwd_val(val.data))
get_fwd_val(val::BoundedSoftMax, id, fit, non_fit) = val.lower .+ (val.upper - val.lower) .* sigmoid.(get_fwd_val(val.data, id, fit, non_fit))

get_inv_val(val::BoundedSoftMax, fct=identity) = get_inv_val(val.data, (x) -> logit.((fct.(x) .- val.lower) ./ (val.upper - val.lower)))
