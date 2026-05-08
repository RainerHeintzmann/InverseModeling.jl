
function dim_GR(d, weight, arr)
        idx_range = ntuple(i -> i == d ? (1:size(arr, i)-1) : (1:size(arr, i)-1), ndims(arr))
        idx_range_plus = ntuple(i -> i == d ? (2:size(arr, i)) : (1:size(arr, i)-1), ndims(arr))
        # abs^2 of forward differences along dimension d, divided by the abs of local brightness.        
        return (2*weight) .* (abs2.(view(arr, idx_range_plus...) .- view(arr, idx_range...)))./(abs.(view(arr, idx_range_plus...)) .+ abs.(view(arr, idx_range...)))
end

function dim_grad_abs2(d, weight, arr)
        idx_range = ntuple(i -> i == d ? (1:size(arr, i)-1) : (1:size(arr, i)-1), ndims(arr))
        idx_range_plus = ntuple(i -> i == d ? (2:size(arr, i)) : (1:size(arr, i)-1), ndims(arr))
        # abs^2 of Forward differences along dimension d 
        # Accumulate squared gradients
        return weight .* abs2.(view(arr, idx_range_plus...) .- view(arr, idx_range...))
end

function dim_laplace_abs2(d, weight, arr)
        idx_range = ntuple(i -> i == d ? (2:size(arr, i)-1) : (1:size(arr, i)-1), ndims(arr))
        idx_range_plus = ntuple(i -> i == d ? (3:size(arr, i)) : (1:size(arr, i)-1), ndims(arr))
        idx_range_minus = ntuple(i -> i == d ? (1:size(arr, i)-2) : (1:size(arr, i)-1), ndims(arr))
        # Forward differences along dimension d        
        # Accumulate squared gradients
        return weight .* abs2.(2 .*view(arr, idx_range...) .- view(arr, idx_range_plus...) .- view(arr, idx_range_minus...))
end

"""
    TV_views(arr::AbstractArray; dims=nothing, weights=nothing, ϵ=1f-8)

Compute Total Variation regularization using pure Julia operations (Zygote-compatible).

This version avoids in-place modifications and uses array slicing with views, making it
fully compatible with Zygote for automatic differentiation. Works on CPU and GPU
(with some performance penalty on GPU compared to `TV_KA`).

The computation builds arrays of gradients along each specified dimension, then combines
them using broadcasting (all operations are immutable and differentiable).

# Arguments
- `arr`: Input array of any dimensionality
- `dims`: Tuple of dimension indices for gradient computation. If `nothing`, uses all dimensions.
  Example: `dims=(1,2)` computes gradients in dimensions 1 & 2 only.
- `weights`: Weight for each dimension. If `nothing`, defaults to ones.
  Must be a tuple or vector of length `ndims(arr)`.
- `ϵ`: Smoothing parameter for TV norm (default: 1f-8).

# Returns
Scalar Float value of the total variation (differentiable)

# Examples
```julia
using Zygote

arr = rand(Float32, 10, 20)
tv, grad = Zygote.withgradient(TV_views, arr)

# With custom dimensions and weights
tv = TV_views(arr, dims=(1,), weights=(1.0, 0.5))
```
"""
function TV_views(arr::AbstractArray{T}; dims::Union{Nothing,NTuple}=nothing, 
                      weights::Union{Nothing,Vector,NTuple}=nothing, ϵ=1f-8) where T
    ndim = ndims(arr)
    # Default: use all dimensions
    dims = isnothing(dims) ? ntuple(i -> i, ndim) : dims
    # Default: equal weights for all dimensions
    weights = isnothing(weights) ? ntuple(i -> one(T), ndim) : weights 
    
    # @fastmath is slower for the line below:
    return sum(sqrt.(.+(ϵ , [dim_grad_abs2(dims[d], weights[d], arr) for d=1:length(dims)]...))) 

    # dim_map = map((d)->dim_abs2(dims[d], weights[d], arr), 1:length(dims))
    # tv_out = sum(sqrt.(.+(ϵ , dim_map...))) 

    # tv_out = sum(sqrt.(.+(ϵ , dim_abs2(dims[1], weights[1], arr), dim_abs2(dims[2], weights[2], arr), dim_abs2(dims[3], weights[3], arr))))

    # tv_out = sum(sqrt.(ϵ .+ sum(dim_abs2.(dims, weights, Ref(arr)))))
    # return tv_out
end

function Tikhonov_views(arr::AbstractArray{T}; dims::Union{Nothing,NTuple}=nothing, 
                      weights::Union{Nothing,Vector,NTuple}=nothing, ϵ=1f-8) where T
    ndim = ndims(arr); dims = isnothing(dims) ? ntuple(i -> i, ndim) : dims
    weights = isnothing(weights) ? ntuple(i -> one(T), ndim) : weights 
    
    return sum([dim_abs2(dims[d], weights[d], arr) for d=1:length(dims)]...)
end

function Laplace_views(arr::AbstractArray{T}; dims::Union{Nothing,NTuple}=nothing, 
                      weights::Union{Nothing,Vector,NTuple}=nothing, ϵ=1f-8) where T
    ndim = ndims(arr); dims = isnothing(dims) ? ntuple(i -> i, ndim) : dims
    weights = isnothing(weights) ? ntuple(i -> one(T), ndim) : weights 
    
    return sum([dim_abs2(dims[d], weights[d], arr) for d=1:length(dims)]...)
end

function GR_views(arr::AbstractArray{T}; dims::Union{Nothing,NTuple}=nothing, 
                      weights::Union{Nothing,Vector,NTuple}=nothing, ϵ=1f-8) where T
    ndim = ndims(arr); dims = isnothing(dims) ? ntuple(i -> i, ndim) : dims
    weights = isnothing(weights) ? ntuple(i -> one(T), ndim) : weights 
    
    return sum([dim_abs2(dims[d], weights[d], arr) for d=1:length(dims)]...)
end
