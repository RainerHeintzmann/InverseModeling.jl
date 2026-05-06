export TV_KA, TV_KA_zygote

"""
    TV_KA(arr::AbstractArray; dims=nothing, weights=nothing, ϵ=1f-8)

Compute Total Variation regularization using KernelAbstractions.jl for GPU acceleration.

**Now fully Zygote-compatible!** Uses KernelAbstractions kernels for fast GPU computation
while supporting automatic differentiation via ChainRulesCore.

This function computes the TV norm as sqrt(ϵ + sum(weights[d] * grad[d]^2)) where gradients 
are computed via forward differences along the specified dimensions.

# Arguments
- `arr`: Input array of any dimensionality
- `dims`: Tuple of dimension indices for gradient computation. If `nothing`, uses all dimensions.
- `weights`: Weight for each dimension. If `nothing`, defaults to ones.
- `ϵ`: Smoothing parameter for TV norm (default: 1f-8).

# Returns
Scalar Float value of the total variation

# Examples
```julia
using Zygote

arr = rand(Float32, 10, 20)
tv_val = TV_KA(arr)

# With gradients
tv_val, back = Zygote.pullback(TV_KA, arr)
grad_arr = back(1.0f0)[1]
```
"""
function TV_KA(arr::AbstractArray{T}; dims::Union{Nothing,NTuple}=nothing, 
               weights::Union{Nothing,Vector,NTuple}=nothing, ϵ=1f-8) where T
    ndim = ndims(arr)
    
    # Default: use all dimensions
    if isnothing(dims)
        dims = ntuple(i -> i, ndim)
    end
    
    # Default: equal weights for all dimensions
    if isnothing(weights)
        weights = ntuple(i -> 1f0, ndim)
    else
        # Convert to tuple if vector provided
        weights = Tuple(weights)
    end
    
    # Get backend (auto-detects CUDA, AMD, CPU, etc.)
    backend = KernelAbstractions.get_backend(arr)
    
    # Determine output size: reduce by 1 in each dimension in dims
    out_size = ntuple(i -> (i in dims) ? size(arr, i) - 1 : size(arr, i), ndim)
    tv_out = KernelAbstractions.zeros(backend, T, out_size)
    
    # Launch kernel
    kernel = tv_kernel_generic!(backend)
    kernel(tv_out, arr, dims, weights, ϵ, ndrange=out_size)
    
    return sum(tv_out)
end

# Generic TV kernel for any number of dimensions
# Computes TV norm along specified dimensions
@kernel function tv_kernel_generic!(tv_out, arr, dims, weights, ϵ)
    idx = @index(Global, NTuple)
    
    # Check bounds: ensure we can access arr[idx...] and arr[idx with increment in each dim in dims]
    valid = true
    for d in dims
        if idx[d] >= size(arr, d)
            valid = false
            break
        end
    end
    
    if valid
        # Accumulate squared gradients for each dimension in dims
        grad_sq_sum = zero(eltype(arr))
        
        for d in dims
            # Create index with increment in dimension d
            idx_plus = ntuple(i -> i == d ? idx[i] + 1 : idx[i], length(idx))
            
            # Compute forward difference (gradient) along dimension d
            grad = weights[d] * (arr[idx_plus...] - arr[idx...])
            grad_sq_sum += grad^2
        end
        
        # Compute TV norm: sqrt(ϵ + sum of squared gradients)
        tv_out[idx...] = sqrt(ϵ + grad_sq_sum)
    end
end

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
    
    # Compute gradients along each dimension using views (immutable operations)
    # Collect all gradient components
    # grad_sq_sum = T(0) # nothing
    
    # compute the gradient over the requested dimensions, sum over the rest later
    # for d in dims
    #     # Create views: one shifted by 1 in dimension d
    #     idx_range = ntuple(i -> i == d ? (1:size(arr, i)-1) : (1:size(arr, i)-1), ndim)
    #     idx_range_plus = ntuple(i -> i == d ? (2:size(arr, i)) : (1:size(arr, i)-1), ndim)
        
    #     # Forward differences along dimension d        
    #     # Accumulate squared gradients
    #     grad_sq_sum = grad_sq_sum .+ weights[d] .* abs2.(view(arr, idx_range_plus...) .- view(arr, idx_range...))
    # end
    # grad_sq_sum = 
        
    # Compute TV norm: sqrt(ϵ + sum of squared gradients)
    # dim_grad = ntuple((d) -> dim_abs2(dims[d], weights[d], arr), length(dims))
    # tv_out = sum(sqrt.(.+(ϵ , dim_grad...)))
    # dim_generator = (dim_abs2(dims[d], weights[d], arr) for d=1:length(dims))
    # tv_out = sum(sqrt.(.+(ϵ , dim_generator...))) 

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
