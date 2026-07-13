export loss_gaussian, loss_poisson, loss_anscombe, loss_poisson_pos, loss_anscombe_pos, loss_sqrt_anscombe

logmul_safe(dat, val) = ifelse.(val == zero(eltype(val)) && dat == zero(eltype(dat)), zero(eltype(val)), dat*log(val))
clip_pos(val) = max(zero(eltype(val)), val)

"""
    loss_gaussian(data, fwd, bg=0) = sum(abs2.(data.-fwd))

Calculate the Gaussian loss assuming constant variance between `data` and `fwd`, with an optional background value `bg`. Note that `bg` is here only for compatibililty reasons and has no effect.

+ Arguments
    - `data`: The observed data.
    - `fwd`: The forward model prediction.
    - `bg`: is ignored
"""
loss_gaussian(data, fwd, bg=0) = sum(abs2.(data.-fwd))

"""
    loss_poisson(data, fwd, bg=eltype(data)(0)) = sum((fwd.+bg) .- logmul_safe.(data.+bg,fwd.+bg))

Calculate the Poisson loss between `data` and `fwd`, with an optional background value `bg` to avoid numerical issues.
The background value is added to both `data` and `fwd` before the calculation.
+ Arguments
    - `data`: The observed data.
    - `fwd`: The forward model prediction.
    - `bg`: A background value to avoid numerical issues, defaulting to zero.
"""
loss_poisson(data, fwd, bg=eltype(data)(0)) = sum((fwd.+bg) .- logmul_safe.(data.+bg,fwd.+bg))

"""
    loss_poisson_pos(data, fwd, bg=eltype(data)(0.01)) = sum(clip_pos.(fwd) .- logmul_safe.(clip_pos.(data), clip_pos.(fwd)))   

Calculate the Poisson loss between `data` and `fwd`, ensuring that both are non-negative by clipping them to zero before the calculation.
The background value `bg` is added to both `data` and `fwd` before the calculation to avoid numerical issues.
+ Arguments
    - `data`: The observed data.
    - `fwd`: The forward model prediction.
    - `bg`: A background value to avoid numerical issues, defaulting to 0.01.
"""
loss_poisson_pos(data, fwd, bg=eltype(data)(0.01)) = sum(clip_pos.(fwd).+bg .- logmul_safe.(clip_pos.(data).+bg, clip_pos.(fwd).+bg))

"""
    loss_anscombe(data, fwd, bg=eltype(data)(0)) = sum(abs2.(sqrt.(data.+bg) .- sqrt.(fwd.+bg)))

Calculate the Anscombe loss between `data` and `fwd`, with an optional background value `bg` to avoid numerical issues.
+ Arguments
    - `data`: The observed data.
    - `fwd`: The forward model prediction.
    - `bg`: A background value to avoid numerical issues, defaulting to zero.
"""
loss_anscombe(data, fwd, bg=eltype(data)(0)) = sum(abs2.(sqrt.(data.+bg) .- sqrt.(fwd.+bg)))

"""
    loss_sqrt_anscombe(data, fwd, bg=eltype(data)(0)) = sum(abs2.(sqrt.(data.+bg) .- sqrt.(fwd.+bg)))

Calculate the Squareroot of the Anscombe loss between `data` and `fwd`, with an optional background value `bg` to avoid numerical issues.
The squareroot was investigated by R. Koch indicating potential speed advances
+ Arguments
    - `data`: The observed data.
    - `fwd`: The forward model prediction.
    - `bg`: A background value to avoid numerical issues, defaulting to zero.
"""
loss_sqrt_anscombe(data, fwd, bg=eltype(data)(0)) = sqrt(sum(abs2.(sqrt.(data.+bg) .- sqrt.(fwd.+bg))))

"""
    loss_anscombe_pos(data, fwd, bg=eltype(data)(0.01)) = sum(abs2.(sqrt.(clip_pos.(data).+bg) .- sqrt.(clip_pos.(fwd).+bg)))

Calculate the Anscombe loss between `data` and `fwd`, ensuring that both are non-negative by clipping them to zero before the calculation.
The background value `bg` is added to both `data` and `fwd` before the calculation to avoid numerical issues.
+ Arguments
    - `data`: The observed data.
    - `fwd`: The forward model prediction.
    - `bg`: A background value to avoid numerical issues, defaulting to 0.01.
"""
loss_anscombe_pos(data, fwd, bg=eltype(data)(0.01)) = sum(abs2.(sqrt.(clip_pos.(data).+bg) .- sqrt.(clip_pos.(fwd).+bg)))

