module InverseModeling

using ComponentArrays
using IndexFunArrays
using Optim, Optimisers, Zygote
using SeparableFunctions # for gaussian
using NDTools # for select_region_view
using ChainRulesCore # for the rrule definitions
using Random # for randperm in optimizers in batches

include("utilities.jl")
include("noise_models.jl")
include("modeling_core.jl")
include("model_gauss.jl")
include("modifyers.jl")
include("optimizers.jl")

# Optional: Dual-aware FFT/IFFT/BFFT for CuArrays (for forward-over-reverse HVP)
# try
#     using CUDA: CuArray
#     using AbstractFFTs
#    include("cufft_duals.jl")
# catch
#     @debug "CUDA not available — skipping Dual-aware cuFFT methods"
# end

end # module
