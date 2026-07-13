module InverseModeling

using ComponentArrays
using IndexFunArrays
using Optim, Zygote
using SeparableFunctions # for gaussian
using NDTools # for select_region_view
using ChainRulesCore # for the rrule definitions

include("utilities.jl")
include("noise_models.jl")
include("modeling_core.jl")
include("model_gauss.jl")
include("modifyers.jl")
include("optimizers.jl")

end # module
