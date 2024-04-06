module MaternRegression

using LinearAlgebra
using Statistics
using StaticArrays

function getlog2(::Type{T})::T where T <: AbstractFloat
    #return log1p(one(T))
    return convert(T, log(2))
end

function getlog4(::Type{T})::T where T <: AbstractFloat
    #return log1p(3*one(T))
    return convert(T, log(4))
end

include("types.jl")
include("filtering.jl")
include("smoothing.jl")

# analytical discretizations of linear SDEs.
include("eq_discrete/basic.jl")

include("frontend.jl")

include("hp_inference.jl")

export TimeVarType,
create_Matern3HalfsKernel,
create_sdegp,
query_sdegp,
query_sdegp!,
setup_ml,
eval_ml,
eval_ml!,

GainTrait,
InferGain,
UnityGain,
hp_optim,
UseMetaheuristics,
parse_ml_result,

hp_inference,
UseDynamicHMC,
simulate_sdegps,
compute_mean_var

end # module MaternRegression
