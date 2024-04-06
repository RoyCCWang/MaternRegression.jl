module MetaheuristicsExt

using MaternRegression
const GS = MaternRegression

import Metaheuristics
const EVO = Metaheuristics

include("helpers/evo.jl")


function MaternRegression._hp_optim(
    ::MaternRegression.UseMetaheuristics,
    trait::MaternRegression.GainTrait,
    ts,
    y::AbstractVector,
    lbs::Vector{T},
    ubs::Vector{T},
    f_calls_limit::Integer,
    p0s::Vector{Vector{T}},
    ) where T <: Real

    evo_config = ECAConfig(f_calls_limit, p0s) #p0)

    ML_buffer = GS.setup_ml(ts, y)
    ml_func = pp->GS.eval_ml!(trait, ML_buffer, pp, ts, y)
    costfunc = pp->(-ml_func(pp))


    result = runevofull(costfunc, lbs, ubs, evo_config)
    sol_vars = EVO.minimizer(result)
    sol_cost = EVO.minimum(result)

    return sol_vars, sol_cost
end


end