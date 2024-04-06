
abstract type ExtensionPkgs end

"""
    struct UseDynamicHMC <: ExtensionPkgs
        pkg_list::Vector{Module}
    end

Container for verifying whether the weak dependencies for `DynamicHMCExt.jl` is loaded in the user's working scope.

The dependencies are:
- `FiniteDiff`
- `SimpleUnPack`
- `TransformVariables`
- `TransformedLogDensities`
- `LogDensityProblems`
- `DynamicHMC`
"""
struct UseDynamicHMC <: ExtensionPkgs
    pkg_list::Vector{Module}
end


"""
    hp_inference(
        alg_trait::UseDynamicHMC,
        gain_trait::GainTrait,
        N_draws::Integer,
        α::Real,
        β::Real,
        ts::TimeVarType,
        y::Vector,
    )

Checks if the weak dependencies are loaded in the user's working scope, then checks if the corresponding package extensions are loaded. If so, call the appropriate hyperparameter inference routine from the package extension.

- `alg_trait`: see `UseDynamicHMC`.
- `gain_trait` is a trait variable that specifies the order of the hyperparameters in `p`. See *trait-based dispatch* in the Julia documentation. If `typeof(trait) <: InferGain`, then the hyperparameters in `p` are ordered `[λ; σ²; b]`. If `typeof(trait) <: UnityGain`, then the hyperparameters in `p` are ordered `[λ; σ²]`, and `b` is set to `1`. `p` is used internally by `hy_optim`.
- `α` and `β` are the shared hyperparameters of the inverse gamma prior for the hyperparameters.
- `ts` is the set of training inputs.
- `y` is the set of training outputs.

See the tutorial for the return type and subsequent usage.
"""
function hp_inference(
    alg_trait::UseDynamicHMC,
    gain_trait::GainTrait,
    N_draws::Integer,
    α::Real,
    β::Real,
    ts::TimeVarType,
    y::Vector,
    )

    pkg_list = alg_trait.pkg_list
    ext_hmc = Base.get_extension(@__MODULE__, :DynamicHMCExt)

    # avoid type instability; write one for each extension.
    proceed_flag = !isnothing(ext_hmc) && (ext_hmc.DynamicHMC in pkg_list) && (ext_hmc.SimpleUnPack in pkg_list) && (ext_hmc.FiniteDiff in pkg_list) && (ext_hmc.TransformVariables in pkg_list) && (ext_hmc.TransformedLogDensities in pkg_list) && (ext_hmc.LogDensityProblems in pkg_list)

    if !proceed_flag
        loaderr_ext_mcmc_pkg()
    end
    
    return _hp_inference(alg_trait, gain_trait, N_draws, α, β, ts, y)
end

function _hp_inference(::Nothing, args...)
    return loaderr_ext_mcmc_pkg()
end

function loaderr_ext_mcmc_pkg()
    error("For infering the hyperparameters via Hamiltonian Monte Carlo, the following Julia packages should be loaded: FiniteDiff, SimpleUnPack, TransformVariables, TransformedLogDensities, LogDensityProblems, DynamicHMC, Random")  
end

# #### point-estimate, i.e., optimization.

"""
    struct UseMetaheuristics <: ExtensionPkgs
        pkg::Module
    end

Container for verifying whether the weak dependencies for `MetaheuristicsExt.jl` is loaded in the user's working scope.

The only dependency required to be loaded in the user's working scope is `Metaheuristics`.
"""
struct UseMetaheuristics <: ExtensionPkgs
    pkg::Module
end

# dim 1 is λ, dim 2 is σ².
function generate_grid(::UnityGain, lbs, ubs, N)
    return vec(
        collect.(
            collect(Iterators.product(
            LinRange(lbs[begin], ubs[begin], N),
            LinRange(lbs[end], ubs[end], N),
        ))
    ))
end

# dim 1 is λ, dim 2 is σ², dim 3 is b.
function generate_grid(::InferGain, lbs, ubs, N)
    return vec(
        collect.(
            collect(Iterators.product(
            LinRange(lbs[begin], ubs[begin], N),
            LinRange(lbs[begin+1], ubs[begin+1], N),
            LinRange(lbs[end], ubs[end], N),
        ))
    ))
end

"""
    hp_optim(
        alg_trait::UseMetaheuristics,
        model_trait::GainTrait,
        ts::TimeVarType,
        y::Vector,
        lbs::Vector{T},
        ubs::Vector{T};
        f_calls_limit = 10_000,
        p0s::Vector{Vector{T}} = generate_grid(model_trait, lbs, ubs, 10),
    )

Checks if the weak dependencies are loaded in the user's working scope, then checks if the corresponding package extensions are loaded. If so, call the appropriate hyperparameter optimization routine from the package extension.

- `alg_trait`: see `UseMetaheuristics`.
- `model_trait` is a trait variable that specifies the order of the hyperparameters in `p`. See *trait-based dispatch* in the Julia documentation. If `typeof(trait) <: InferGain`, then the hyperparameters in `p` are ordered `[λ; σ²; b]`. If `typeof(trait) <: UnityGain`, then the hyperparameters in `p` are ordered `[λ; σ²]`, and `b` is set to `1`. `p` is used internally by `hy_optim`.
- `ts` is the set of training inputs.
- `y` is the set of training outputs.
- `lbs` and `ubs` are lower and upper bounds for the hyperparameter vector, `p`. They follow the ordering as specified by `model_trait`.
- `f_calls_limit` is a *soft* upperbound on the number of marginal likelihood evaluations used during optimization such that the evolutionary algorithm in `Metaheuristics.jl` tries to not exceed.
- `p0s` is a nested `Vector` array that contains hyperparameter states that you want to force the optimization algorithm to evaluate. These states can be interpreted as initial guesses to the solution. The default creates a uniform grid of `10x10x10` (if `typeof(model_trait) <: InferGain`) or `10x10` (if `typeof(model_trait) <: UnityGain`) from the lower and upper bounds specified.

See the tutorial for the return type and subsequent usage.
"""
function hp_optim(
    alg_trait::UseMetaheuristics,
    model_trait::GainTrait,
    ts::TimeVarType,
    y::Vector,
    lbs::Vector{T},
    ubs::Vector{T};
    f_calls_limit = 10_000,
    p0s::Vector{Vector{T}} = generate_grid(model_trait, lbs, ubs, 10),
    ) where T <: Real

    alg = alg_trait.pkg
    ext_evo = Base.get_extension(@__MODULE__, :MetaheuristicsExt)

    # avoid type instability; write one for each extension.
    proceed_flag = !isnothing(ext_evo) && alg == ext_evo.Metaheuristics
    if !proceed_flag
        loaderr_ext_optim_pkg()
    end
    
    return _hp_optim(
        alg_trait, model_trait, ts, y, lbs, ubs, f_calls_limit, p0s,
    )
end

function _hp_optim(::Nothing, args...)
    return loaderr_ext_optim_pkg()
end

function loaderr_ext_optim_pkg()
    error("Please load a supported numerical optimization package in your active Julia environment in order to use hp_optim(). Supported optimization packages: Metaheuristics.jl.")
end