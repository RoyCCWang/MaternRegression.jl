module DynamicHMCExt

using MaternRegression

import FiniteDiff
const SFD = FiniteDiff

using SimpleUnPack

import TransformVariables
const TV = TransformVariables

import TransformedLogDensities
const TD = TransformedLogDensities

import LogDensityProblems

import DynamicHMC
const HMC = DynamicHMC

import Random

# # MCMC problem. Likelihood and prior.
# Gamma priors for both λ and σ². Shared gamma hyperparameters, α, β.
# DynamicHMCExt = ["DynamicHMC", "FiniteDiff", "LogDensityProblems", "SimpleUnPack", "TransformedLogDensities", "TransformVariables"]

# The trait-dispatch pattern won't work here since the HMC and Logdensity libraries requires we have a callable object. We will just make two identical objects with different names for the two traits.
# see https://docs.julialang.org/en/v1/manual/methods/#Trait-based-dispatch

abstract type MLProblem end

struct UnityML{
    T <: AbstractFloat,
    BT,
    RT <: Union{AbstractRange, AbstractVector},
    } <: MLProblem

    buffer::BT
    ts::RT
    y::Vector{T}

    # prior hyperparameters. Shared for λ and σ².
    α::T
    β::T
end

function (problem::UnityML{T})(p) where T <: Real
    @unpack λ, σ² = p
    return eval_ml_problem(problem, λ, σ², one(T))
end

struct InferML{
    T <: AbstractFloat,
    BT,
    RT <: Union{AbstractRange, AbstractVector},
    } <: MLProblem

    buffer::BT
    ts::RT
    y::Vector{T}

    # prior hyperparameters. Shared for λ and σ².
    α::T
    β::T
end

function (problem::InferML)(p)
    @unpack λ, σ², b = p
    return eval_ml_problem(problem, λ, σ², b)
end

# # common routines for the type MLProblem
function eval_ml_problem(problem::MLProblem, λ, σ², b)

    @unpack buffer, ts, y, α, β = problem

    ln_likelihood = MaternRegression.eval_ml!(
        buffer, λ, σ², b, ts, y,
    )

    ln_prior =  eval_ln_prior(λ, α, β)
    return ln_likelihood + ln_prior
end

function eval_ln_prior(λ, α, β)

    # # all the same priors. Inverse gamma.
    ln_prior_λ = (-α-1)*log(λ) -β/λ
    ln_prior_σ² = (-α-1)*log(λ) -β/λ
    ln_prior_b = (-α-1)*log(λ) -β/λ

    return ln_prior_σ² + ln_prior_λ + ln_prior_b
end

# # Manual specification of the likelihood and prior.

# ## Manual transformation.

# function LogDensityProblems.capabilities(::Type{<:MLProblem})
#     LogDensityProblems.LogDensityOrder{0}()
# end

LogDensityProblems.dimension(::UnityML) = 2

function LogDensityProblems.logdensity(problem::UnityML, x)
    ln_λ, ln_σ² = x
    σ² = exp(ln_σ²)
    λ = exp(ln_λ)

    return problem((λ = λ, σ² = σ²)) + ln_σ² + ln_λ
end

LogDensityProblems.dimension(::InferML) = 3

function LogDensityProblems.logdensity(problem::InferML, x)
    ln_λ, ln_σ², ln_b = x
    σ² = exp(ln_σ²)
    λ = exp(ln_λ)
    b = exp(ln_b)

    return problem((λ = λ, σ² = σ², b = b)) + ln_σ² + ln_λ + ln_b
end

# ## Manual specify gradient via finite differences

function LogDensityProblems.capabilities(::Type{<:MLProblem})
    LogDensityProblems.LogDensityOrder{1}() # can do gradient
end

function LogDensityProblems.logdensity_and_gradient(problem::MLProblem, x)
    logdens = LogDensityProblems.logdensity(problem, x)

    f = xx->LogDensityProblems.logdensity(problem, xx)
    grad = SFD.finite_difference_gradient(f, x)
    
    return logdens, grad
end

###

# X_mat[1,:] is λ, X_mat[2,:] is σ².
function MaternRegression._hp_inference(
    ::MaternRegression.UseDynamicHMC,
    trait::MaternRegression.UnityGain,
    N_draws::Integer,
    α::Real, β::Real, ts, y::AbstractVector,
    )
    
    ML_buffer = MaternRegression.setup_ml(ts, y)
    problem = UnityML(ML_buffer, ts, y, α, β)

    mcmc_results = HMC.mcmc_with_warmup(
        Random.default_rng(), problem, N_draws,
    )

    X_mat = exp.(mcmc_results.posterior_matrix)
    λ_samples = vec(X_mat[1,:])
    σ²_samples = vec(X_mat[2,:])

    return λ_samples, σ²_samples, mcmc_results
end

# X_mat[1,:] is λ, X_mat[2,:] is σ². X_mat[3,:] is b.
function MaternRegression._hp_inference(
    ::MaternRegression.UseDynamicHMC,
    trait::MaternRegression.InferGain,
    N_draws::Integer,
    α::Real, β::Real, ts, y::AbstractVector,
    )
    
    ML_buffer = MaternRegression.setup_ml(ts, y)
    problem = InferML(ML_buffer, ts, y, α, β)

    mcmc_results = HMC.mcmc_with_warmup(
        Random.default_rng(), problem, N_draws,
    )
    X_mat = exp.(mcmc_results.posterior_matrix)
    λ_samples = vec(X_mat[1,:])
    σ²_samples = vec(X_mat[2,:])
    b_samples = vec(X_mat[3,:])

    return λ_samples, σ²_samples, b_samples, mcmc_results
end

end # end module.