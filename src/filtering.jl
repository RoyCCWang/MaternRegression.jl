# Kalman filtering

# p( x_t_k | y_{1:k} ), for each k in [N].
function runkalmanfilter!(
    FD::FilterDistributions, # mutates, output.
    system::DiscreteDiscreteSystem,
    prior::MVNParameters,
    )
    
    posteriors, predictives = FD.posteriors, FD.predictives

    # initialization of the filtration.
    posteriors[begin] = computemarginalposterior(
        DiscardInnovation(),
        prior,
        system.observations[begin],
        system.measurements[begin],
    )
    predictives[begin] = prior 

    # subsequent entries.
    for k in Iterators.drop(eachindex(predictives), 1)
        predictives[k], posteriors[k] = computekalmaniter(
            DiscardInnovation(),
            posteriors[k-1],
            system.observations[k],
            system.latent_steps[k-1],
            system.measurements[k],
        )
    end

    return nothing
end

# one iteration of Algorithm 10.18, Sarkka book 2019.
# get the mean and covariance of p( x_t_k | y_{1:k} ) via one Kalman step.
# subscript 0 is previous time instances, 1 is current time instance.
# output: predictive. For the current predictive filter density. p( x_t_k | y_{1:k-1} )
# output: posterior. For the current (k) filter density. p( x_t_k | y_{1:k} )
# posterior_tuple is (posterior, innovation) if rt == MLMode.
function computekalmaniter(
    rt::InnovationTrait,
    posterior_prev::MVNParameters,
    observation::ObservationInstance,
    latent::LatentTransition, # the discrete system transition of the latent process from the previous observation instance to current instance.
    y, # y_k.
    )
    
    # Kalman prediction step.
    predictive = computepredictive(posterior_prev, latent)

    # Kalman update step.
    posterior_tuple = computemarginalposterior(
        rt, predictive, observation, y,
    )
    
    return predictive, posterior_tuple
end

# Prediction step:
# output: predictive. current filtration element, k.
function computepredictive(
    posterior::NT, # previous filtration element, k-1.
    latent::LatentTransition, # the discrete system transition of the latent process from the previous filtration element to the current filtration element.
    )::NT where NT <: MVNParameters

    m, P = posterior.m, posterior.P
    A, Σ = latent.A, latent.Σ
    
    # # Prediction step:
    out_m = A*m
    out_P = A * P * A' .+ Σ

    return MVNParametersSI(out_m, out_P)
end

# Kalman update step.
# second filtration time. p(x(t_1) | y_1). The predictive is just p(x_t_1), the prior.
# output is posterior for # for the current (k) filter density. p( x_t_k | y_{1:k} )
function computemarginalposterior(
    rt::InnovationTrait,
    predictive::MVNParameters,
    observation::ObservationInstance,
    y::AbstractFloat, # y_k.
    )
    
    #A_prev, Σ_prev = latent.A, latent.Σ
    Ht, R = observation.Ht, observation.R
   
    # # buffers.
    # v = y - H * predictive.m
    # B = predictive.P * H'
    # S = H * B + R
    # # updates.
    # posterior.m[:] = predictive.m + B*S\v
    # posterior.P[:] = predictive.P - B*S\(B')

    m, P = predictive.m, predictive.P
    v = y - dot(Ht, m)

    # # optimized for axpy!.
    k2 = P*Ht
    s = dot(Ht,k2) + R

    out_m = m .+ k2 .* (v/s)
    out_P = P .- k2*k2' .* (1/s)

    #return MVNParametersSI(out_m, out_P)
    return returnkalman(rt, MVNParametersSI(out_m, out_P), v, s)
end

function returnkalman(::KeepInnovation, A::MVNParameters, v::AbstractFloat, s::AbstractFloat)
    return A, v, s
end

function returnkalman(::DiscardInnovation, A::MVNParameters, args...)
    return A
end


# # marginal likelihood via innovation mean and variance. 

struct MLBuffers{FT <: FilterDistributions, LT <: LatentTransition}
    #
    FD::FT
    latents::Vector{LT}
end

# front end, set up.
"""
    setup_ml(
        θ_sde::Kernel1D,
        ts::TimeVarType,
        y::Vector{T};
        σ² = one(T),
    ) where T <: Real

Returns a buffer variable of type `MLBuffers` for use with `eval_ml!`, which computes the marginal likelihood. This avoids additional creation and allocation of this buffer variable if the marginal likelihood is to be computed multiple times for different hyperparameters.
"""
function setup_ml(
    θ_sde::Kernel1D,
    ts::TimeVarType,
    y::Vector{T};
    σ² = one(T),
    ) where T <: Real

    sde = kernel2sde(θ_sde)

    system = setupdiscretesystem(sde, ts, y, σ²)
    prior = kernel2prior(θ_sde)
    FD = allocate_FilterDistributions(prior, length(y))

    return MLBuffers(FD, system.latent_steps)
end

function setup_ml(ts::TimeVarType, y::Vector{T}; σ² = one(T)) where T <: Real

    θ_sde = create_Matern3HalfsKernel(one(T), one(T))
    return setup_ml(θ_sde, ts, y; σ² = σ²)
end

# front end, recurring calls for when θ_sde or σ² changes..
"""
    eval_ml!(
        B::MLBuffers, # mutates, buffer
        θ_sde::Kernel1D,
        ts::TimeVarType,
        y::Vector,
        σ²::Real,
    )

Inputs:
- `buffer` is the return variable from `setup_ml`.
- `θ_sde` is the covariance function variable.
- `ts` is the set of training inputs.
- `y` is the set of training outputs.
- `σ²` is the observation noise variance. Increase this if you experience numerical issues.

Returns the log marginal likelihood over the training set. Additive constants are dropped.
"""
function eval_ml!(
    B::MLBuffers, # mutates, buffer
    θ_sde::Kernel1D,
    ts::TimeVarType,
    y::Vector,
    σ²::Real,
    )

    sde = kernel2sde(θ_sde)
    system = setupdiscretesystem!(B.latents, sde, ts, y, σ²)
    prior = kernel2prior(θ_sde)
    
    return eval_ml_internal!(B.FD, system, prior)
end

# based on runkalmanfilter!().
function eval_ml_internal!(
    FD::FilterDistributions, # mutates, buffer.
    system::DiscreteDiscreteSystem,
    prior::MVNParameters,
    )
    
    posteriors, predictives = FD.posteriors, FD.predictives

    # initialization of the filtration.
    posteriors[begin], v, s = computemarginalposterior(
        KeepInnovation(),
        prior,
        system.observations[begin],
        system.measurements[begin],
    )
    lli = computeML(v, s)
    #predictives[begin] = prior

    # subsequent entries.
    for k in Iterators.drop(eachindex(predictives), 1)
        _, (posteriors[k], v, s) = computekalmaniter(
            KeepInnovation(),
            posteriors[k-1],
            system.observations[k],
            system.latent_steps[k-1],
            system.measurements[k],
        )

        lli += computeML(v, s)
    end

    return lli
end

function eval_ml_internal(
    system::DiscreteDiscreteSystem,
    prior::MVNParameters,
    )

    # initialization of the filtration.
    post_current, v, s = computemarginalposterior(
        KeepInnovation(),
        prior,
        system.observations[begin],
        system.measurements[begin],
    )
    #post_prev = copy_mvn(post_current)
    post_prev = post_current
    lli = computeML(v, s)

    # subsequent entries.
    N = getNobs(system)
    for k = 2:N
        _, (post_current, v, s) = computekalmaniter(
            KeepInnovation(),
            post_prev,
            system.observations[k],
            system.latent_steps[k-1],
            system.measurements[k],
        )
        post_prev = post_current

        lli += computeML(v, s)
    end

    return lli
end

# inputs are the innovation mean dna variance, respectively.
function computeML(v::T, s::T)::T where T <: AbstractFloat
    return -log(s) -v*v/s # without the 2pi constant.
    #return (-log(2*pi*s) -v*v/s)/2
end