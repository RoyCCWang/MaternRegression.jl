# RTS smoother.

# ## dynamic size, mutable content; i.e., MVNParameters.

function runRTSsmoother(
    system::DiscreteDiscreteSystem,
    prior::NT,
    ) where NT <: MVNParameters

    FD = allocate_FilterDistributions(prior, getNobs(system))
    posteriors = Vector{NT}(undef, getNobs(system))

    runRTSsmoother!(FD, posteriors, system, prior)

    return FD, posteriors
end

function runRTSsmoother!(
    FD::FilterDistributions,
    posteriors::Vector{NT},
    system::DiscreteDiscreteSystem,
    prior::MVNParameters,
    ) where NT <: MVNParameters

    # # Kalman.
    runkalmanfilter!(FD, system, prior)

    # # RTS smoother
    run_backwards_recursion!(posteriors, FD, system)

    return nothing
end

# assumes the filter distributions are computed.
function run_backwards_recursion!(
    posteriors::Vector{NT}, # mutates, output.
    FD::FilterDistributions,
    system::DiscreteDiscreteSystem,
    ) where NT <: MVNParameters
    
    N = getNobs(system)
    @assert N == length(posteriors) == length(FD.predictives) == length(FD.posteriors)

    posteriors[end] = FD.posteriors[end]

    for k = N-1:-1:1
        posteriors[k] = run_backwards_recursion_iter(
            posteriors[k+1],
            FD.predictives[k+1],
            FD.posteriors[k],
            system.latent_steps[k],
        )
    end

    return nothing
end

# output: posterior of current filtration element, k.
function run_backwards_recursion_iter(
    posterior_next::MVNParameters, # next filtration element, k+1.
    predictive_next::MVNParameters, # next filtration element, k+1.
    marginal_posterior::MVNParameters, # current filtration element, k.
    latent::LatentTransition, # the discrete system transition of the latent process from the the current (k) filtration element to the next (k+1) filtration element.    
    )

    pred, marg = predictive_next, marginal_posterior
    p_next = posterior_next
    A = latent.A

    # Original
    Gt = pred.P \ (A*marg.P)
    out_m = marg.m .+ Gt'*(p_next.m .- pred.m)
    out_P = marg.P .+ Gt'*(p_next.P .- pred.P)*Gt

    return MVNParametersSI(out_m, out_P)
end


###### query.

# assume ts is sorted in ascending order.
# assumes FD and posteriors are computed for all entries in ts.
# assumes tq ∈ [ts[1], ts[end]].
# output: query_dist.
function queryposterior(
    #predictive_ta_tq::MVNParameters, # mutates, buffer.
    #mP_buffer::MVNParameters, # mutates, buffer
    posteriors::Vector{NT},
    FD::FilterDistributions,
    sde::SDEContainer,
    ts,
    tq::T,
    ) where {T <: AbstractFloat, NT <: MVNParameters}

    # end point cases.
    if tq == ts[begin]
        return copy_mvn(posteriors[begin])
    end

    if tq == ts[end]
        return copy_mvn(posteriors[end])
    end

    if !(ts[begin] < tq < ts[end])
        
        println("Error: tq is not in ts. Returning NaN for query distribution parameters.")
        return create_invalid_mvn(posteriors[begin])
    end

    ind = findfirst(xx->xx>tq, ts)
    if isnothing(ind) || ind == 1
        
        println("Unknown error. Returning NaN for query distribution parameters.")
        return create_invalid_mvn(posteriors[begin])
    end

    # prepare current filtration's discretization system, A_k, Σ_k.
    k_b = ind
    k_a = ind - 1
    #Δt = ts[k_next] - tq
    # we've: t_k <= tq <= t_{k_next}
    
    # t_k_next.
    posterior_tb = posteriors[k_b]
    predictive_tb = FD.predictives[k_b]
    
    # get predictive distribution at tq, using Kalman and ta, which is forward.
    latent_ta_tq = discretizeSDE(sde, ts[k_a], tq)

    predictive_ta_tq = computepredictive(
        FD.posteriors[k_a],
        latent_ta_tq,
    )
    
    # # set up for getting the posterior at tq using tb and RTS, which is done backwards.
    latent_tq_tb = discretizeSDE(sde, tq, ts[k_b])

    # get GP query distribution.
    marginal_posterior_tq = predictive_ta_tq
    query_dist = run_backwards_recursion_iter(
        posterior_tb,
        predictive_tb,
        marginal_posterior_tq,   
        latent_tq_tb,
    )

    return query_dist
end

# batch version.
function batchqueryposterior!(
    dists::Vector{NT}, # mutates, output.
    posteriors::Vector{NT},
    FD::FilterDistributions,
    sde::SDEContainer,
    ts,
    tqs,
    ) where NT <: MVNParameters

    @assert length(dists) == length(tqs)

    for m in eachindex(tqs)
        dists[m] = queryposterior(
            posteriors,
            FD,
            sde,
            ts,
            tqs[m],
        )
    end

    return nothing
end


function batchqueryposterior(
    posteriors::Vector{NT},
    FD::FilterDistributions,
    sde::SDEContainer,
    ts,
    tqs,
    ) where NT <: MVNParameters

    Nq = length(tqs)
    dists = collect( create_mvn(sde) for _ = 1:Nq )
    
    batchqueryposterior!(dists, posteriors, FD, sde, ts, tqs)
    return dists
end


# apply the H matrix to get GP guqery distributions.

function applyaffinetransform(
    ::Type{T},
    dists::Vector{NT},
    Ht::AbstractVector,
    ) where {T <: AbstractFloat, NT <: MVNParameters}
    
    M = length(dists)
    mqs = Vector{T}(undef, M)
    vqs = Vector{T}(undef, M)
    applyaffinetransform!(mqs, vqs, dists, Ht)

    return mqs, vqs
end

function applyaffinetransform!(
    mqs::Vector{T}, # mutates, output.
    vqs::Vector{T}, # mutates, output.
    dists::Vector{NT},
    Ht::AbstractVector,
    ) where {T <: AbstractFloat, NT <: MVNParameters}
    
    @assert !isempty(dists)
    
    D = getdim(dists[begin])
    @assert length(Ht) == D
    h = Ht
    
    M = length(dists)
    resize!(mqs, M)
    resize!(vqs, M)

    for m in eachindex(dists)
        mqs[m] = dot(h, dists[m].m) # H*m, mean under linear transformation.
        vqs[m] = dot(h, dists[m].P, h) # H*P*H', covmat under linear transformation.
    end

    return nothing
end