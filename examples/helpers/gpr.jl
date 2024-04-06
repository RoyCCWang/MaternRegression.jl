# conventional GPR routines.

function fitgpr(
    θ,
    X_in::Union{AbstractVector, AbstractRange},
    y::Vector{T},
    σ²::T, 
    ) where T <: Real

    # RKHSRegularization.jl require each input be of type Vector.
    X = collect( [x;] for x in X_in )

    η = RK.Problem(RK.UseCholeskyGP(), X, θ, RK.Variance(σ²))
    RK.fit!(η, y)

    return η
end

function querygpr(
    θ,
    Xq_in::Union{AbstractArray, AbstractRange},
    X_in::Union{AbstractVector, AbstractRange},
    y::Vector{T},
    σ²::T, 
    ) where T <: Real

    # RKHSRegularization.jl require each input be of type Vector.
    
    Xq = collect( [x;] for x in Xq_in )

    η = fitgpr(θ, X_in, y, σ²)

    # allocate output.
    Nq = length(Xq)
    mq = Vector{T}(undef, Nq)
    vq = Vector{T}(undef, Nq)

    # query.
    RK.batchqueryGP!(mq, vq, Xq, η)

    return mq, vq, η
end

# Eq. 5.8, GPML book, without the multiplicative constant of 0.5 and the additive constant.
function evalnlli_gpr(η, y)

    # dot(y, Ky\y) - logdet(Ky)
    #C = cholesky(η.U)
    #L = C.L
    #logdet_term = 2*sum( log(L[i,i]) for i in axes(L,1) ) # might need guard against non-positive entries being in the diagonal of L when we compute it.
    logdet_term = logdet(η.U)

    #@show dot(y, η.c), logdet_term
    log_likelihood_eval = -dot(y, η.c) - logdet_term

    return log_likelihood_eval
end

function evalnlli_sde()
    #


end
