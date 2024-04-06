"""
    const TimeVarType = Union{AbstractVector, AbstractRange}

A supertype for specifying the set of training inputs.
"""
const TimeVarType = Union{AbstractVector, AbstractRange}

abstract type MVNParameters end

# parameters of the multi-variate normal density.
# dynamic size, mutable content.
struct MVNParametersDM{T <: AbstractFloat} <: MVNParameters
    m::Vector{T}
    P::Matrix{T}
end

# static size, mutable content.
struct MVNParametersSI{T <: AbstractFloat, N, L} <: MVNParameters
    m::SVector{N,T}
    P::SMatrix{N,N,T,L}
end


function getdim(A::MVNParameters)
    return length(A.m)
end

# # default..Type Vector{T} and Matrix{T}
# function create_mvn(::Type{T}, D::Integer) where T <: AbstractFloat
#     return MVNParametersDM(zeros(T, D), diagm(ones(T,D)))
# end


# function create_MVector_mvn(::Type{T}, s::DiscreteDiscreteSystem) where T <: AbstractFloat
#     D = getstatedim(s)

#     return MVNParameters(zeros(T, D), diagm(ones(T,D)))
# end


function getinvalidMVN(::Type{T}, D::Integer) where T <: AbstractFloat
    
    m = zeros(T, D)
    fill!(m, convert(T, NaN))

    P = zeros(T, D,D)
    fill!(P, convert(T, NaN))
    return MVNParameters(m, P)
end

# function fillinvalid!(A::MVNParametersDM{T}) where T <: AbstractFloat
    
#     fill!(A.m, convert(T, NaN))
#     fill!(A.P, convert(T, NaN))
    
#     return nothing
# end

# function copydatastructure!(dest::MVNParametersDM, src::MVNParameters)
#     dest.m[:] = src.m
#     dest.P[:] = src.P
    
#     return nothing
# end

function copy_mvn(N::NT)::NT where NT <: MVNParameters
    return NT(copy(N.m), copy(N.P))
end

function create_invalid_mvn(N::NT)::NT where NT <: MVNParameters
    
    Sm, SP = copy(N.m), copy(N.P)
    
    # created mutable version.
    Mm, MP = MVector(Sm), MMatrix(SP)
    fill!(Mm, NaN)
    fill!(MP, NaN)

    return NT(Mm, MP)
end


# # 1-D case.
# function MVNParametersDM(m::T, v::T) where T <: AbstractFloat
#     P = ones(T, 1, 1)
#     P[begin] = v
#     return MVNParametersDM([m;], P)
# end
# 1-D case.
# function create_mvn(m::T, v::T) where T <: AbstractFloat
#     return MVNParametersSI(SVector{1}(m), SMatrix{1,1}(v))
# end

# helper.
function getmoments(As::Vector{NT}) where NT <: MVNParameters
    ms = collect( As[n].m for n in eachindex(As) )
    Ps = collect( As[n].P for n in eachindex(As) )
    return ms, Ps
end

# function getscalarmoments(As::Vector{MVNParameters{T}})::Tuple{Vector{T}, Vector{T}} where T <: AbstractFloat
#     ms = collect( As[n].m[begin] for n in eachindex(As) )
#     vs = collect( As[n].P[begin] for n in eachindex(As) )
#     return ms, vs
# end


####

abstract type LatentTransition end

# dynamically sized, mutable contents.
# one step from the equivalent discrete system of a linear SDE with MVN distribution as intiial condition.
# takes the latent process x from x(t0) to x(t1).
struct LatentTransitionDM{T <: AbstractFloat} <: LatentTransition
    A::Matrix{T} # D x D, D is dim(x_t).
    Σ::Matrix{T} # D x D
end
# # default.
# function LatentTransitionDM(::Type{T}, D::Integer) where T <: AbstractFloat
#     return LatentTransitionDM(diagm(ones(T, D)), diagm(ones(T, D)))
# end

# static size, immutable contents.
struct LatentTransitionSI{T <: AbstractFloat, N, L} <: LatentTransition # L must be N*N.
    # D is dim(x_t).
    A::SMatrix{N,N,T, L} # D x D
    Σ::SMatrix{N,N,T,L} # D x D
end

# function LatentTransition(a::T, s::T) where T <: AbstractFloat
#     A = Matrix{T}(undef, 1, 1)
#     A[begin] = a
    
#     Σ = Matrix{T}(undef, 1, 1)
#     Σ[begin] = s

#     return LatentTransition(A, Σ)
# end

function getstatedim(L::LatentTransition)
    return size(L.A,1)
end

# a discrete-time observation.
struct ObservationInstance{T <: Real, VT <: AbstractVector}
    # D_y is dim(y_k). This is 1, since we're doing scalar-valued regression here.
    #H::Matrix{T} # D_y x D.
    #R::Matrix{T} # D_y x D_y.
    Ht::VT
    R::T
end

# function IIDobservations(σ²::T, N::Integer) where T <: AbstractFloat
#     H = ones(T, 1, 1)
    
#     R = Matrix{T}(undef, 1, 1)
#     R[begin] = σ²

#     return collect(
#         ObservationInstance(H, Σ) for _ = 1:N
#     )
# end

# function IIDobservations_old(H::Matrix{T}, σ²::T, N::Integer) where T
#     @assert size(H, 1) == 1 # scalar-valued measurement case.

#     R = Matrix{T}(undef, 1, 1)
#     R[begin] = σ²
    
#     return collect(
#         ObservationInstance(H, R) for _ = 1:N
#     )
# end

function IIDobservations(Ht::AbstractVector, σ²::Real, N::Integer)
    return collect(
        ObservationInstance(Ht, σ²) for _ = 1:N
    )
end

# discrete system with linear Guassian observation model.
# #q. 19.66, pg 216, Sarkka SDE book 2019.
# dx = F_t x dt + L_t dβ
# y_k = H_k x_{t_k} + r_k, 
# r_k ~ N(0, R_k), β has diffusion matrix Q.
struct DiscreteDiscreteSystem{T <: AbstractFloat, OT <: ObservationInstance, RT <: LatentTransition}
    
    # N -1 entries. The k-th entry contain the affine transformation that takes p(x_{k}) to p(x_{k+1}).
    latent_steps::Vector{RT}

    # N entries. index k.
    observations::Vector{OT}
    
    #measurements::Vector{Vector{T}} # {y_k}, N entries. Assumes each y_k is vector-valued. Defines the filtration of the Kalman filter marginal posterior and predicitive distributions.
    measurements::Vector{T} # {y_k}, N entries. Defines the filtration of the Kalman filter marginal posterior and predicitive distributions.
    # TODO specialize for scalar-valued GP regression: H = ones(1,1), R = σ², y::Vector{T}. This would speed up the filtering operations.
end

function getNobs(s::DiscreteDiscreteSystem)
    return length(s.measurements)
end

function getstatedim(s::DiscreteDiscreteSystem)
    return getstatedim(s.latent_steps[begin])
end

function getHts(s::DiscreteDiscreteSystem)
    return collect( s.observations[n].Ht for n in eachindex(s.observations) )
end

function getRs(s::DiscreteDiscreteSystem)
    return collect( s.observations[n].R for n in eachindex(s.observations) )
end

function getAs(s::DiscreteDiscreteSystem)
    return collect( s.latent_steps[n].A for n in eachindex(s.latent_steps) )
end

function getΣs(s::DiscreteDiscreteSystem)
    return collect( s.latent_steps[n].Σ for n in eachindex(s.latent_steps) )
end

# marginal posteriors for every instance in a filtration.
# Eq 6.11, 6.12, pg 94, Sarkka's 2013 book.
struct FilterDistributions{NT <: MVNParameters}
    predictives::Vector{NT} # predictive distribution for state x(t_k): p(x_k | y_{1:k-1}).
    posteriors::Vector{NT} # posterior distribution for state x(t_k): p(x_k | y_{1:k}).
end

function allocate_FilterDistributions(::NT, N::Integer) where NT <: MVNParameters

    return FilterDistributions(Vector{NT}(undef, N), Vector{NT}(undef, N))
end

# function resetdistributions!(FD::FilterDistributions, prior::MVNParameters)
#     FD.predictives[begin] = prior
#     return nothing
# end

function getmarginalposteriors(F::FilterDistributions)
    return collect( F.posteriors[n] for n in eachindex(F.posteriors) )
end

function getpredictives(F::FilterDistributions)
    return collect( F.predictives[n] for n in eachindex(F.predictives) )
end

# continuous-discrete linear SDE, Gaussian initial conditions.

# continuous-discrete LTI SDE, Gaussian initial conditions.

#### covariance functions.
abstract type Kernel1D end

#### SDE containers.
abstract type SDEContainer end

# control whether the Kalman innovation distribution is returned
abstract type InnovationTrait end
struct KeepInnovation <: InnovationTrait end
struct DiscardInnovation <: InnovationTrait end

# whether to treat the gain kernel parameter, b, as a variable or fix it to 1.

"""
    abstract type GainTrait end

Subtypes are `InferGain` and `UnityGain`
"""
abstract type GainTrait end

"""
    struct InferGain <: GainTrait end
    
This trait specifies that the hyperparameter ordering of the parameter array is `λ`, `σ²`, `b`.
"""
struct InferGain <: GainTrait end # infers `b`, the gain.

"""
    struct UnityGain <: GainTrait end
    
This trait specifies that the hyperparameter ordering of the parameter array is `λ`, `σ²`. The `b` hyperparameter is fixed at `1`.
"""
struct UnityGain <: GainTrait end # fixes `b= 1`.
