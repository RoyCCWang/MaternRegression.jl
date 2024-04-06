

# LTI is linaer time-invariant.

# ### Ornstein-Uhlenbeck, or exponential kernel.
struct ExponentialKernel{T <: AbstractFloat} <: Kernel1D
    λ::Vector{T} # single-entry.
    q::Vector{T} # single-entry.
end

function ExponentialKernel(λ::T, q::T)::ExponentialKernel{T} where T <: AbstractFloat
    @assert λ > zero(T)
    @assert q > zero(T)

    return ExponentialKernel([λ;], [q;])
end

function kernel2sde(θ::ExponentialKernel{T})::ConstantOU{T} where T <: AbstractFloat
    return ConstantOU(θ.λ[begin], θ.q[begin])
end

# function kernel2prior_dm(θ::ExponentialKernel{T})::MVNParametersDM{T} where T <: AbstractFloat
#     λ, q = θ.λ[begin], θ.q[begin]
#     return MVNParametersDM(zero(T), q/(2*λ))
# end

function kernel2prior(θ::ExponentialKernel{T})::MVNParametersSI{T} where T <: AbstractFloat
    λ, q = θ.λ[begin], θ.q[begin]
    return MVNParametersSI(SVector{1}(zero(T)), SMatrix{1,1}(q/(2*λ)))
end

# ### Matern (3/2).
struct Matern3HalfsKernel{T <: AbstractFloat} <: Kernel1D
    a::T # this is λ
    b::T # this is q/(4*λ^2).
end

"""
    create_Matern3HalfsKernel(λ::T, b::T)::Matern3HalfsKernel{T} where T <: AbstractFloat

Creates a Matern covariance function variable with parameters `λ` and `b`.
Given two time inputs `x` and `z`, the distance between two time inputs, this covariance function has the following formula:
```
output = b*(1 + λ*norm(x-z))*exp(-λ*norm(x-z))
```
"""
function create_Matern3HalfsKernel(a::T, b::T)::Matern3HalfsKernel{T} where T <: AbstractFloat
    @assert a > zero(T)
    @assert b > zero(T)
    return Matern3HalfsKernel(a, b)
end

"""
    create_Matern3HalfsKernel(λ::T)::Matern3HalfsKernel{T} where T <: AbstractFloat

Returns `create_Matern3HalfsKernel(λ, one(T))`
"""
function create_Matern3HalfsKernel(a::T)::Matern3HalfsKernel{T} where T <: AbstractFloat
    @assert a > zero(T)
    return Matern3HalfsKernel(a, one(T))
end

# create place-holder, dummy kernel.
"""
    create_Matern3HalfsKernel(::Type{T})::Matern3HalfsKernel{T} where T <: AbstractFloat

Returns a template or dummy variable by calling `create_Matern3HalfsKernel(one(T), one(T))`
"""
function create_Matern3HalfsKernel(::Type{T})::Matern3HalfsKernel{T} where T <: AbstractFloat
    return Matern3HalfsKernel(one(T), one(T))
end

function kernel2sde(θ::Matern3HalfsKernel{T})::Matern3HalfsSDE{T} where T <: AbstractFloat
    λ, b = θ.a[begin], θ.b[begin]
    q = 4*b*λ^3
    return Matern3HalfsSDE(λ, q)
end

# Eq. 12.49, Sarkka book 2019.
# function kernel2prior(θ::Matern3HalfsKernel{T})::MVNParametersDM{T} where T <: AbstractFloat
#     λ, b = θ.a[begin], θ.b[begin]
#     #q = 4*b*λ^3
    
#     P = zeros(T, 2,2)
#     P[1] = b    
#     P[end] = b*λ^2

#     return MVNParametersDM(zeros(T,2), P)
# end

# Eq. 12.49, Sarkka book 2019.
function kernel2prior(θ::Matern3HalfsKernel{T})::MVNParametersSI{T} where T <: AbstractFloat
    λ, b = θ.a[begin], θ.b[begin]
    return MVNParametersSI(SVector{2}(zero(T), zero(T)), SMatrix{2,2}(b, zero(T), zero(T), b*λ^2))
end

# # Specific common SDEs

# dimension of x(t) for any given t, for this SDE.
function getstatedim(s::SDEContainer)
    return length(s.Ht)
end


# ## LTI Ornstein-Uhlenbeck (OU) SDE.
struct ConstantOU{T <: AbstractFloat} <: SDEContainer
    λ::Vector{T} # single-element.
    q::Vector{T} # single-element.
    c::Vector{T} # q/(2*λ)

    H::Matrix{T} # ones(1,1) # TODO
end

function ConstantOU(λ::T, q::T) where T <: AbstractFloat
    @assert λ > zero(T)
    @assert q > zero(T)

    return ConstantOU([λ;], [q;], [q/(2*λ);], ones(T,1,1))
end


# get the transition from time tq to time t_next.
function discretizeSDE(
    p::ConstantOU{T},
    t_current::T,
    t_next::T,
    ) where T
    
    @assert t_next >= t_current
    
    Δt = t_next - t_current # ts[i+1] - ts[i]
    
    λ, c = p.λ[begin], p.c[begin]
    A = SMatrix{1,1}(exp(-λ*Δt))
    Σ = SMatrix{1,1}(c * (one(T)-exp(-2*λ*Δt)))

    return LatentTransitionSI(A, Σ)
end

# ## Matern (3/2).

struct Matern3HalfsSDE{T <: AbstractFloat, VT <: SVector} <: SDEContainer
    # λ::Vector{T} # single-element. # a.
    # q::Vector{T} # single-element. # 4*b*λ^3
    λ::T # a.
    q::T # 4*b*λ^3
    ln_λ::T
    ln_q::T

    #c::Vector{T} # b*λ^2
    Σ_lb::T
    Σ_ub::T

    #H::Matrix{T} # [1 0;]
    Ht::VT
end

function Matern3HalfsSDE(
    λ::T, q::T;
    Σ_lb::T = eps(T)*100,
    Σ_ub::T = floatmax(T)/100,
    ) where T <: AbstractFloat

    @assert λ > zero(T)
    @assert q > zero(T)

    #b = 1/(4*λ^3)
    #H = [one(T) zero(T);]
    Ht = SVector{2, Int}(1, 0)
    #Ht = [one(T); zero(T)]
    
    return Matern3HalfsSDE(λ, q, log(λ), log(q), Σ_lb, Σ_ub, Ht)
end

# NOTE profiler shows this is slow.
function computeΣMatern32(
    ln_λ::T, ln_q::T, ln_s::T;
    diag_lb::T = eps(T)*100,
    diag_ub = floatmax(T)/100,
    ) where T <: AbstractFloat
    
    # Based on Eq. 6.71, Sarkka SDE book 2019.
    # A direct implementation of that equation has overflow/underflow issues when λ*s approaches 0.
    # Use log-sum-exp, expm1, log1p.
    λs = exp( ln_λ + ln_s )
    two_λs = 2*λs
    
    #ϵ11 = exp( getlog2(T) + ln_λ + ln_s + log1p(λs) ) # might be more accurate, but definately slower.
    ϵ11 = two_λs*(λs+1)
    η11 = -two_λs + log1p(ϵ11)
    s11 = exp(ln_q - getlog4(T) - 3*ln_λ)
    c11 = -expm1(η11)*s11 # NOTE profiler shows this is slow.

    ϵ22 = two_λs*(λs-1)
    
    η22 = -two_λs + log1p(ϵ22)
    s22 = exp(ln_q - getlog4(T) - ln_λ)
    c22 = -expm1(η22)*s22 # NOTE profiler shows this is slow.

    c12 = exp( ln_q + 2*ln_s -two_λs - getlog2(T) )
    
    # in case if λs is still too small for what we have here, we should still guard the covmat.
    # the limit of c12 as λs -> 0 is 0, so to get C to be posdef, we need to ensure c11 and c22 are positive.
    c11 = clamp(c11, diag_lb, diag_ub)
    c22 = clamp(c22, diag_lb, diag_ub)

    return c11, c12, c22
end

# static size, immutable.
function discretizeSDE(
    p::Matern3HalfsSDE{T},
    t_current::T,
    t_next::T,
    ) where T
    
    @assert t_next >= t_current
    
    Δt = t_next - t_current # ts[i+1] - ts[i]
    
    #λ, q = p.λ[begin], p.q[begin]
    λ = p.λ

    s = Δt
    as = λ*Δt
    eas = exp(-as)

    a11 = eas*(as+1)
    a21 = eas*(-as*λ)
    a12 = eas*s
    a22 = eas*(1-as)
    A = SMatrix{2,2}(a11, a21, a12, a22)
    
    s11, s12, s22 = computeΣMatern32(
        p.ln_λ, p.ln_q, log(Δt);
        #p.λ, p.q, Δt;
        diag_lb = p.Σ_lb,
        diag_ub = p.Σ_ub,
    )
    Σ = SMatrix{2,2}(s11, s12, s12, s22)

    return LatentTransitionSI(A, Σ)
end



# starts with zeros.
function create_latent_transition(::Matern3HalfsSDE{T}) where T <: AbstractFloat
    return LatentTransitionSI(
        SMatrix{2, 2}(zero(T), zero(T), zero(T), zero(T)),
        SMatrix{2, 2}(zero(T), zero(T), zero(T), zero(T)),
    )
end

# starts with zeros.
function create_mvn(::Matern3HalfsSDE{T}) where T <: AbstractFloat
    return MVNParametersSI(
        SVector{2}(zero(T), zero(T)),
        SMatrix{2, 2}(zero(T), zero(T), zero(T), zero(T)),
    )
end

# # common SDE GP setup routines.

# get equivalent discrete-time system from a SDE and discretization times, ts.
# This updates A and Σ for each step.
function getequivalentsys!(
    steps::Vector{LT}, ts, sde::SDEContainer,
    ) where LT <: LatentTransition
    
    @assert length(ts) == length(steps) + 1
    for i in eachindex(steps)
        steps[i] = discretizeSDE(sde, ts[i], ts[i+1])
    end

    return nothing
end

# IID observation model, scalar-valued regression latent and measurement process.
# ys, sde should not be modified for the life-time of the output.
function setupdiscretesystem(
    sde::SDEContainer,
    ts,
    ys::Vector{T}, # not AbstractVector since we're putting it into DiscreteDiscreteSystem
    σ²::T,
    ) where T <: AbstractFloat
    
    N = length(ys)
    @assert length(ts) == N

    latents = collect(
        discretizeSDE(sde, ts[begin+i-1], ts[begin+i])
        for i in Iterators.take(eachindex(ts), length(ts)-1)
    )

    return DiscreteDiscreteSystem(
        latents,
        IIDobservations(sde.Ht, σ², length(ys)),
        ys,
    )
end

# ys, sde should not be modified for the life-time of the output.
function setupdiscretesystem!(
    latents::Vector{LT},
    sde::SDEContainer,
    ts,
    ys::Vector{T}, # not AbstractVector since we're putting it into DiscreteDiscreteSystem
    σ²::T,
    ) where {T <: AbstractFloat, LT <: LatentTransition}
    
    N = length(ys)
    @assert length(ts) == N
    
    resize!(latents, N-1)
    for i in eachindex(latents)
        latents[i] = discretizeSDE(sde, ts[begin+i-1], ts[begin+i])
    end

    return DiscreteDiscreteSystem(
        latents,
        IIDobservations(sde.Ht, σ², length(ys)),
        ys,
    )
end
