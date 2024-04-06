# ts must be in ascending order.


struct SDEGP{ST <: SDEContainer, DT <: DiscreteDiscreteSystem, NT <: MVNParameters}
    
    sde::ST
    system::DT # might not need this during query.
    prior::NT # might not need this during query.

    SD::Vector{NT}
    FD::FilterDistributions{NT}
end

"""
    create_sdegp(
        θ_sde::Kernel1D,
        ts::TimeVarType,
        y::Vector{T},
        σ²::T,
    ) where T <: Real

To run Gaussian process query via the state-space implementation, one needs to do a cache phase then a query phase. This function runs the cache phase.

Inputs:
- `θ_sde` is a Matern kernel variable
- `ts` is the set of training inputs.
- `y` is an array of training outputs.
- `σ²` is the Gaussian process regression's observation variance.

Returns a variable of `SDEGP` data type, which is used in the query phase.
"""
function create_sdegp(
    θ_sde::Kernel1D,
    ts::TimeVarType,
    y::Vector{T},
    σ²::T,
    ) where T <: Real

    prior = kernel2prior(θ_sde)
    sde = kernel2sde(θ_sde)
    system = setupdiscretesystem(sde, ts, y, σ²)
    
    FD, SD = runRTSsmoother(system, prior)
    
    return SDEGP(sde, system, prior, SD, FD)
end

"""
    query_sdegp!(
        mqs::Vector{T}, # mutates, output.
        vqs::Vector{T}, # mutates, output.
        S::SDEGP,
        tqs,
        ts::TimeVarType,
    ) where T <: Real

To run Gaussian process query via the state-space implementation, one needs to do a cache phase then a query phase. This function runs the query phase.

Inputs:
- `mqs` Buffer that is mutated and stores the queried predictive means corresponding to entries in `tqs`.
- `vqs` Buffer that is mutated and stores the queried predictive variances corresponding to entries in `tqs`.
- `S` is the output of the cached phase; see `create_sdegp`.
- `tqs` is the set of query inputs.
- `ts` is the set of training inputs.

The size of `mqs` and `vqs` must be the same as `tqs`.

`ts` should be the same training inputs used to create the cache `S`, otherwise `S` needs to be recomputed via `create_sdegp`.

Returns `nothing`.
"""
function query_sdegp!(
    mqs::Vector{T}, # mutates, output.
    vqs::Vector{T}, # mutates, output.
    S::SDEGP,
    tqs,
    ts::TimeVarType,
    ) where T <: Real

    sde = S.sde

    # I am here. find out how to make batchqueryposterior and applyaffinetransform! more efficient.
    query_dists = batchqueryposterior(
        S.SD, S.FD, sde, ts, tqs,
    )

    applyaffinetransform!(
        mqs, vqs, query_dists, sde.Ht,
    )

    return mqs, vqs
end

"""
    query_sdegp(
        S::SDEGP,
        tqs,
        ts,
    ) where T <: Real

This function allocates the output buffers `mqs` and `vqs`, then calls `query_sdegp!`. See `query_sdegp!` for details on the inputs.

Returns `mqs` and `vqs`.
"""
function query_sdegp(::Type{T}, S::SDEGP, tqs, ts) where T <: AbstractFloat

    M = length(tqs)
    mqs = Vector{T}(undef, M)
    vqs = Vector{T}(undef, M)
    return query_sdegp!(mqs, vqs, S, tqs, ts)
end

# `s` is the standard deviation.
# `m` is the mean.
function draw_normal(m::T, s::T)::T where T <: Real
    return s*randn(T) + m
end

# S[i,j,l] is the i-th epoch, j-th model, l-th query position.
"""
    compute_mean_var(S::Array{T,3}) where T <: Real

Assumes `S` is a `M x N x K` array of drawn samples, where:
- `M` is the number of samples drawn from a model.
- `N` is the number of models.
- `K` is the number of query positions.

`compute_mean_var` computes the empirical mean and variance for each query position.

Outputs:
- `mqs` the empirical means. Length `K`.
- `vqs` the empirical variances. Length `K`.
"""
function compute_mean_var(S::Array{T,3}) where T <: Real

    mqs = zeros(T, size(S,3))
    vqs = zeros(T, size(S,3))
    for l in axes(S,3)
        S_l = @view S[:,:,l]
        mqs[l] = mean(S_l)
        vqs[l] = var(S_l)
    end

    return mqs, vqs
end

abstract type HPSamplesContainer end

function get_num_models(A::HPSamplesContainer)
    return length(A.λ)
end

struct HPSamplesUnit{T} <: HPSamplesContainer
    λ::Vector{T}
    σ²::Vector{T}
end

"""
    simulate_sdegps(
        λ_set::Vector{T},
        σ²_set::Vector{T},
        M::Integer,
        ts::TimeVarType,
        tqs,
        y::Vector,
    ) where T <: Real

Returns the drawn samples of the ensemble of Gaussian process models that are specified by `λ_set` and `σ²_set`. The number of ensemble models is the length of `λ_set`.

The gain `b` is set to `1` for the simulation.

Inputs:
- `λ_set` contain samples of the `λ` parameter. Same length as `σ²_set`.
- `σ²_set` contain samples of the `σ²` parameter.
- `M` is the number of samples `simulate_sdegps` simulates per model.
- `ts` is the ordered set of training inputs.
- `tqs` is the ordered set of query inputs.
- `y` is the ordered set of training outputs.

The output, `S`, is a `M x N x K` array, where `N` is the number of ensemble models, and `K` is the number of query inputs.
"""
function simulate_sdegps(
    λ_set::Vector{T},
    σ²_set::Vector{T},
    M::Integer,
    ts::TimeVarType,
    tqs,
    y::Vector,
    ) where T <: Real

    @assert length(λ_set) == length(σ²_set)

    return simulate_sdegps_internal(
        HPSamplesUnit(λ_set, σ²_set),
        M, ts, tqs, y,
    )
end

function parse_hp(S::HPSamplesUnit{T}, k::Integer) where T
    return S.λ[k], S.σ²[k], one(T)
end

struct HPSamplesInfer{T} <: HPSamplesContainer
    λ::Vector{T}
    σ²::Vector{T}
    b::Vector{T}
end


"""
    simulate_sdegps(
        λ_set::Vector{T},
        σ²_set::Vector{T},
        b_set::Vector{T},
        M::Integer,
        ts::TimeVarType,
        tqs,
        y::Vector,
    ) where T <: Real

Returns the drawn samples of the ensemble of Gaussian process models that are specified by `λ_set`, `σ²_set`, and `b_set`. The number of ensemble models is the length of `λ_set`.

Inputs:
- `λ_set` contain samples of the `λ` parameter. Same length as `σ²_set`.
- `σ²_set` contain samples of the `σ²` parameter.
- `b_set` contain samples of the `b` parameter.
- `M` is the number of samples `simulate_sdegps` simulates per model.
- `ts` is the ordered set of training inputs.
- `tqs` is the ordered set of query inputs.
- `y` is the ordered set of training outputs.

The output, `S`, is a `M x N x K` array, where `N` is the number of ensemble models, and `K` is the number of query inputs.
"""
function simulate_sdegps(
    λ_set::Vector{T},
    σ²_set::Vector{T},
    b_set::Vector{T},
    M::Integer,
    ts::TimeVarType,
    tqs,
    y::Vector,
    ) where T <: Real

    @assert length(λ_set) == length(σ²_set) == length(b_set)

    return simulate_sdegps_internal(
        HPSamplesInfer(λ_set, σ²_set, b_set),
        M, ts, tqs, y,
    )
end

function parse_hp(S::HPSamplesInfer{T}, k::Integer) where T
    return S.λ[k], S.σ²[k], S.b[k]
end

function simulate_sdegps_internal(
    C::HPSamplesContainer,
    N_epochs::Integer,
    ts,
    tqs,
    y::AbstractVector{T},
    ) where T <: Real
    
    N_query = length(tqs)
    N_models = get_num_models(C)
    
    S = zeros(T, N_epochs, N_models, N_query)
    for k in axes(S,2) # 1:N_models
        
        λ, σ², b = parse_hp(C, k)

        # get the gpr query mean and variance.
        θ_sde = create_Matern3HalfsKernel(λ, b)
        sde_gp = create_sdegp(θ_sde, ts, y, σ²)
        mqs, vqs = query_sdegp(T, sde_gp, tqs, ts)
        
        # draw a normal sample.
        for nq in eachindex(mqs)
            m = mqs[nq]
            s = sqrt(vqs[nq])
            #s = vqs[nq]

            for ne in axes(S,1)
                S[ne, k, nq] = draw_normal(m, s)
            end
        end
    end

    return S
end


# # marginal likelihood

function parse_params(::UnityGain, p, zero_tol)

    λ, σ² = p
    λ = max(λ, zero_tol)
    σ² = max(σ², zero_tol)
    return create_Matern3HalfsKernel(λ), σ² # θ_sde
end


function parse_params(::InferGain, p, zero_tol)

    λ, σ², b = p
    λ = max(λ, zero_tol)
    σ² = max(σ², zero_tol)
    b = max(b, zero_tol)
    return create_Matern3HalfsKernel(λ, b), σ² # θ_sde
end

# assumes p = [λ; σ²].
# This version might be useful for Enzyme.jl, and maybe Zygote.jl. Untested.
# a bit slower. Allocates more.
"""
    eval_ml(
        trait::GainTrait,
        p::Vector{T},
        ts::Union{AbstractRange, AbstractVector},
        y::Vector;
        zero_tol = eps(T)*100,
    ) where T <: AbstractFloat


Inputs:
- `trait` is a trait variable that specifies the order of the hyperparameters in `p`. See *trait-based dispatch* in the Julia documentation. If `typeof(trait) <: InferGain`, then the hyperparameters in `p` are ordered `[λ; σ²; b]`. If `typeof(trait) <: UnityGain`, then the hyperparameters in `p` are ordered `[λ; σ²]`, and `b` is set to `1`.
- `p` is an ordered set of hyperparameter as an array.
- `ts` is the set of training inputs.
- `y` is the set of training outputs.
- `zero_tol` needs to be a small positive number. It is the lower bound on the covariance function hyperparameters.

Returns the log marginal likelihood over the training set. Some additive constants might be dropped.
"""
function eval_ml(
    trait::GainTrait,
    p::Vector{T},
    ts::Union{AbstractRange, AbstractVector},
    y::Vector;
    zero_tol = eps(T)*100,
    ) where T <: AbstractFloat

    # parse.
    θ_sde, σ² = parse_params(trait, p, zero_tol)

    # get systems, prior.
    sde = kernel2sde(θ_sde)
    system = setupdiscretesystem(sde, ts, y, σ²)
    prior = kernel2prior(θ_sde)

    return eval_ml_internal(system, prior)
end

"""
    eval_ml!(
        trait::GainTrait,
        buffer::MLBuffers,
        p::Vector{T},
        ts::TimeVarType,
        y::Vector{T};
        zero_tol = eps(T)*100,
    ) where T <: AbstractFloat

Inputs:
- `trait` is a trait variable that specifies the order of the hyperparameters in `p`. See *trait-based dispatch* in the Julia documentation. If `typeof(trait) <: InferGain`, then the hyperparameters in `p` are ordered `[λ; σ²; b]`. If `typeof(trait) <: UnityGain`, then the hyperparameters in `p` are ordered `[λ; σ²]`, and `b` is set to `1`.
- `buffer` is the return variable from `setup_ml`.
- `p` is an ordered set of hyperparameter as an array.
- `ts` is the set of training inputs.
- `y` is the set of training outputs.
- `zero_tol` needs to be a small positive number. It is the lower bound on the covariance function hyperparameters.

Returns the log marginal likelihood over the training set. Some additive constants might be dropped.
"""
function eval_ml!(
    trait::GainTrait,
    buffer::MLBuffers,
    p::Vector{T},
    ts::TimeVarType,
    y::Vector{T};
    zero_tol = eps(T)*100,
    ) where T <: AbstractFloat
    
    # parse.
    θ_sde, σ² = parse_params(trait, p, zero_tol)

    eval_ml!(buffer, θ_sde, ts, y, σ²)
end

"""
    eval_ml!(
        buffer::MLBuffers,
        λ::T,
        σ²::T,
        ts::TimeVarType,
        y::Vector{T};
        zero_tol = eps(T)*100,
    ) where T <: Real

Inputs:
- `buffer` is the return variable from `setup_ml`.
- `λ` and `σ²` are hyperparameters. The `b` hyperparameter is set to 1.
- `ts` is the set of training inputs.
- `y` is the set of training outputs.
- `zero_tol` needs to be a small positive number. It is the lower bound on the covariance function hyperparameters.

Returns the log marginal likelihood over the training set. Some additive constants might be dropped.
"""
function eval_ml!(
    buffer::MLBuffers,
    λ::T,
    σ²::T,
    ts::TimeVarType,
    y::Vector{T};
    zero_tol = eps(T)*100,
    ) where T <: Real

    return eval_ml!(
        buffer, λ, σ², one(T), ts, y;
        zero_tol = zero_tol,
    )
end


"""
    eval_ml!(
        buffer::MLBuffers,
        λ::T,
        σ²::T,
        b::T,
        ts::TimeVarType,
        y::Vector{T};
        zero_tol = eps(T)*100,
    ) where T <: Real

Inputs:
- `buffer` is the return variable from `setup_ml`.
- `λ`, `σ²`, and `b` are hyperparameters.
- `ts` is the set of training inputs.
- `y` is the set of training outputs.
- `zero_tol` needs to be a small positive number. It is the lower bound on the covariance function hyperparameters.

Returns the log marginal likelihood over the training set. Some additive constants might be dropped.
"""
function eval_ml!(
    buffer::MLBuffers,
    λ::T,
    σ²::T,
    b::T,
    ts::TimeVarType,
    y::Vector{T};
    zero_tol = eps(T)*100,
    ) where T <: Real

    λ = max(λ, zero_tol)
    σ² = max(σ², zero_tol)
    b = max(b, zero_tol)

    θ_sde = create_Matern3HalfsKernel(λ, b)
    return eval_ml!(buffer, θ_sde, ts, y, σ²)
end

# assumes p = [λ; σ²]
# outputs θ_sde, σ²
# function parse_ml_result(::Matern3HalfsKernel, p::Vector{T}) where T <: AbstractFloat
#     return create_Matern3HalfsKernel(p[begin], one(T)), p[end]
# end

"""
    parse_ml_result(trait::GainTrait, p::Vector{T}) where T <: AbstractFloat

Returns a Matern covariance function variable with the hyperparameters specified in `p`. The ordering of the hyperparameters in `p` is specified by `trait`.
"""
function parse_ml_result(trait::GainTrait, p::Vector{T}) where T <: AbstractFloat
    θ_sde_ref = create_Matern3HalfsKernel(T)
    return parse_ml_result_internal(θ_sde_ref, trait, p)
end

function parse_ml_result_internal(::Matern3HalfsKernel, ::UnityGain, p::Vector{T}) where T <: AbstractFloat
    return create_Matern3HalfsKernel(p[1], one(T)), p[2]
end

function parse_ml_result_internal(::Matern3HalfsKernel, ::InferGain, p::Vector{T}) where T <: AbstractFloat
    return create_Matern3HalfsKernel(p[1], p[3]), p[2]
end