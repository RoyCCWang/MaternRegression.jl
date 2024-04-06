
struct ECAConfig{T <: AbstractFloat}
    f_calls_limit::Int
    initial_iterate::Matrix{T} # empty for no initial guess
end

function ECAConfig(::Type{T}, max_fcalls::Int)::ECAConfig{T} where T <: AbstractFloat
    return ECAConfig(max_fcalls, Vector{T}(undef, 0))
end

function ECAConfig(max_fcalls::Int, x0::Vector{T})::ECAConfig{T} where T <: AbstractFloat
    mat = Matrix{T}(undef, 1, length(x0))
    mat[1,:] = x0
    return ECAConfig(max_fcalls, mat)
end

function ECAConfig(max_fcalls::Int, x0s::Vector{Vector{T}})::ECAConfig{T} where T <: AbstractFloat
    
    @assert !isempty(x0s)
    D = length(x0s[begin])
    
    mat = Matrix{T}(undef, length(x0s), D)
    for i in axes(mat,1)
        mat[i, :] = x0s[i]
    end

    return ECAConfig(max_fcalls, mat)
end

# result = runevofull(costfunc, lb, ub, f_calls_limit, x0)
function runevofull(
    costfunc, lb::Vector{T}, ub::Vector{T}, config,
    ) where T

    f_calls_limit, x0 = config.f_calls_limit, config.initial_iterate

    bounds = EVO.boxconstraints(lb = lb, ub = ub)
    algo  = EVO.ECA(
        N = 61,
        options = EVO.Options(
            f_calls_limit = f_calls_limit,
            seed = 1,
        ),
    )

    if !isempty(x0)
        EVO.set_user_solutions!(algo, x0, costfunc);
    end

    result = EVO.optimize(costfunc, bounds, algo)
    costfunc(result.best_sol.x) # force update of mgp with best solution.

    return result
end

# minimalist (and perhaps type-stable) return of best solution vector only, instead of custom result container.
function runevosimple(costfunc, lb::Vector{T}, ub::Vector{T}, config)::Vector{T} where T

    result = runevofull(costfunc, lb, ub, config)
    return result.best_sol.x
end

function convert2interval(x::T, c::T, d::T)

    # real number line to (-1, 1)
    x/sqrt(1+x^2)


end

#converts compact domain x ∈ [a,b] to compact domain out ∈ [c,d].
function convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real

    return (x-a)*(d-c)/(b-a)+c
end

function convertcompactdomain(x::Vector{T}, a::Vector{T}, b::Vector{T}, c::Vector{T}, d::Vector{T})::Vector{T} where T <: Real

    return collect( convertcompactdomain(x[i], a[i], b[i], c[i], d[i]) for i = 1:length(x) )
end