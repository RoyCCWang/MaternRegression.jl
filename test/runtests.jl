
import MaternRegression as GS
import RKHSRegularization as RK

import Random
Random.seed!(25)

using LinearAlgebra
using Test

include("../examples/helpers/gpr.jl")
include("../examples/helpers/utils.jl")


@testset "1D Matern ν = 3/2" begin

    N_data_scenarios = 100
    N_param_scenarios = 10
    Ts = [BigFloat; Float64]
    
    N_ub = 17
    Nq = 100

    for T in Ts

        #rel_tol = eps(T)*1000 # too stringent.
        rel_tol = convert(T, 1e-7)

        for _ = 1:N_data_scenarios

            # training set preparation.
            N_on_grid = rand(2:N_ub) # need at least two points for LinRange.
            N_randomly_generated = rand(1:N_ub)
            ub = abs(randn(T))
            lb = -ub
            ts = generate_almost_grid(N_on_grid, N_randomly_generated, lb, ub)

            f = xx->sinc(4*xx)*xx^3 # oracle function.
            y = f.(ts)
            
            θ_dummy = GS.create_Matern3HalfsKernel(T)
            ML_buffer = GS.setup_ml(θ_dummy, ts, y)

            for _ = 1:N_param_scenarios
                # generate kernel.
                λ = abs(randn(T))
                b = abs(randn(T))
                σ² = rand(T)/100

                θ = RK.Matern3Halfs(λ, b)
                θ_sde = GS.create_Matern3HalfsKernel(λ, b)

                # query positions.
                tqs = LinRange(lb, ub, Nq)
                
                # conventional GPR
                mqs, vqs, η = querygpr(θ, tqs, ts, y, σ²)

                # SDE GPR
                sde_gp = GS.create_sdegp(θ_sde, ts, y, σ²)
                mqs2, vqs2 = GS.query_sdegp(T, sde_gp, tqs, ts)
                
                @test norm(mqs - mqs2)/norm(mqs2) < rel_tol
                @test norm(vqs - vqs2)/norm(vqs2) < rel_tol

                # # marginal likelihood, without multiplicative and additive constants.
                
                # reference.
                ML_gpr = evalnlli_gpr(η, y)

                # sde.
                ML_sde = GS.eval_ml!(ML_buffer, θ_sde, ts, y, σ²)
                
                @test abs(ML_sde - ML_gpr)/abs(ML_gpr) < rel_tol
            end
        end
    end
end