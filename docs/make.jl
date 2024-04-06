using Documenter
using MaternRegression

# # local.
makedocs(
    sitename = "MaternRegression",
    modules = [MaternRegression],
    #format = Documenter.HTML(),
    pages = [
        "Overview" => "index.md",
        "Public API" => "api.md",
        
        "Basic query" =>
        "generated/sde_gp_lit.md",

        "Hyperparameter optimization" =>
        "generated/optim_lit.md",

        "Hyperparameter inference" =>
        "generated/mcmc_lit.md",
    ],
)

# # github.
# makedocs(
#     sitename="MaternRegression.jl",
#     modules=[MaternRegression],
#     format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing)=="true"),
#     pages = [
#         "Overview" => "index.md",
#         "Public API" => "api.md",
        
#         "Basic query" =>
#         "generated/sde_gp_lit.md",

#         "Hyperparameter optimization" =>
#         "generated/optim_lit.md",

#         "Hyperparameter inference" =>
#         "generated/mcmc_lit.md",
#     ],
# )
# deploydocs(
#     repo = "github.com/RoyCCWang/MaternRegression.jl",
#     target = "build",
#     branch = "gh-pages",
#     versions = ["stable" => "v^", "v#.#" ],
# )