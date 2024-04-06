using Literate

include("helpers/gen_utils.jl")

# `dest_dir` is where the generated files will end up in. We delete all the files in that directory first.
dest_dir = "../docs/src/generated"
reset_dir(dest_dir) # creates the path if it doesn't exist.

# # Basic query given hyperparameters
# fix the URL. This is generated because we're using Documenter.jl-flavoured Markdown.
postprocfunc = xx->replace(
    xx,
    "EditURL = \"sde_gp.jl\"" =>
    "EditURL = \"../../../examples/sde_gp.jl\"" # as ifthe pwd() is in the `dest_dir`
)

Literate.markdown(
    "sde_gp.jl";
    execute = true,
    name = "sde_gp_lit", # make this different ahn "bernstein_filter.jl" so it is easier to find and delete all generated files.
    postprocess = postprocfunc,
)

move_prefix_name = "sde_gp_lit"
move_md(dest_dir, move_prefix_name)


# # Hyperparameter optimization
postprocfunc = xx->replace(
    xx,
    "EditURL = \"optim.jl\"" =>
    "EditURL = \"../../../examples/optim.jl\"" # as ifthe pwd() is in the `dest_dir`
)

Literate.markdown(
    "optim.jl";
    execute = true,
    name = "optim_lit", # make this different ahn "bernstein_filter.jl" so it is easier to find and delete all generated files.
    postprocess = postprocfunc,
)

move_prefix_name = "optim_lit"
move_md(dest_dir, move_prefix_name)

# # Hyperparameter inference
postprocfunc = xx->replace(
    xx,
    "EditURL = \"mcmc.jl\"" =>
    "EditURL = \"../../../examples/mcmc.jl\"" # as ifthe pwd() is in the `dest_dir`
)

Literate.markdown(
    "mcmc.jl";
    execute = true,
    name = "mcmc_lit", # make this different ahn "bernstein_filter.jl" so it is easier to find and delete all generated files.
    postprocess = postprocfunc,
)

move_prefix_name = "mcmc_lit"
move_md(dest_dir, move_prefix_name)

nothing

