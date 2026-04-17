using Documenter, FiniteStatePopulationDynamics

makedocs(;
    modules = [FiniteStatePopulationDynamics],
    warnonly = true,
    authors = "Simon Frost",
    sitename = "FiniteStatePopulationDynamics.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ecorecipes.github.io/FiniteStatePopulationDynamics.jl",
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/ecorecipes/FiniteStatePopulationDynamics.jl.git",
)
