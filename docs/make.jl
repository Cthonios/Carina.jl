using Documenter

makedocs(;
    sitename="Carina",
    authors="Carina contributors",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true"),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Running Carina" => "running.md",
        "Features" => "features.md",
        "Testing" => "testing.md",
        "Examples" => "examples.md",
        "Troubleshooting" => "troubleshooting.md",
        "Input File Reference" => [
            "Overview" => "reference/index.md",
            "Mesh and output files" => "reference/mesh-and-io.md",
            "Model" => "reference/model.md",
            "Materials" => "reference/materials.md",
            "Time integrators" => "reference/time-integrators.md",
            "Solvers" => "reference/solvers.md",
            "Boundary conditions" => "reference/boundary-conditions.md",
            "Initial conditions" => "reference/initial-conditions.md",
            "Quadrature" => "reference/quadrature.md",
            "Output fields" => "reference/output.md",
            "Function expressions" => "reference/functions.md",
        ],
    ],
    # The reference cross-links with relative .md links; that is intentional and
    # portable, so do not fail the build on the checks that dislike them.
    warnonly=[:cross_references, :missing_docs],
)

deploydocs(;
    repo="github.com/Cthonios/Carina.jl.git",
    devbranch="main",
    push_preview=true,
)
