using Documenter
using MatrixEquationsAD

DocMeta.setdocmeta!(
    MatrixEquationsAD, :DocTestSetup,
    :(using MatrixEquationsAD); recursive = true,
)

makedocs(;
    modules = [MatrixEquationsAD],
    checkdocs = :none,
    authors = "Jesse Perla <jesseperla@gmail.com> and contributors",
    sitename = "MatrixEquationsAD.jl",
    format = Documenter.HTML(;
        canonical = "https://QuantEcon.github.io/MatrixEquationsAD.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Klein Policy Map" => "klein_map.md",
        "Discrete Lyapunov (Schur)" => "lyapd.md",
        "Kronecker Discrete Lyapunov" => "lyapdkr.md",
        "Generalised Sylvester" => "sylvester.md",
        "Algebraic Riccati (DARE)" => "ared.md",
    ],
)

deploydocs(;
    repo = "github.com/QuantEcon/MatrixEquationsAD.jl",
    devbranch = "main",
    push_preview = true,
)
