"""
    SOLPSNN

Julia inference wrapper for the SOLPS-NN edge-plasma surrogate
(Dasbach & Wiesen), consuming ONNX artifacts produced by the offline
`convert/` pipeline. Reproduces the upstream pre-processing, 5-fold ensemble
averaging, and sklearn QuantileTransformer inverse in pure Julia.

Quick start:

    ENV["FUSE_SOLPS_NN_DIR"] = "/path/to/solps-nn-onnx"   # or \$PSCRATCH default
    te  = SOLPSNN.load_model("te")
    ne  = SOLPSNN.load_model("na"; species="D1")
    q   = SOLPSNN.load_model("pwmxap")

    X = [6.2 5.3 1e8 1e22 1e20 9.1e21 0.3 1.0]            # (1,8), ITER-like
    field  = SOLPSNN.predict(te, X)                       # (1, 104, 50)
    qpeak  = SOLPSNN.predict(q, X)                         # (1,)

Input order: `[R, B, P, Dpuff, Npuff, Dcore, Dperp, chi_perp]`.
"""
module SOLPSNN

import ONNXRunTime as ORT
import NPZ
import JSON
import Downloads
import SHA
import IMASdd
using SpecialFunctions: erfc

export SOLPSModel, load_model, predict, load_geometry, scaled_geometry
export build_edge_profiles_ggd!, add_ggd_field!, ggd_time_slice!
export build_edge_profiles_ggd_solps2imas!

include("quantile.jl")
include("geometry.jl")
include("download.jl")
include("model.jl")
include("ggd.jl")

"""
    build_edge_profiles_ggd_solps2imas!(dd, b2gmtry; R, R_JET, time0)

Build the `dd.edge_profiles.grid_ggd` from a SOLPS `b2fgmtry` file using
`SOLPS2imas.solps2imas`, so the surrogate's mesh (and its full set of physical
grid subsets: separatrix, targets, OMP/IMP, core/SOL, …) matches exactly what a
real SOLPS run loaded via SOLPS2imas produces. Optionally rescales the grid to
major radius `R` (lengths ∝ R/R_JET). Returns the `grid_index` to reference from
field entries; write those fields with `add_ggd_field!(...; order=:solps)`.

This is provided by the `SOLPS2imasExt` package extension and requires
`SOLPS2imas` to be loaded alongside `SOLPSNN`.
"""
function build_edge_profiles_ggd_solps2imas! end

end # module SOLPSNN
