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
import IMAS
using SpecialFunctions: erfc

export SOLPSModel, load_model, predict, load_geometry, scaled_geometry
export build_edge_profiles_ggd!, add_ggd_field!, ggd_time_slice!

include("quantile.jl")
include("geometry.jl")
include("download.jl")
include("model.jl")
include("ggd.jl")

end # module SOLPSNN
