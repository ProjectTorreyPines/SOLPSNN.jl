# SOLPSNN.jl

![Format Check](https://github.com/ProjectTorreyPines/SOLPSNN.jl/actions/workflows/format_check.yml/badge.svg)
![Tests](https://github.com/ProjectTorreyPines/SOLPSNN.jl/actions/workflows/test.yml/badge.svg)

Julia inference wrapper for the **SOLPS-NN** tokamak edge/SOL plasma surrogate
of [Dasbach & Wiesen](https://github.com/sdasbach/solps-nn), for use in FUSE.

SOLPS-NN maps 8 scalar inputs to steady-state SOLPS-ITER edge quantities:
either 2D fields on a fixed `104x50` B2 grid (`te`, `ti`, species densities
`na`, parallel velocities `ua`) or scalars (`pwmxap` peak outer-target heat
flux, `fnixap` integrated ion flux, `psol` power across the separatrix). Each
quantity is an **ensemble of 5 folds** (prediction = mean).

This package runs the networks via [`ONNXRunTime.jl`](https://github.com/jw3126/ONNXRunTime.jl)
and reproduces the upstream pre-processing and sklearn `QuantileTransformer`
inverse in pure Julia. The original TensorFlow/Keras weights are converted to
ONNX by the tooling in [`convert/`](convert/README.md), run automatically on
first use (see [Model artifacts](#model-artifacts)).

## Inputs

`predict` takes an `(N, 8)` matrix (or a length-8 vector) in this order and in
SI-ish units:

| # | Symbol | Meaning | Unit |
|---|--------|---------|------|
| 1 | `R`        | major radius | m |
| 2 | `B`        | toroidal field on axis | T |
| 3 | `P`        | input power into domain | W |
| 4 | `Dpuff`    | deuterium gas puff rate | atoms/s |
| 5 | `Npuff`    | nitrogen gas puff rate | atoms/s |
| 6 | `Dcore`    | deuterium core fueling rate | atoms/s |
| 7 | `Dperp`    | cross-field particle transport coeff | m²/s |
| 8 | `chi_perp` | cross-field heat transport coeff | m²/s |

Trained ranges (extrapolation not recommended): `R` 1–10 m, `B` 1–10 T,
`P` 10–200 MW, `Dpuff` 1e18–1e24, `Npuff` 1e18–1e23, `Dcore` 1e19–1e24,
`Dperp`/`chi_perp` 0.1–2 m²/s.

## Model artifacts

The ONNX weights (~1.3 GB for the headline set) live **outside git**. Upstream
ships only TensorFlow SavedModels (on SURFdrive), which are the single ground
truth — there is no ONNX mirror. The first `load_model(...)` for a quantity
**produces the ONNX automatically**: it runs the [`convert/`](convert/README.md)
pipeline, which fetches that quantity's weights from SURFdrive and converts them
to ONNX (`tf2onnx`) in a conda env created on demand. This needs `conda` on
`PATH` (e.g. `module load conda` on NERSC, or the FUSE conda env). A new upstream
TensorFlow release is picked up by re-converting (delete the cache dir or call
`convert_models!(...)`).

```julia
using SOLPSNN
convert_models!(["all"])         # optional: pre-build every quantity up front
```

Or run the pipeline directly and point the package at the output:

```bash
cd convert && ./run.sh           # downloads + converts te, ti, na1, pwmxap, psol
export FUSE_SOLPS_NN_DIR=$PSCRATCH/solps-nn-onnx
```

Resolution order for the artifact directory: explicit `dir=` argument →
`ENV["FUSE_SOLPS_NN_DIR"]` → `$PSCRATCH/solps-nn-onnx` → a `Scratch.jl` scratch
space under the Julia depot (always writable, so `$PSCRATCH` is optional).
Disable auto-conversion with `ENV["FUSE_SOLPS_NN_AUTOCONVERT"] = "0"` to fail
fast on missing artifacts instead.

## Usage

```julia
using SOLPSNN

te = load_model("te")                    # 2D electron temperature [eV]
ne = load_model("na"; species="D1")      # 2D D+ density [m^-3]
q  = load_model("pwmxap")                # peak outer-target heat flux [W/m^2]

X = [6.2 5.3 1e8 1e22 1e20 9.1e21 0.3 1.0]   # (1,8) ITER-like
Te_field = predict(te, X)                # (1, 104, 50)
qpeak    = predict(q, X)                 # (1,)

# single-sample convenience (drops the batch dimension)
predict(q,  vec(X))                      # scalar
predict(te, vec(X))                      # (104, 50)
```

Geometry (for mapping fields to real space / IMAS GGD):

```julia
geo = load_geometry(SOLPSNN.resolve_dir())   # crx, cry, vol, cuts, R_JET
g   = scaled_geometry(geo, 6.2)               # rescale to R = 6.2 m
```

Write 2D fields into an IMAS `edge_profiles` GGD:

```julia
using IMAS
dd = IMAS.dd()
gi = build_edge_profiles_ggd!(dd, geo; R=6.2, time0=0.0)   # builds nodes+cells grid
ep = ggd_time_slice!(dd, 0.0)
add_ggd_field!(ep.electrons.temperature, predict(te, vec(X)); grid_index=gi)
add_ggd_field!(ep.electrons.density,     predict(ne, vec(X)); grid_index=gi)
```

The grid uses the standard GGD "cells" subset (`identifier.index = 5`,
`dimension = 3`); field values are flattened in cell order `c=(ix-1)*(ny+2)+iy`.

### SOLPS2imas-backed grid

For interoperability with [`GGDUtils`](https://github.com/ProjectTorreyPines/GGDUtils.jl)
/ [`SOLPS2ctrl`](https://github.com/ProjectTorreyPines/SOLPS2ctrl.jl) tooling,
the grid can instead be built through [`SOLPS2imas.jl`](https://github.com/ProjectTorreyPines/SOLPS2imas.jl),
reproducing the exact mesh and full set of physical grid subsets (separatrix,
inner/outer target, OMP/IMP, core/SOL, …) of a real SOLPS run. This is provided
by the `SOLPS2imasExt` package extension, active once `SOLPS2imas` is loaded:

```julia
using SOLPSNN
import SOLPS2imas                        # activates SOLPS2imasExt

dd = IMASdd.dd()
gi = build_edge_profiles_ggd_solps2imas!(dd, bundled_b2fgmtry(); R=6.2, R_JET=R_JET, time0=0.0)
ep = ggd_time_slice!(dd, 0.0)
add_ggd_field!(ep.electrons.temperature, predict(te, X); grid_index=gi, order=:solps)
```

`bundled_b2fgmtry()` returns the `b2fgmtry` grid file shipped inside the package,
so no external geometry files are needed. Note the SOLPS2imas grid uses the
transposed cell order, hence `order=:solps` when writing fields.

## Testing

```bash
export FUSE_SOLPS_NN_DIR=$PSCRATCH/solps-nn-onnx
julia --project=. -e 'using Pkg; Pkg.test()'
```

Parity tests compare `predict` against reference outputs produced by the real
upstream sklearn post-processing (`test/gen_reference.py`); unit tests cover the
quantile inverse, normal CDF, geometry scaling, and item routing. Model tests
skip gracefully if the artifacts are absent.

## Citation

Users of the model are kindly asked to cite all of the following publications:

- Dasbach, S., & Wiesen, S. (2023). Towards fast surrogate models for
  interpolation of tokamak edge plasmas. Nuclear Materials and Energy, 34,
  101396. <https://doi.org/10.1016/j.nme.2023.101396>
- Wiesen, S., et al. (2024). Data-driven models in fusion exhaust: AI methods
  and perspectives. Nuclear Fusion, 64(8), 086046.
  <https://doi.org/10.1088/1741-4326/ad5a1d>
- S. Dasbach et al., Towards fast surrogate models for interpolation of tokamak
  edge plasmas, [arXiv:2604.19223](https://arxiv.org/abs/2604.19223).

Upstream model: <https://github.com/sdasbach/solps-nn>.
