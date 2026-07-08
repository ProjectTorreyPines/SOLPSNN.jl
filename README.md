# SOLPSNN.jl

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
ONNX offline by the tooling in [`convert/`](convert/README.md).

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

The ONNX weights (~1.3 GB for the headline set) live **outside git**. Produce
them once with the conversion pipeline, then point the package at them:

```bash
cd convert && ./run.sh            # downloads + converts te, ti, na1, pwmxap, psol
export FUSE_SOLPS_NN_DIR=$PSCRATCH/solps-nn-onnx
```

Resolution order for the artifact directory: explicit `dir=` argument →
`ENV["FUSE_SOLPS_NN_DIR"]` → `$PSCRATCH/solps-nn-onnx`. If `manifest.json`
carries a `base_url`, missing files are fetched and SHA-256 verified on demand.

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

The SOLPS-NN model and its training are the work of Dasbach et al. If you use
this wrapper, please cite:

> S. Dasbach et al., *Towards fast surrogate models for interpolation of tokamak
> edge plasmas*, [arXiv:2604.19223](https://arxiv.org/abs/2604.19223).

Related: Dasbach, S. & Wiesen, S. (2023), Nucl. Mater. Energy 34, 101396;
Wiesen, S. et al. (2024), Nucl. Fusion 64, 086046. Upstream model:
<https://github.com/sdasbach/solps-nn>.
