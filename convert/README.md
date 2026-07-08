# SOLPS-NN → ONNX conversion pipeline

Offline tooling that turns the upstream [SOLPS-NN](https://github.com/sdasbach/solps-nn)
TensorFlow/Keras SavedModels into ONNX artifacts consumed by `SOLPSNN.jl`.
This runs **once** in a Python + TensorFlow environment; it is **not** part of
the Julia package's runtime and is not on the Julia load path.

## Why

SOLPS-NN ships each predicted quantity as an ensemble of **5 Keras folds**
(prediction = mean of folds) in TensorFlow SavedModel format, plus sklearn
preprocessing sidecars. Julia/FUSE has no TensorFlow; the established pattern
(FUSE pedestal predictor, `TroyonBetaNN`) is ONNX + `ONNXRunTime.jl`. So we
convert the folds to ONNX and reimplement the pre/post-processing in Julia.

## Environment

Created on Perlmutter (kept off `$HOME`, which is quota-limited):

```bash
export CONDA_PKGS_DIRS=$PSCRATCH/.conda/pkgs
export PIP_CACHE_DIR=$PSCRATCH/.cache/pip
conda create -y --prefix $PSCRATCH/.conda/envs/solpsnn-convert python=3.10 pip
conda run --prefix $PSCRATCH/.conda/envs/solpsnn-convert \
    python -m pip install -r requirements.txt
```

Verified versions: TF-CPU 2.13.1, tf2onnx 1.16.1, onnx 1.14.1,
onnxruntime 1.16.3, keras 2.13.1, scikit-learn 1.7.2, numpy 1.24.3.

## Run

```bash
# headline set (te, ti, na1[=D+ density], pwmxap, psol):
./run.sh

# or an explicit subset / everything:
./run.sh te ti na1 ua1 pwmxap
VALIDATE=1 ./run.sh psol          # validate each ONNX fold vs Keras
./run.sh all                      # full set (>20 GB download)
```

Outputs (default `$PSCRATCH/solps-nn-onnx`):

```
X_mean.npy  X_std.npy                 # global input scaler
geometry/crx.npy cry.npy vol.npy geometry.json
<item>/references.npy quantiles.npy   # sklearn QuantileTransformer state
<item>/fold{1..5}.onnx                # converted ensemble
manifest.json                         # sizes + sha256 for SOLPSNN.jl
```

Then point the Julia package at it: `export FUSE_SOLPS_NN_DIR=$PSCRATCH/solps-nn-onnx`.

## Scripts

| Script | Purpose |
|---|---|
| `download_models.py` | Fetch SavedModels + sidecars from SURFdrive (SHA-256 verified, resumable, subset selection). |
| `convert_to_onnx.py` | `tf2onnx` each fold SavedModel → `foldN.onnx` (opset 17); copy sidecars; optional Keras-vs-ONNX validation. |
| `parse_geometry.py` | Parse `geometry_data/b2fgmtry` → `crx/cry/vol.npy` + `geometry.json`. |
| `build_manifest.py` | Record path/size/sha256 (+ optional `base_url`) for the Julia fetcher. |
| `run.sh` | Orchestrate all of the above. |

## Input parameter order

SOLPS-NN expects `[R, B, P, Dpuff, Npuff, Dcore, Dperp, chi_perp]`. Upstream
preprocessing (reproduced in Julia): `P/2`, `log10` of the three rates, then
standardize with `X_mean`/`X_std`. Post-processing is the sklearn
`QuantileTransformer` inverse (from `references.npy`/`quantiles.npy`).

## Notes

- `config.json` (vendored) is the upstream download manifest (URLs + hashes).
- On Perlmutter, `onnxruntime` prints harmless `pthread_setaffinity_np`
  warnings unless thread counts are pinned; `convert_to_onnx.py --validate`
  sets `intra_op_num_threads=1` to avoid them.
- SavedModels made with TF 2.3 load fine under TF 2.13 (plain Keras nets).
