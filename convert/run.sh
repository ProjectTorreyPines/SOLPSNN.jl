#!/usr/bin/env bash
# Orchestrate the full SOLPS-NN -> ONNX conversion pipeline.
#
#   1. download the requested quantities (SavedModels + sidecars) from SURFdrive
#   2. convert each fold SavedModel -> ONNX (opset 17)
#   3. parse the b2fgmtry geometry -> npy + json
#   4. build manifest.json for SOLPSNN.jl
#
# Usage:
#   ./run.sh [item ...]
#
# Env overrides:
#   CONVERT_ENV   conda env prefix (default: $PSCRATCH/.conda/envs/solpsnn-convert)
#   RAW_DIR       raw download dir  (default: $PSCRATCH/solps-nn-data)
#   OUT_DIR       ONNX output dir   (default: $PSCRATCH/solps-nn-onnx)
#   BASE_URL      re-hosting URL prefix recorded in manifest.json (default: empty)
#   VALIDATE      set to 1 to validate each ONNX fold against Keras
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERT_ENV="${CONVERT_ENV:-$PSCRATCH/.conda/envs/solpsnn-convert}"
RAW_DIR="${RAW_DIR:-$PSCRATCH/solps-nn-data}"
OUT_DIR="${OUT_DIR:-$PSCRATCH/solps-nn-onnx}"
BASE_URL="${BASE_URL:-}"

# Keep all caches/temp off $HOME (which is quota-limited on Perlmutter).
export TMPDIR="${TMPDIR:-$PSCRATCH/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PSCRATCH/.cache/pip}"
mkdir -p "$TMPDIR" "$RAW_DIR" "$OUT_DIR"

# Default headline set (matches ActorSOLPSNN defaults) if none given.
ITEMS=("$@")
if [ ${#ITEMS[@]} -eq 0 ]; then
    ITEMS=(te ti na1 pwmxap psol)
fi

RUN=(conda run --no-capture-output --prefix "$CONVERT_ENV" python)

echo "== [1/4] download =="
"${RUN[@]}" "$HERE/download_models.py" --items "${ITEMS[@]}" --raw-dir "$RAW_DIR"

echo "== [2/4] convert to ONNX =="
CONV_ARGS=(--items "${ITEMS[@]}" --raw-dir "$RAW_DIR" --out-dir "$OUT_DIR")
if [ "${VALIDATE:-0}" = "1" ]; then CONV_ARGS+=(--validate); fi
"${RUN[@]}" "$HERE/convert_to_onnx.py" "${CONV_ARGS[@]}"

echo "== [3/4] parse geometry =="
"${RUN[@]}" "$HERE/parse_geometry.py" \
    --b2fgmtry "$HERE/geometry_data/b2fgmtry" --out-dir "$OUT_DIR"

echo "== [4/4] build manifest =="
"${RUN[@]}" "$HERE/build_manifest.py" --out-dir "$OUT_DIR" --base-url "$BASE_URL"

echo
echo "Pipeline complete. Converted artifacts: $OUT_DIR"
echo "Point SOLPSNN.jl at it with: export FUSE_SOLPS_NN_DIR=$OUT_DIR"
