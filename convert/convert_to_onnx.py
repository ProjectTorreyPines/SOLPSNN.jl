#!/usr/bin/env python3
"""
Convert downloaded SOLPS-NN TensorFlow SavedModels to ONNX (opset 17) and lay
out the artifacts consumed by SOLPSNN.jl.

Input layout (produced by download_models.py):
  <raw-dir>/
    X_mean.npy  X_std.npy
    <item>/references.npy  <item>/quantiles.npy
    <item>/fold{1..5}/saved_model.pb + variables/...

Output layout (consumed by SOLPSNN.jl):
  <out-dir>/
    X_mean.npy  X_std.npy
    <item>/references.npy  <item>/quantiles.npy
    <item>/fold{1..5}.onnx

Conversion uses `python -m tf2onnx.convert --saved-model ... --opset 17`,
the path validated on the solpsnn-convert env. Optionally validates each
ONNX file against the original Keras SavedModel with onnxruntime.

Example:
  python convert_to_onnx.py --items te ti na1 pwmxap psol \
      --raw-dir $PSCRATCH/solps-nn-data \
      --out-dir $PSCRATCH/solps-nn-onnx --validate
"""
import argparse
import os
import shutil
import subprocess
import sys

FOLDS = range(1, 6)
SIDECARS = ("references.npy", "quantiles.npy")


def convert_fold(saved_model_dir, onnx_path, opset):
    cmd = [sys.executable, "-m", "tf2onnx.convert",
           "--saved-model", saved_model_dir,
           "--output", onnx_path,
           "--opset", str(opset)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + "\n" + r.stderr[-4000:] + "\n")
        raise RuntimeError(f"tf2onnx failed for {saved_model_dir}")


def validate_fold(saved_model_dir, onnx_path, n_inputs=8, atol=1e-4):
    import numpy as np
    import onnxruntime as ort
    import tensorflow as tf

    x = np.random.randn(4, n_inputs).astype(np.float32)

    model = tf.keras.models.load_model(saved_model_dir, compile=False)
    y_tf = np.asarray(model.predict(x, verbose=0))

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # avoid Perlmutter pthread_setaffinity warnings
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    y_ort = np.asarray(sess.run(None, {iname: x})[0])

    if y_tf.shape != y_ort.shape:
        raise RuntimeError(f"shape mismatch {y_tf.shape} vs {y_ort.shape}")
    diff = float(np.max(np.abs(y_tf - y_ort)))
    rng = float(np.max(np.abs(y_tf))) or 1.0
    if diff > atol * max(1.0, rng):
        raise RuntimeError(f"validation diff too large: {diff} (range {rng})")
    return diff, iname


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", nargs="+", required=True,
                    help="quantity keys to convert (must be already downloaded)")
    ap.add_argument("--raw-dir", default=os.path.join(
        os.environ.get("PSCRATCH", os.getcwd()), "solps-nn-data"))
    ap.add_argument("--out-dir", default=os.path.join(
        os.environ.get("PSCRATCH", os.getcwd()), "solps-nn-onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--validate", action="store_true",
                    help="check each ONNX fold against Keras (needs TF)")
    ap.add_argument("--force", action="store_true",
                    help="reconvert even if the .onnx already exists")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Copy global input scaler files.
    for name in ("X_mean.npy", "X_std.npy"):
        src = os.path.join(args.raw_dir, name)
        if not os.path.isfile(src):
            sys.exit(f"missing {src}; run download_models.py first")
        shutil.copy2(src, os.path.join(args.out_dir, name))

    for item in args.items:
        raw_item = os.path.join(args.raw_dir, item)
        out_item = os.path.join(args.out_dir, item)
        if not os.path.isdir(raw_item):
            sys.exit(f"missing {raw_item}; run download_models.py first")
        os.makedirs(out_item, exist_ok=True)
        print(f"[{item}]")

        for name in SIDECARS:
            src = os.path.join(raw_item, name)
            if not os.path.isfile(src):
                sys.exit(f"missing sidecar {src}")
            shutil.copy2(src, os.path.join(out_item, name))

        for k in FOLDS:
            sm = os.path.join(raw_item, f"fold{k}")
            onnx_path = os.path.join(out_item, f"fold{k}.onnx")
            if not os.path.isdir(sm):
                sys.exit(f"missing SavedModel {sm}")
            if os.path.isfile(onnx_path) and not args.force:
                print(f"  fold{k}.onnx exists, skipping")
            else:
                print(f"  converting fold{k} -> {os.path.basename(onnx_path)}")
                convert_fold(sm, onnx_path, args.opset)
            if args.validate:
                diff, iname = validate_fold(sm, onnx_path)
                print(f"    validated (input '{iname}', max|diff|={diff:.2e})")

    print(f"\nDone. ONNX artifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
