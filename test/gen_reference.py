#!/usr/bin/env python3
"""
Generate reference outputs for the SOLPSNN.jl parity tests.

Faithfully reproduces the upstream `solpsnn.Model.predict` post-processing
(sklearn QuantileTransformer inverse, C-order flatten for 2D) on top of the
converted ONNX folds, and dumps inputs + reference outputs to test/data/ for
the Julia test suite to compare against.

Run in the solpsnn-convert conda env with FUSE_SOLPS_NN_DIR (or $PSCRATCH default):
  conda run --prefix $PSCRATCH/.conda/envs/solpsnn-convert \
      python test/gen_reference.py
"""
import json
import os

import numpy as np
import onnxruntime as ort
from sklearn.preprocessing import QuantileTransformer

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, "data")
D = os.environ.get("FUSE_SOLPS_NN_DIR",
                   os.path.join(os.environ.get("PSCRATCH", ""), "solps-nn-onnx"))

SCALARS = {"pwmxap", "fnixap", "psol"}

# ITER-like test points [R, B, P, Dpuff, Npuff, Dcore, Dperp, chi] (see example.py)
X = np.array([
    [6.2, 5.3, 1.0e8, 1.0e22, 1.0e20, 9.1e21, 0.3, 1.0],
    [6.2, 5.3, 1.0e8, 1.0e23, 1.0e20, 9.1e21, 0.3, 1.0],
    [5.0, 4.0, 5.0e7, 5.0e21, 5.0e19, 3.0e21, 0.5, 0.8],
], dtype=np.float64)


def preprocess(X):
    Y = X.copy()
    Y[:, 2] = X[:, 2] / 2.0
    Y[:, 3] = np.log10(X[:, 3])
    Y[:, 4] = np.log10(X[:, 4])
    Y[:, 5] = np.log10(X[:, 5])
    return Y


def make_qt(item_dir):
    qt = QuantileTransformer(n_quantiles=100, output_distribution="normal")
    qt.references_ = np.load(os.path.join(item_dir, "references.npy"))
    qt.quantiles_ = np.load(os.path.join(item_dir, "quantiles.npy"))
    qt.n_quantiles_ = len(qt.references_)
    qt.n_features_in_ = qt.quantiles_.shape[1]
    return qt


def predict(item):
    idir = os.path.join(D, item)
    Xmean = np.load(os.path.join(D, "X_mean.npy"))
    Xstd = np.load(os.path.join(D, "X_std.npy"))
    Xn = ((preprocess(X) - Xmean) / Xstd).astype(np.float32)

    qt = make_qt(idir)
    scalar = item in SCALARS

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    per_fold = []
    for k in range(1, 6):
        s = ort.InferenceSession(os.path.join(idir, f"fold{k}.onnx"),
                                 sess_options=so, providers=["CPUExecutionProvider"])
        iname = s.get_inputs()[0].name
        oname = s.get_outputs()[0].name
        y = np.asarray(s.run([oname], {iname: Xn})[0])  # (N,1) or (N,104,50)
        if scalar:
            inv = qt.inverse_transform(y)                # (N,1)
        else:
            shp = y.shape
            y2 = y.reshape(shp[0], shp[1] * shp[2])      # C-order flatten
            inv = qt.inverse_transform(y2).reshape(shp)  # (N,104,50)
        per_fold.append(inv)
    out = np.mean(per_fold, axis=0)
    return out[:, 0] if scalar else out


def main():
    os.makedirs(DATA, exist_ok=True)
    np.save(os.path.join(DATA, "X_inputs.npy"), X)
    items = [it for it in ("te", "ti", "na1", "pwmxap", "psol")
             if os.path.isdir(os.path.join(D, it))]
    for it in items:
        out = predict(it)
        np.save(os.path.join(DATA, f"ref_{it}.npy"), out)
        print(f"{it}: ref shape {out.shape} range [{out.min():.4g}, {out.max():.4g}]")
    print(f"wrote references for {items} to {DATA}")


if __name__ == "__main__":
    main()
