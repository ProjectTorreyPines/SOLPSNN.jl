#!/usr/bin/env python3
"""
Build `manifest.json` describing the converted SOLPS-NN ONNX artifacts, so that
SOLPSNN.jl can fetch + SHA-256 verify them from a re-hosting location
(HuggingFace, SURFdrive, S3, ...), mirroring how the FUSE pedestal predictor
resolves and downloads its ONNX bundles.

Scans <out-dir> and records, per quantity item, every file's relative path,
size, and sha256. `base_url` (optional) is stored so the Julia side can build
download URLs as f"{base_url}/{path}". Files are grouped:
  root      -> X_mean.npy, X_std.npy
  geometry  -> geometry/{crx,cry,vol}.npy, geometry/geometry.json
  <item>    -> <item>/references.npy, quantiles.npy, fold{1..5}.onnx

Example:
  python build_manifest.py --out-dir $PSCRATCH/solps-nn-onnx \
      --base-url https://huggingface.co/ProjectTorreyPines/SOLPSNN/resolve/main
"""
import argparse
import hashlib
import json
import os
import time

CHUNK = 1 << 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def entry(out_dir, relpath):
    full = os.path.join(out_dir, relpath)
    return {"path": relpath.replace(os.sep, "/"),
            "size": os.path.getsize(full),
            "sha256": sha256(full)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(
        os.environ.get("PSCRATCH", os.getcwd()), "solps-nn-onnx"))
    ap.add_argument("--base-url", default="",
                    help="URL prefix where artifacts will be hosted")
    args = ap.parse_args()

    out = args.out_dir
    items = {}

    # root scaler files
    root = [f for f in ("X_mean.npy", "X_std.npy")
            if os.path.isfile(os.path.join(out, f))]
    if root:
        items["root"] = [entry(out, f) for f in root]

    # geometry
    geo_dir = os.path.join(out, "geometry")
    if os.path.isdir(geo_dir):
        geo_files = ["geometry/crx.npy", "geometry/cry.npy",
                     "geometry/vol.npy", "geometry/geometry.json"]
        items["geometry"] = [entry(out, f) for f in geo_files
                             if os.path.isfile(os.path.join(out, f))]

    # quantity items: any subdir containing fold*.onnx
    for name in sorted(os.listdir(out)):
        d = os.path.join(out, name)
        if not os.path.isdir(d) or name == "geometry":
            continue
        files = []
        for f in ("references.npy", "quantiles.npy"):
            if os.path.isfile(os.path.join(d, f)):
                files.append(f"{name}/{f}")
        for k in range(1, 6):
            f = f"{name}/fold{k}.onnx"
            if os.path.isfile(os.path.join(out, f)):
                files.append(f)
        if any(f.endswith(".onnx") for f in files):
            items[name] = [entry(out, f) for f in files]

    manifest = {
        "schema": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "opset": 17,
        "input_order": ["R", "B", "P", "Dpuff", "Npuff", "Dcore", "Dperp", "chi_perp"],
        "items": items,
    }
    dest = os.path.join(out, "manifest.json")
    with open(dest, "w") as f:
        json.dump(manifest, f, indent=2)

    n_files = sum(len(v) for v in items.values())
    total = sum(e["size"] for v in items.values() for e in v)
    print(f"manifest: {len(items)} groups, {n_files} files, "
          f"{total/1e6:.1f} MB -> {dest}")
    for k, v in items.items():
        print(f"  {k}: {len(v)} files")


if __name__ == "__main__":
    main()
