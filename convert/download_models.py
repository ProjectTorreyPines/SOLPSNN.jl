#!/usr/bin/env python3
"""
Download SOLPS-NN TensorFlow SavedModels + preprocessing sidecars from the
upstream SURFdrive host, with SHA-256 verification.

This mirrors the logic of the upstream `solpsnn/download.py`, but:
  * lets you select a subset of quantities (default: the FUSE "headline" set),
  * verifies SHA-256 and skips files that are already present and correct,
  * downloads into a flat `<raw-dir>/<item>/...` layout that
    `convert_to_onnx.py` then consumes.

The upstream `config.json` (vendored next to this script) provides, for each
quantity key, a list of {filename, url, hash} entries. The special key "root"
holds the global input scaler files `X_mean.npy` / `X_std.npy`.

Item naming (matches upstream `species_dict`):
  te, ti                      -> 2D fields
  na{i}, ua{i}  (i=0..9)      -> 2D fields per species
                                 D0=0 D1=1 N0=2 N1=3 ... N7=9
  pwmxap, fnixap, psol        -> scalars

Example:
  python download_models.py --items te ti na1 pwmxap psol \
      --raw-dir $PSCRATCH/solps-nn-data
"""
import argparse
import hashlib
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("requests is required: pip install requests (or use the "
             "solpsnn-convert conda env)")

HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config.json")

# Default "headline" set wired into ActorSOLPSNN: electron/ion temperature,
# main-ion (D+) density, peak outer-target heat flux, power across separatrix.
DEFAULT_ITEMS = ["te", "ti", "na1", "pwmxap", "psol"]

CHUNK = 1 << 20  # 1 MiB


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected):
    return os.path.isfile(path) and sha256(path) == expected


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f}{unit}"
        n /= 1024


def download_one(url, dest, expected_hash, retries=3):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                t0 = time.time()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if chunk:
                            f.write(chunk)
                            done += len(chunk)
            if expected_hash and sha256(tmp) != expected_hash:
                raise IOError("hash mismatch after download")
            os.replace(tmp, dest)
            dt = time.time() - t0
            print(f"    downloaded {human(done)} in {dt:.1f}s")
            return
        except Exception as e:  # noqa: BLE001
            print(f"    attempt {attempt}/{retries} failed: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="path to upstream config.json (default: vendored copy)")
    ap.add_argument("--items", nargs="+", default=DEFAULT_ITEMS,
                    help=f"quantity keys to download (default: {DEFAULT_ITEMS}). "
                         "Use 'all' for the complete set.")
    ap.add_argument("--raw-dir", default=os.path.join(
        os.environ.get("PSCRATCH", os.getcwd()), "solps-nn-data"),
        help="destination directory for raw SavedModels + sidecars")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.items == ["all"]:
        items = [k for k in config if k != "root"]
    else:
        items = args.items

    unknown = [it for it in items if it not in config]
    if unknown:
        sys.exit(f"unknown item(s): {unknown}\navailable: "
                 f"{sorted(k for k in config if k != 'root')}")

    # Always include the global scaler files ("root").
    groups = ["root"] + items
    print(f"raw-dir : {args.raw_dir}")
    print(f"items   : {items}")

    n_ok = n_dl = 0
    for group in groups:
        print(f"[{group}]")
        for info in config[group]:
            # normalize "./X_mean.npy" -> "X_mean.npy"
            rel = info["filename"].lstrip("./")
            dest = os.path.join(args.raw_dir, rel)
            if verify(dest, info["hash"]):
                n_ok += 1
                continue
            print(f"  {rel}")
            download_one(info["url"], dest, info["hash"])
            n_dl += 1
    print(f"\nDone. {n_dl} downloaded, {n_ok} already present/verified.")


if __name__ == "__main__":
    main()
