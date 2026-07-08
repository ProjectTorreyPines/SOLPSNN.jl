#!/usr/bin/env python3
"""
Parse the SOLPS-ITER `b2fgmtry` B2 grid file into artifacts that SOLPSNN.jl can
load directly (via NPZ.jl + JSON), so no b2 parser is needed on the Julia side.

Reproduces the parsing in the upstream `solpsnn/geometry.py`:
  * nx, ny                       grid dimensions (interior cells)
  * crx, cry  (nx+2, ny+2, 4)    R,Z of the 4 corners of every cell (incl guard)
  * vol       (nx+2, ny+2)       cell volumes
  * leftcut, rightcut, topcut    poloidal/radial cut indices (magnetic topology)

The B2 arrays in the file are Fortran-ordered; we reshape with order='F' and
store C-contiguous so NPZ.jl reads the same logical (nx+2, ny+2, 4) shape.

R_JET is the reference major radius used by SOLPS-NN to rescale geometry with
`R/R_JET` (constant taken verbatim from the upstream GeometryModel).

Output:
  <out-dir>/geometry/crx.npy  cry.npy  vol.npy  geometry.json

Example:
  python parse_geometry.py --b2fgmtry geometry_data/b2fgmtry \
      --out-dir $PSCRATCH/solps-nn-onnx
"""
import argparse
import json
import os

import numpy as np

# Reference major radius from upstream solpsnn.geometry.GeometryModel
R_JET = 3.000727179161820


def read_b25formfile(file_path, field):
    """Read one named block from a b2 *cf-formatted file (port of upstream)."""
    found = False
    data = []
    n = None
    t = None
    with open(file_path, "r", buffering=100000, encoding="UTF-8") as src:
        for line in src:
            if n is not None and len(data) == n:
                break
            if len(line) == 0:
                break
            if field in line and "*cf:" in line and not found:
                s = line.split()
                t = s[1]
                n = int(s[2])
                found = True
            elif found:
                if t == "char":
                    data = str(line[0:n])
                    break
                elif t == "int":
                    data.extend(int(v) for v in line.split())
                elif t == "real":
                    data.extend(float(v) for v in line.split())
    return np.array(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.realpath(__file__))
    ap.add_argument("--b2fgmtry",
                    default=os.path.join(here, "geometry_data", "b2fgmtry"))
    ap.add_argument("--out-dir", default=os.path.join(
        os.environ.get("PSCRATCH", os.getcwd()), "solps-nn-onnx"))
    args = ap.parse_args()

    f = args.b2fgmtry
    nx, ny = (int(v) for v in read_b25formfile(f, " nx,ny "))
    crx = np.ascontiguousarray(
        read_b25formfile(f, " crx ").reshape((nx + 2, ny + 2, 4), order="F"))
    cry = np.ascontiguousarray(
        read_b25formfile(f, " cry ").reshape((nx + 2, ny + 2, 4), order="F"))
    vol = np.ascontiguousarray(
        read_b25formfile(f, " vol ").reshape((nx + 2, ny + 2), order="F"))
    rightcut = int(read_b25formfile(f, " rightcut ")[0])
    leftcut = int(read_b25formfile(f, " leftcut ")[0])
    topcut = int(read_b25formfile(f, " topcut ")[0])

    geo_dir = os.path.join(args.out_dir, "geometry")
    os.makedirs(geo_dir, exist_ok=True)
    np.save(os.path.join(geo_dir, "crx.npy"), crx.astype(np.float64))
    np.save(os.path.join(geo_dir, "cry.npy"), cry.astype(np.float64))
    np.save(os.path.join(geo_dir, "vol.npy"), vol.astype(np.float64))

    meta = {
        "nx": nx, "ny": ny,
        "leftcut": leftcut, "rightcut": rightcut, "topcut": topcut,
        "R_JET": R_JET,
        "crx_shape": list(crx.shape),
        "cry_shape": list(cry.shape),
        "vol_shape": list(vol.shape),
        "note": "crx/cry indexed [ix, iy, corner]; corners are the 4 cell "
                "corners in B2 ordering. Scale geometry by R/R_JET.",
    }
    with open(os.path.join(geo_dir, "geometry.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"nx={nx} ny={ny} grid=({nx+2}x{ny+2}) "
          f"cuts(left={leftcut},right={rightcut},top={topcut})")
    print(f"crx {crx.shape}  cry {cry.shape}  vol {vol.shape}")
    print(f"wrote geometry artifacts to {geo_dir}")


if __name__ == "__main__":
    main()
