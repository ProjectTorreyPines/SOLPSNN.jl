# Third-party licenses

SOLPSNN.jl itself is licensed under the Apache License 2.0 (see `LICENSE`).
The files listed below are copied from, or derived from, the upstream
**SOLPS-NN** repository <https://github.com/sdasbach/solps-nn>, which is
distributed under the MIT License. The MIT copyright and permission notice
reproduced below applies to those portions.

| Path | Relationship to upstream |
|---|---|
| `convert/geometry_data/b2fgmtry` | verbatim copy of `solpsnn/geometry_data/b2fgmtry` (fixed SOLPS-ITER B2 grid) |
| `convert/config.json` | verbatim copy of `solpsnn/config.json` (SURFdrive download manifest: filenames, URLs, SHA-256) |
| `convert/parse_geometry.py` | `read_b25formfile` is a port of the function of the same name in `solpsnn/geometry.py`; `R_JET` is taken from `solpsnn.geometry.GeometryModel` |
| `convert/download_models.py` | re-implements the download logic of `solpsnn/download.py` |
| `src/model.jl` | `SPECIES_INDEX` reproduces the upstream `species_dict`; pre-processing follows `solpsnn.Model` |
| `src/geometry.jl` | `R_JET` and the `R / R_JET` rescaling rule follow the upstream `GeometryModel` |

The neural-network weights and sklearn sidecars are **not** part of this
repository. They are fetched from the upstream SURFdrive host at first use and
converted locally; their terms are those set by the upstream authors.

## SOLPS-NN — MIT License

```
MIT License

Copyright (c) 2024 Stefan Dasbach (Forschungszentrum Jülich)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The upstream authors ask that users of the model cite the publications listed
in the [Citation](README.md#citation) section of the README.
