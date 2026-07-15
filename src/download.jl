#= =============================================================== =#
#  Locating and (automatically) providing the converted ONNX artifacts
#
#  Upstream SOLPS-NN (github.com/sdasbach/solps-nn) distributes only
#  *TensorFlow SavedModels* on SURFdrive (see convert/config.json). SOLPSNN.jl
#  runs on ONNX (ONNXRunTime.jl), so the weights must first be converted by the
#  offline `convert/` pipeline (TF + tf2onnx, in a conda env). Those upstream
#  TensorFlow weights are the single ground truth — there is no ONNX mirror, so
#  a new TF release is picked up simply by re-running the conversion.
#
#  Directory resolution order (always returns a path; population is on demand):
#    1. explicit `dir` argument
#    2. ENV["FUSE_SOLPS_NN_DIR"]
#    3. $PSCRATCH/solps-nn-onnx      (Perlmutter default of convert/run.sh)
#    4. a Scratch.jl scratch space under the Julia depot (always writable; when
#       JULIA_DEPOT_PATH points at $PSCRATCH it still lands on scratch)
#
#  When a requested quantity is missing under `dir`, [`ensure_available`](@ref)
#  runs the `convert/` pipeline locally (fetching the TF weights from SURFdrive
#  and converting them). See `convert.jl`.
#= =============================================================== =#

const ENV_DIR = "FUSE_SOLPS_NN_DIR"

"""
    resolve_dir(; dir=nothing)

Return the directory holding the converted SOLPS-NN ONNX artifacts. Always
returns a path (creating/populating it is handled by [`ensure_available`](@ref)):
an explicit `dir`, then `ENV["$ENV_DIR"]`, then `\$PSCRATCH/solps-nn-onnx`, then a
[`Scratch.jl`](https://github.com/JuliaPackaging/Scratch.jl) scratch space under
the Julia depot (always writable; lands on `\$PSCRATCH` when `JULIA_DEPOT_PATH`
is configured there).
"""
function resolve_dir(; dir::Union{Nothing,AbstractString}=nothing)
    dir !== nothing && return String(dir)
    haskey(ENV, ENV_DIR) && return ENV[ENV_DIR]
    if haskey(ENV, "PSCRATCH")
        return joinpath(ENV["PSCRATCH"], "solps-nn-onnx")
    end
    return Scratch.get_scratch!(SOLPSNN, "solps-nn-onnx")
end

# ── Expected on-disk layout (matches convert/run.sh output) ─────────────────

_root_files() = ["X_mean.npy", "X_std.npy"]
_geometry_files() = ["geometry/crx.npy", "geometry/cry.npy", "geometry/vol.npy", "geometry/geometry.json"]
_item_files(key) = String["$key/references.npy", "$key/quantiles.npy",
    ("$key/fold$k.onnx" for k in 1:N_FOLDS)...]

"""Relative paths expected for manifest group `g` (`root`, `geometry`, or an item key)."""
function group_files(g::AbstractString)
    g == "root" && return _root_files()
    g == "geometry" && return _geometry_files()
    return _item_files(g)
end

"""True iff every file of group `g` is present under `dir`."""
group_complete(dir::AbstractString, g::AbstractString) =
    all(f -> isfile(joinpath(dir, f)), group_files(g))

# ── Integrity ────────────────────────────────────────────────────────────────

"""sha256 hex digest of a file."""
_sha256(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))
_verify(path, expected) = isfile(path) && _sha256(path) == expected

# ── Public entry point ──────────────────────────────────────────────────────

"""
    ensure_available(dir, groups; verify=false)

Ensure every file for the manifest `groups` (e.g. `["root", "geometry", "te"]`)
is present under `dir`. If all are present it is a no-op (optionally SHA-256
verified against the local `manifest.json`). Otherwise the missing groups are
produced by running the local `convert/` pipeline ([`convert_groups!`](@ref)),
which fetches the upstream TensorFlow weights and converts them to ONNX.
"""
function ensure_available(dir::AbstractString, groups::AbstractVector{<:AbstractString}; verify::Bool=false)
    incomplete = String[g for g in groups if !group_complete(dir, g)]
    if isempty(incomplete)
        verify && _verify_local(dir, groups)
        return dir
    end
    return convert_groups!(dir, incomplete)   # defined in convert.jl
end

"""Best-effort SHA-256 check of already-present `groups` against the local manifest."""
function _verify_local(dir::AbstractString, groups::AbstractVector{<:AbstractString})
    mp = joinpath(dir, "manifest.json")
    isfile(mp) || return dir
    items = get(JSON.parsefile(mp), "items", Dict{String,Any}())
    for g in groups
        haskey(items, g) || continue
        for entry in items[g]
            path = joinpath(dir, entry["path"])
            _verify(path, entry["sha256"]) ||
                error("SOLPSNN: SHA-256 mismatch for '$(entry["path"])' in $dir")
        end
    end
    return dir
end
