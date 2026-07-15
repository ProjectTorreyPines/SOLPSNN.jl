#= =============================================================== =#
#  Locating and (automatically) providing the converted ONNX artifacts
#
#  Upstream SOLPS-NN (github.com/sdasbach/solps-nn) distributes only
#  *TensorFlow SavedModels* on SURFdrive (see convert/config.json). SOLPSNN.jl
#  runs on ONNX (ONNXRunTime.jl), so the weights must first be converted by the
#  offline `convert/` pipeline (TF + tf2onnx, in a conda env).
#
#  Directory resolution order (always returns a path; population is on demand):
#    1. explicit `dir` argument
#    2. ENV["FUSE_SOLPS_NN_DIR"]
#    3. $PSCRATCH/solps-nn-onnx      (Perlmutter default of convert/run.sh)
#    4. <SOLPSNN repo>/../solps-nn-onnx  (writable sibling cache, dev fallback)
#
#  When a requested quantity is missing under `dir`, [`ensure_available`](@ref):
#    * if a mirror is configured (ENV["FUSE_SOLPS_NN_BASE_URL"] /
#      ENV["FUSE_SOLPS_NN_HF_REPO"], or a local manifest.json carrying a
#      non-empty `base_url`) -> downloads + SHA-256 verifies from it, or
#    * otherwise -> runs the `convert/` pipeline locally (auto-fetching the TF
#      weights from SURFdrive and converting them). See `convert.jl`.
#= =============================================================== =#

const ENV_DIR = "FUSE_SOLPS_NN_DIR"
const HF_REPO_ENV = "FUSE_SOLPS_NN_HF_REPO"
const HF_REVISION_ENV = "FUSE_SOLPS_NN_HF_REVISION"
const BASE_URL_ENV = "FUSE_SOLPS_NN_BASE_URL"
const HF_REVISION_DEFAULT = "main"

"""
    resolve_dir(; dir=nothing)

Return the directory holding the converted SOLPS-NN ONNX artifacts. Always
returns a path (creating/populating it is handled by [`ensure_available`](@ref)):
an explicit `dir`, then `ENV["$ENV_DIR"]`, then `\$PSCRATCH/solps-nn-onnx`, then
a writable `solps-nn-onnx` cache next to the installed package.
"""
function resolve_dir(; dir::Union{Nothing,AbstractString}=nothing)
    dir !== nothing && return String(dir)
    haskey(ENV, ENV_DIR) && return ENV[ENV_DIR]
    if haskey(ENV, "PSCRATCH")
        return joinpath(ENV["PSCRATCH"], "solps-nn-onnx")
    end
    return normpath(joinpath(pkgdir(SOLPSNN), "..", "solps-nn-onnx"))
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

# ── Mirror configuration & integrity ────────────────────────────────────────

"""
    configured_base_url(dir) -> String | nothing

The URL prefix of an explicit re-hosting mirror, or `nothing` if none is set
(in which case artifacts are produced by local conversion). A mirror is taken
from `ENV["$BASE_URL_ENV"]`, else `ENV["$HF_REPO_ENV"]`
(+`ENV["$HF_REVISION_ENV"]`), else a non-empty `base_url` in a local manifest.
"""
function configured_base_url(dir::AbstractString)
    haskey(ENV, BASE_URL_ENV) && return rstrip(ENV[BASE_URL_ENV], '/')
    if haskey(ENV, HF_REPO_ENV)
        rev = get(ENV, HF_REVISION_ENV, HF_REVISION_DEFAULT)
        return "https://huggingface.co/$(ENV[HF_REPO_ENV])/resolve/$rev"
    end
    mp = joinpath(dir, "manifest.json")
    if isfile(mp)
        b = rstrip(String(get(JSON.parsefile(mp), "base_url", "")), '/')
        !isempty(b) && return b
    end
    return nothing
end

"""sha256 hex digest of a file."""
_sha256(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))
_verify(path, expected) = isfile(path) && _sha256(path) == expected

"""Download `url` -> `path` atomically (via a temp file), cleaning up on error."""
function _download(url::AbstractString, path::AbstractString)
    mkpath(dirname(path))
    tmp = path * ".part"
    try
        Downloads.download(url, tmp)
    catch err
        isfile(tmp) && rm(tmp; force=true)
        error("SOLPSNN: failed to download $url\n  -> $err")
    end
    if filesize(tmp) == 0
        rm(tmp; force=true)
        error("SOLPSNN: $url downloaded empty")
    end
    mv(tmp, path; force=true)
    return path
end

# ── Public entry point ──────────────────────────────────────────────────────

"""
    ensure_available(dir, groups; verify=false)

Ensure every file for the manifest `groups` (e.g. `["root", "geometry", "te"]`)
is present under `dir`. If all are present it is a no-op (optionally SHA-256
verified against a local `manifest.json`). Otherwise the missing groups are
provided either by downloading from a configured mirror
([`configured_base_url`](@ref)) or, by default, by running the local
`convert/` pipeline ([`convert_groups!`](@ref)).
"""
function ensure_available(dir::AbstractString, groups::AbstractVector{<:AbstractString}; verify::Bool=false)
    incomplete = String[g for g in groups if !group_complete(dir, g)]
    if isempty(incomplete)
        verify && _verify_local(dir, groups)
        return dir
    end
    host = configured_base_url(dir)
    if host !== nothing
        return _download_groups!(dir, groups, host; verify)
    end
    return convert_groups!(dir, incomplete)   # defined in convert.jl
end

"""Best-effort SHA-256 check of already-present `groups` against a local manifest."""
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

# ── Hosted-mirror download path (opt-in via a configured base_url) ───────────

"""Fetch `dir/manifest.json` from `host` if missing; return the parsed manifest."""
function _ensure_manifest_from_host!(dir::AbstractString, host::AbstractString)
    mp = joinpath(dir, "manifest.json")
    if !isfile(mp)
        @info "SOLPSNN: fetching manifest.json" url = "$host/manifest.json"
        _download("$host/manifest.json", mp)
    end
    return JSON.parsefile(mp)
end

function _download_groups!(dir::AbstractString, groups::AbstractVector{<:AbstractString},
        host::AbstractString; verify::Bool=false)
    manifest = _ensure_manifest_from_host!(dir, host)
    items = manifest["items"]
    for g in groups
        haskey(items, g) || error("mirror manifest has no group '$g' (available: $(collect(keys(items))))")
        for entry in items[g]
            rel = entry["path"]
            path = joinpath(dir, rel)
            (isfile(path) && (!verify || _verify(path, entry["sha256"]))) && continue
            @info "SOLPSNN: downloading $rel"
            _download("$host/$rel", path)
            if verify && !_verify(path, entry["sha256"])
                error("SOLPSNN: SHA-256 mismatch after downloading '$rel'")
            end
        end
    end
    return dir
end

"""
    download_all!(; dir=resolve_dir(), verify=false) -> dir

Eagerly fetch **every** group in a configured mirror's manifest into `dir` (for
pre-baking containers / warming a shared cache). Requires a mirror
([`configured_base_url`](@ref)); without one, produce artifacts locally with
[`convert_models!`](@ref)(`["all"]`) instead.
"""
function download_all!(; dir::Union{Nothing,AbstractString}=nothing, verify::Bool=false)
    d = resolve_dir(; dir)
    host = configured_base_url(d)
    host === nothing && error(
        "SOLPSNN.download_all! needs a mirror (set ENV[\"$BASE_URL_ENV\"] or " *
        "ENV[\"$HF_REPO_ENV\"]); to build artifacts locally use SOLPSNN.convert_models!([\"all\"]).")
    manifest = _ensure_manifest_from_host!(d, host)
    groups = collect(keys(manifest["items"]))
    @info "SOLPSNN: prefetching $(length(groups)) groups from mirror -> $d" groups
    return _download_groups!(d, groups, host; verify)
end
