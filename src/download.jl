#= =============================================================== =#
#  Locating and (optionally) fetching the converted ONNX artifacts
#
#  Artifacts are kept out of the git repo (~1.3 GB for the headline set).
#  Resolution order:
#    1. explicit `dir` argument
#    2. ENV["FUSE_SOLPS_NN_DIR"]
#    3. $PSCRATCH/solps-nn-onnx (Perlmutter default of the conversion pipeline)
#
#  If `manifest.json` carries a non-empty `base_url`, missing files can be
#  fetched + SHA-256 verified on demand (mirrors the FUSE pedestal predictor).
#= =============================================================== =#

const ENV_DIR = "FUSE_SOLPS_NN_DIR"

"""
    resolve_dir(; dir=nothing)

Return the directory holding the converted SOLPS-NN ONNX artifacts.
"""
function resolve_dir(; dir::Union{Nothing,AbstractString}=nothing)
    dir !== nothing && return String(dir)
    haskey(ENV, ENV_DIR) && return ENV[ENV_DIR]
    if haskey(ENV, "PSCRATCH")
        cand = joinpath(ENV["PSCRATCH"], "solps-nn-onnx")
        isdir(cand) && return cand
    end
    error("""
        Could not locate the SOLPS-NN ONNX artifacts.
        Set ENV["$ENV_DIR"] to the directory produced by SOLPSNN/convert/run.sh,
        e.g. export $ENV_DIR=\$PSCRATCH/solps-nn-onnx
        """)
end

"""sha256 hex digest of a file."""
_sha256(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))

_verify(path, expected) = isfile(path) && _sha256(path) == expected

"""
    ensure_available(dir, groups; verify=false)

Ensure every file for the manifest `groups` (e.g. `["root", "te"]`) is present
under `dir`, downloading from `base_url` when necessary. With `verify=true`,
SHA-256 is checked (and enforced after any download).
"""
function ensure_available(dir::AbstractString, groups::AbstractVector{<:AbstractString}; verify::Bool=false)
    manifest_path = joinpath(dir, "manifest.json")
    isfile(manifest_path) || error("no manifest.json in $dir; run SOLPSNN/convert/run.sh")
    manifest = JSON.parsefile(manifest_path)
    base_url = String(get(manifest, "base_url", ""))
    items = manifest["items"]
    for g in groups
        haskey(items, g) || error("manifest has no group '$g' (available: $(collect(keys(items))))")
        for entry in items[g]
            rel = entry["path"]
            path = joinpath(dir, rel)
            ok = isfile(path) && (!verify || _verify(path, entry["sha256"]))
            ok && continue
            if isempty(base_url)
                error("missing artifact '$rel' in $dir and manifest has no base_url to download from")
            end
            url = rstrip(base_url, '/') * "/" * rel
            mkpath(dirname(path))
            @info "SOLPSNN: downloading $rel"
            Downloads.download(url, path)
            if verify && !_verify(path, entry["sha256"])
                error("SHA-256 mismatch after downloading '$rel'")
            end
        end
    end
    return dir
end
