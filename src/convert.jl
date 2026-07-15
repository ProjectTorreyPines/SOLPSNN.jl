#= =============================================================== =#
#  On-demand TF -> ONNX conversion
#
#  Upstream ships TensorFlow SavedModels (not ONNX). When the ONNX artifacts a
#  quantity needs are absent and no download mirror is configured, we run the
#  bundled `convert/run.sh` pipeline, which:
#     1. downloads the requested items' SavedModels + sidecars from SURFdrive,
#     2. converts each fold to ONNX (tf2onnx, opset 17),
#     3. parses the bundled b2fgmtry geometry,
#     4. writes manifest.json (sizes + sha256).
#
#  The pipeline runs in a conda env with TensorFlow + tf2onnx (see
#  convert/requirements.txt). Users on NERSC get conda via `module load conda`;
#  FUSE users have the bundled `fuse` conda env. The convert env itself is
#  created on first use if missing.
#
#  Controls (all optional):
#    FUSE_SOLPS_NN_AUTOCONVERT        "0" to disable auto-conversion (fail fast)
#    FUSE_SOLPS_NN_CONVERT_ENV        conda env prefix (default
#                                     $PSCRATCH/.conda/envs/solpsnn-convert)
#    FUSE_SOLPS_NN_CREATE_CONVERT_ENV "0" to forbid creating that env
#    FUSE_SOLPS_NN_RAW_DIR            raw SavedModel download dir
#    FUSE_SOLPS_NN_CONDA              conda/mamba executable (else auto-detected)
#= =============================================================== =#

const AUTOCONVERT_ENV = "FUSE_SOLPS_NN_AUTOCONVERT"
const CONVERT_ENV_ENV = "FUSE_SOLPS_NN_CONVERT_ENV"
const CREATE_ENV_ENV = "FUSE_SOLPS_NN_CREATE_CONVERT_ENV"
const RAW_DIR_ENV = "FUSE_SOLPS_NN_RAW_DIR"
const CONDA_ENV = "FUSE_SOLPS_NN_CONDA"

_env_on(name, default) = lowercase(get(ENV, name, default)) ∉ ("0", "false", "no", "off")
autoconvert_enabled() = _env_on(AUTOCONVERT_ENV, "1")
create_convert_env_enabled() = _env_on(CREATE_ENV_ENV, "1")

"""Absolute path to the bundled `convert/` pipeline directory."""
convert_dir() = pkgdir(SOLPSNN, "convert")

"""Conda env prefix used for the TF -> ONNX conversion."""
function convert_env_prefix()
    haskey(ENV, CONVERT_ENV_ENV) && return ENV[CONVERT_ENV_ENV]
    base = get(ENV, "PSCRATCH", homedir())
    return joinpath(base, ".conda", "envs", "solpsnn-convert")
end

"""Locate a conda-like executable (`conda`/`mamba`/`micromamba`), or `nothing`."""
function find_conda()
    for c in
        (get(ENV, CONDA_ENV, ""), get(ENV, "CONDA_EXE", ""), "conda", "mamba", "micromamba")
        isempty(c) && continue
        exe = Sys.which(c)
        exe !== nothing && return exe
    end
    return nothing
end

convert_env_exists(prefix::AbstractString) =
    isfile(joinpath(prefix, "bin", "python")) || isdir(joinpath(prefix, "conda-meta"))

"""Cache/tempdir overrides that keep conda + pip off a quota-limited \$HOME."""
function _convert_env_overrides()
    scratch = get(ENV, "PSCRATCH", nothing)
    ov = Dict{String,String}()
    if scratch !== nothing
        ov["TMPDIR"] = get(ENV, "TMPDIR", joinpath(scratch, "tmp"))
        ov["PIP_CACHE_DIR"] = get(ENV, "PIP_CACHE_DIR", joinpath(scratch, ".cache", "pip"))
        ov["CONDA_PKGS_DIRS"] =
            get(ENV, "CONDA_PKGS_DIRS", joinpath(scratch, ".conda", "pkgs"))
    end
    for (k, v) in ov
        mkpath(v)
    end
    return ov
end

"""
    ensure_convert_env!(conda, prefix; verbose=true) -> prefix

Ensure the conda env at `prefix` exists with the conversion dependencies,
creating it from `convert/requirements.txt` on first use (multi-GB, one-time).
"""
function ensure_convert_env!(
    conda::AbstractString,
    prefix::AbstractString;
    verbose::Bool = true,
)
    convert_env_exists(prefix) && return prefix
    if !create_convert_env_enabled()
        error(
            """
          SOLPS-NN conversion env not found at:
              $prefix
          and creation is disabled ($CREATE_ENV_ENV=0). Create it manually:
              $conda create -y --prefix $prefix python=3.10 pip
              $conda run --prefix $prefix python -m pip install -r $(joinpath(convert_dir(), "requirements.txt"))
          or set ENV["$CONVERT_ENV_ENV"] to an existing env.
          """,
        )
    end
    req = joinpath(convert_dir(), "requirements.txt")
    verbose && @warn "SOLPSNN: creating conversion conda env (TensorFlow + tf2onnx). \
        This is a one-time, multi-GB install and may take several minutes." prefix
    withenv(_convert_env_overrides()...) do
        run(`$conda create -y --prefix $prefix python=3.10 pip`)
        run(`$conda run --no-capture-output --prefix $prefix python -m pip install -r $req`)
    end
    return prefix
end

"""
    convert_models!(items; dir=resolve_dir(), raw_dir=nothing, validate=false,
                    verbose=true) -> dir

Run the bundled `convert/run.sh` to produce the ONNX artifacts for `items`
(SOLPS-NN quantity keys, e.g. `["te","ti","na1","pwmxap","psol"]`, or `["all"]`)
into `dir`. Auto-fetches the upstream TensorFlow weights and converts them; the
conversion conda env is created if missing. Requires a conda-like executable on
`PATH` (e.g. `module load conda` on NERSC, or the FUSE conda env).
"""
function convert_models!(
    items::AbstractVector{<:AbstractString};
    dir::Union{Nothing,AbstractString} = nothing,
    raw_dir::Union{Nothing,AbstractString} = nothing,
    validate::Bool = false,
    verbose::Bool = true,
)
    d = resolve_dir(; dir)
    conda = find_conda()
    conda === nothing && error(
        "SOLPSNN: no conda/mamba/micromamba on PATH. Run `module load conda` (NERSC) or " *
        "activate the FUSE conda env, set ENV[\"$CONDA_ENV\"], or point ENV[\"$ENV_DIR\"] at " *
        "an already-converted artifact dir.",
    )
    prefix = convert_env_prefix()
    ensure_convert_env!(conda, prefix; verbose)

    raw =
        raw_dir !== nothing ? String(raw_dir) :
        get(ENV, RAW_DIR_ENV, normpath(joinpath(d, "..", "solps-nn-data")))
    run_sh = joinpath(convert_dir(), "run.sh")
    isfile(run_sh) || error("SOLPSNN: conversion script not found at $run_sh")
    mkpath(d)
    mkpath(raw)

    overrides = merge(
        _convert_env_overrides(),
        Dict(
            "CONVERT_ENV" => prefix,
            "RAW_DIR" => raw,
            "OUT_DIR" => d,
            "BASE_URL" => "",
            "VALIDATE" => validate ? "1" : "0",
            "PATH" => string(
                dirname(conda),
                Sys.iswindows() ? ";" : ":",
                get(ENV, "PATH", ""),
            ),
        ),
    )
    verbose &&
        @info "SOLPSNN: converting $(collect(items)) via $run_sh" out_dir = d raw_dir = raw env =
            prefix
    withenv(overrides...) do
        run(`bash $run_sh $(collect(String, items))`)
    end
    return d
end

"""
    convert_groups!(dir, groups) -> dir

Produce the missing manifest `groups` under `dir` via local conversion. Item
groups are converted with [`convert_models!`](@ref); a lone `geometry`/`root`
requirement is satisfied as a by-product (the pipeline always writes the input
scaler, geometry, and manifest). Honors `ENV["$AUTOCONVERT_ENV"]`.
"""
function convert_groups!(dir::AbstractString, groups::AbstractVector{<:AbstractString})
    if !autoconvert_enabled()
        error(
            """
          SOLPS-NN ONNX artifacts are missing under:
              $dir
          (missing groups: $(collect(groups))) and auto-conversion is disabled
          ($AUTOCONVERT_ENV=0). Options:
            * run  SOLPSNN.convert_models!($(repr(_items_for(groups))); dir="$dir")
            * run  $(joinpath(convert_dir(), "run.sh"))  yourself, then set ENV["$ENV_DIR"]
            * point ENV["$ENV_DIR"] at an already-converted directory
          """,
        )
    end
    items = _items_for(groups)
    if isempty(items)
        # Only root/geometry requested: convert the headline set (also writes
        # root, geometry, manifest). Rare — load_model always requests an item.
        items = ["te", "ti", "na1", "pwmxap", "psol"]
    end
    convert_models!(items; dir)
    still = String[g for g in groups if !group_complete(dir, g)]
    isempty(still) ||
        error("SOLPSNN: conversion completed but groups still missing: $still")
    return dir
end

"""Model item keys among `groups` (drops the `root`/`geometry` by-products)."""
_items_for(groups) = String[g for g in groups if g ∉ ("root", "geometry")]
