#= =============================================================== =#
#  SOLPSModel: 5-fold ONNX ensemble + pre/post-processing
#
#  Pipeline (per upstream solpsnn.Model):
#    preprocess   : P/2, log10(Dpuff,Npuff,Dcore)
#    standardize  : (X - X_mean) / X_std
#    forward      : mean over 5 ONNX folds of ...
#    inverse      : ... QuantileTransformer.inverse_transform (per column)
#
#  Inputs (order): [R, B, P, Dpuff, Npuff, Dcore, Dperp, chi_perp]
#= =============================================================== =#

# species -> model-folder index (verbatim from upstream species_dict)
const SPECIES_INDEX = Dict(
    "D0" => 0, "D1" => 1, "N0" => 2, "N1" => 3, "N2" => 4,
    "N3" => 5, "N4" => 6, "N5" => 7, "N6" => 8, "N7" => 9)

const SCALAR_ITEMS = ("pwmxap", "fnixap", "psol")
const FIELD_ITEMS = ("te", "ti", "na", "ua", "rqrad")
const SPECIES_ITEMS = ("na", "ua", "rqrad")

const N_FOLDS = 5
const N_INPUTS = 8

"""Resolve the on-disk model-folder key for an `item`/`species` pair."""
function model_key(item::AbstractString, species)
    if item in ("te", "ti") || item in SCALAR_ITEMS
        return String(item)
    elseif item in SPECIES_ITEMS
        species === nothing && error("item=$item requires a `species` (e.g. \"D1\")")
        haskey(SPECIES_INDEX, species) || error("unknown species '$species'")
        return string(item, SPECIES_INDEX[species])
    else
        error("unknown item '$item'")
    end
end

"""
    SOLPSModel

A loaded SOLPS-NN quantity: the 5 ONNX fold sessions (with their per-fold input
and output tensor names, which differ across folds), the input scaler, and the
output QuantileTransformer. For 2D fields `grid == (nx+2, ny+2)`; for scalars
`grid == (1, 1)`.
"""
struct SOLPSModel
    item::String
    species::Union{Nothing,String}
    key::String
    is_scalar::Bool
    grid::Tuple{Int,Int}
    sessions::Vector{ORT.InferenceSession}
    in_keys::Vector{String}
    out_keys::Vector{String}
    X_mean::Vector{Float64}
    X_std::Vector{Float64}
    qt::QuantileTransformer
end

function Base.show(io::IO, m::SOLPSModel)
    tag = m.species === nothing ? m.item : "$(m.item)[$(m.species)]"
    shape = m.is_scalar ? "scalar" : "$(m.grid[1])x$(m.grid[2]) field"
    print(io, "SOLPSModel($tag, $shape, $(length(m.sessions)) folds)")
end

"""
    load_model(item; species=nothing, dir=nothing, verify=false)

Load a SOLPS-NN quantity. `item` is one of `te`, `ti`, `pwmxap`, `fnixap`,
`psol`, or a species field (`na`/`ua`/`rqrad` with `species`, e.g.
`load_model("na"; species="D1")`). `dir` overrides artifact-directory
resolution; `verify=true` SHA-256-checks every file.

Any artifacts not already present in the resolved directory are produced
automatically — converted from the upstream TensorFlow weights on first use
(see [`resolve_dir`](@ref) / [`ensure_available`](@ref)). Disable auto-conversion
with `ENV["$AUTOCONVERT_ENV"] = "0"`.
"""
function load_model(item::AbstractString; species=nothing, dir=nothing, verify::Bool=false)
    d = resolve_dir(; dir)
    key = model_key(item, species)
    # Field items also need the shared geometry group (geometry.json sets the grid).
    groups = item in SCALAR_ITEMS ? ["root", key] : ["root", "geometry", key]
    ensure_available(d, groups; verify)

    X_mean = Float64.(vec(NPZ.npzread(joinpath(d, "X_mean.npy"))))
    X_std = Float64.(vec(NPZ.npzread(joinpath(d, "X_std.npy"))))
    length(X_mean) == N_INPUTS || error("X_mean has $(length(X_mean)) entries, expected $N_INPUTS")

    references = Float64.(vec(NPZ.npzread(joinpath(d, key, "references.npy"))))
    quantiles = Float64.(NPZ.npzread(joinpath(d, key, "quantiles.npy")))
    ndims(quantiles) == 2 || error("quantiles.npy for '$key' must be 2D (nq × nfeat)")
    qt = QuantileTransformer(references, quantiles)

    sessions = ORT.InferenceSession[]
    in_keys = String[]
    out_keys = String[]
    for k in 1:N_FOLDS
        s = ORT.load_inference(joinpath(d, key, "fold$k.onnx"))
        push!(sessions, s)
        push!(in_keys, only_name(ORT.input_names(s), "input", key, k))
        push!(out_keys, only_name(ORT.output_names(s), "output", key, k))
    end

    is_scalar = item in SCALAR_ITEMS
    nfeat = nfeatures(qt)
    if is_scalar
        grid = (1, 1)
        nfeat == 1 || error("scalar '$key' has nfeat=$nfeat (expected 1)")
    else
        meta = JSON.parsefile(joinpath(d, "geometry", "geometry.json"))
        grid = (Int(meta["nx"]) + 2, Int(meta["ny"]) + 2)
        prod(grid) == nfeat ||
            error("grid $(grid) (=$(prod(grid))) != quantile features $nfeat for '$key'")
    end

    return SOLPSModel(String(item), species === nothing ? nothing : String(species),
        key, is_scalar, grid, sessions, in_keys, out_keys, X_mean, X_std, qt)
end

function only_name(names::AbstractVector, kind, key, k)
    length(names) == 1 || error("fold$k of '$key' has $(length(names)) $kind tensors, expected 1")
    return String(names[1])
end

"""Apply upstream unit transform + standardization; return `Float32` `(N,8)`."""
function normalize_inputs(m::SOLPSModel, X::AbstractMatrix{<:Real})
    size(X, 2) == N_INPUTS || error("expected (N,$N_INPUTS) input, got $(size(X))")
    N = size(X, 1)
    Xp = Matrix{Float64}(undef, N, N_INPUTS)
    @inbounds for n in 1:N
        Xp[n, 1] = X[n, 1]            # R
        Xp[n, 2] = X[n, 2]            # B
        Xp[n, 3] = X[n, 3] / 2.0      # P / 2
        Xp[n, 4] = log10(X[n, 4])     # Dpuff
        Xp[n, 5] = log10(X[n, 5])     # Npuff
        Xp[n, 6] = log10(X[n, 6])     # Dcore
        Xp[n, 7] = X[n, 7]            # Dperp
        Xp[n, 8] = X[n, 8]            # chi_perp
    end
    @inbounds for c in 1:N_INPUTS, n in 1:N
        Xp[n, c] = (Xp[n, c] - m.X_mean[c]) / m.X_std[c]
    end
    return Float32.(Xp)
end

"""
    predict(model, X) -> Array

Ensemble prediction (mean of the per-fold inverse transforms).
`X` is `(N,8)` and returns `(N,)` for scalars or `(N, nx+2, ny+2)` for fields.
`X` may also be a length-8 vector, returning a scalar or `(nx+2, ny+2)` field.
"""
function predict(m::SOLPSModel, X::AbstractMatrix{<:Real})
    Xn = normalize_inputs(m, X)
    N = size(Xn, 1)
    if m.is_scalar
        acc = zeros(Float64, N)
        for k in 1:N_FOLDS
            raw = m.sessions[k](Dict(m.in_keys[k] => Xn))[m.out_keys[k]]  # (N,1)
            @inbounds for n in 1:N
                acc[n] += inverse_transform_column(m.qt, raw[n, 1], 1)
            end
        end
        acc ./= N_FOLDS
        return acc
    else
        gx, gy = m.grid
        acc = zeros(Float64, N, gx, gy)
        for k in 1:N_FOLDS
            raw = m.sessions[k](Dict(m.in_keys[k] => Xn))[m.out_keys[k]]  # (N,gx,gy)
            @inbounds for n in 1:N, i in 1:gx, j in 1:gy
                col = (i - 1) * gy + j        # C-order flatten to match quantiles
                acc[n, i, j] += inverse_transform_column(m.qt, raw[n, i, j], col)
            end
        end
        acc ./= N_FOLDS
        return acc
    end
end

function predict(m::SOLPSModel, x::AbstractVector{<:Real})
    out = predict(m, reshape(collect(x), 1, :))
    return m.is_scalar ? out[1] : dropdims(out; dims=1)
end
