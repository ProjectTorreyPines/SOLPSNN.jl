#= =============================================================== =#
#  sklearn QuantileTransformer inverse (output_distribution="normal")
#
#  SOLPS-NN networks emit values in a standard-normal latent space; the
#  physical field is recovered by the *inverse* of an sklearn
#  QuantileTransformer that was fit column-by-column (one mapping per output
#  feature: per grid cell for 2D fields, a single column for scalars).
#
#  sklearn inverse (per column):
#     x01   = Φ(y)                              # standard-normal CDF
#     value = interp(x01, references, quantiles[:, col])
#  where `references` is a length-nq ascending grid on [0,1] and
#  `quantiles[:, col]` are the fitted quantile values for that column.
#  `np.interp` clamps to the endpoints outside [references[1], references[end]],
#  which exactly reproduces sklearn's lower/upper-bound handling because
#  references spans [0,1] and Φ(y) ∈ [0,1].
#= =============================================================== =#

const _INV_SQRT2 = 0.7071067811865476

"""Standard-normal CDF Φ(y) via the complementary error function."""
@inline _normcdf(y::Real) = 0.5 * erfc(-Float64(y) * _INV_SQRT2)

"""
    _interp_clamped(x, xp, yp)

Linear interpolation of the value at `x` given ascending knots `xp` and values
`yp`, clamped to `yp[1]`/`yp[end]` outside the `xp` range (matches `np.interp`).
"""
@inline function _interp_clamped(x::Float64, xp::AbstractVector{Float64}, yp::AbstractVector{Float64})
    n = length(xp)
    @inbounds begin
        x <= xp[1] && return yp[1]
        x >= xp[n] && return yp[n]
        k = searchsortedlast(xp, x)          # xp[k] <= x < xp[k+1]
        k = clamp(k, 1, n - 1)
        x0 = xp[k]
        x1 = xp[k+1]
        dx = x1 - x0
        t = dx > 0 ? (x - x0) / dx : 0.0
        return yp[k] + t * (yp[k+1] - yp[k])
    end
end

"""
    QuantileTransformer(references, quantiles)

Holds the fitted state loaded from `references.npy` (`nq`) and `quantiles.npy`
(`nq × nfeat`, C-order flattened over the field grid for 2D targets).
"""
struct QuantileTransformer
    references::Vector{Float64}   # ascending on [0,1], length nq
    quantiles::Matrix{Float64}    # (nq, nfeat)
end

nfeatures(qt::QuantileTransformer) = size(qt.quantiles, 2)

"""
    inverse_transform_column(qt, y, col)

Map a single network latent value `y` back to physical units using the
per-column quantile mapping `col`.
"""
@inline function inverse_transform_column(qt::QuantileTransformer, y::Real, col::Integer)
    x01 = _normcdf(y)
    return _interp_clamped(x01, qt.references, view(qt.quantiles, :, col))
end
