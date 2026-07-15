#= =============================================================== =#
#  Fixed SOLPS-ITER B2 grid geometry
#
#  SOLPS-NN predicts fields on a single fixed B2 grid (parsed offline from
#  `b2fgmtry` into crx/cry/vol .npy + geometry.json). Different machine sizes
#  are represented purely by rescaling the geometry with `R / R_JET`
#  (lengths ∝ R, cell volumes ∝ R^3), matching the upstream GeometryModel.
#= =============================================================== =#

"""
    SOLPSGeometry

Fixed B2 grid: `crx`/`cry` are the R,Z coordinates of the 4 corners of every
cell, shape `(nx+2, ny+2, 4)` (including guard cells); `vol` is `(nx+2, ny+2)`.
`leftcut`/`rightcut`/`topcut` encode the magnetic topology (separatrix/PFR).
"""
struct SOLPSGeometry
    nx::Int
    ny::Int
    leftcut::Int
    rightcut::Int
    topcut::Int
    R_JET::Float64
    crx::Array{Float64,3}   # (nx+2, ny+2, 4)
    cry::Array{Float64,3}   # (nx+2, ny+2, 4)
    vol::Matrix{Float64}    # (nx+2, ny+2)
end

"""
Reference major radius of the fixed B2 grid, taken verbatim from the upstream
SOLPS-NN `GeometryModel`. All machine sizes are represented by rescaling the
grid with `R / R_JET`. Kept in sync with `convert/parse_geometry.py` and the
`R_JET` field written into `geometry.json`.
"""
const R_JET = 3.000727179161820

"""number of poloidal x radial cells including guard cells: `(nx+2, ny+2)`."""
grid_size(geo::SOLPSGeometry) = (geo.nx + 2, geo.ny + 2)

"""
    load_geometry(dir)

Load the geometry artifacts produced by `convert/parse_geometry.py` from
`<dir>/geometry/`. Missing artifacts are auto-fetched (see [`ensure_available`](@ref)).
"""
function load_geometry(dir::AbstractString)
    ensure_available(dir, ["geometry"])
    gdir = joinpath(dir, "geometry")
    meta = JSON.parsefile(joinpath(gdir, "geometry.json"))
    crx = Float64.(NPZ.npzread(joinpath(gdir, "crx.npy")))
    cry = Float64.(NPZ.npzread(joinpath(gdir, "cry.npy")))
    vol = Float64.(NPZ.npzread(joinpath(gdir, "vol.npy")))
    return SOLPSGeometry(
        Int(meta["nx"]),
        Int(meta["ny"]),
        Int(meta["leftcut"]),
        Int(meta["rightcut"]),
        Int(meta["topcut"]),
        Float64(meta["R_JET"]),
        crx,
        cry,
        vol,
    )
end

"""
    scaled_geometry(geo, R)

Return `(; crx, cry, vol)` rescaled to major radius `R`: coordinates scale as
`R/R_JET`, cell volumes as `(R/R_JET)^3` (verbatim from upstream GeometryModel).
"""
function scaled_geometry(geo::SOLPSGeometry, R::Real)
    s = R / geo.R_JET
    return (crx = geo.crx .* s, cry = geo.cry .* s, vol = geo.vol .* s^3)
end
