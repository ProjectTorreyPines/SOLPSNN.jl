#= =============================================================== =#
#  SOLPS/B2 structured grid -> IMAS edge_profiles GGD
#
#  Builds a General Grid Description grid under `dd.edge_profiles.grid_ggd`
#  from the fixed B2 cell-corner arrays (`crx`/`cry`, shape (nx+2, ny+2, 4)),
#  scaled to major radius `R`, and provides helpers to attach 2D field values
#  (`edge_profiles.ggd[...]`) on that grid.
#
#  Conventions (IMAS GGD, see edge_profiles DD):
#    space.coordinates_type      = [1, 2]  (R, Z)
#    objects_per_dimension[1]    = nodes   (object.geometry = [R, Z])
#    objects_per_dimension[2]    = edges   (left empty)
#    objects_per_dimension[3]    = 2D cells/faces (object.nodes = 4 corners)
#    grid_subset identifier.index= 5 "cells", dimension = 3 (implicit elements)
#
#  Cell ordering (must match field flattening): ix outer, iy inner ->
#    c = (ix-1)*(ny+2) + iy, i.e. `vec(permutedims(field))`.
#= =============================================================== =#

const CELLS_SUBSET = 5   # IMAS grid_subset identifier.index for "cells"
const CELLS_DIM = 3      # objects_per_dimension index / dimension for 2D faces

# B2 stores 4 corners per cell as (LL, LR, UL, UR) in (poloidal, radial); reorder
# to a counter-clockwise polygon (LL, LR, UR, UL) for a valid quad.
const _CORNER_CCW = (1, 2, 4, 3)

"""Flatten a `(nx+2, ny+2)` field to the GGD cell ordering `c=(ix-1)*ny_tot+iy`."""
_flatten_cells(field::AbstractMatrix) = vec(permutedims(field))

"""
    build_edge_profiles_ggd!(dd, geo; R, time0=dd.global_time, grid_name="SOLPS-NN b2")

Populate a new `dd.edge_profiles.grid_ggd` slice with the SOLPS B2 grid scaled
to major radius `R`. Returns the 1-based `grid_index` (position in
`dd.edge_profiles.grid_ggd`) to be referenced by field entries.
"""
function build_edge_profiles_ggd!(dd::IMAS.dd, geo::SOLPSGeometry; R::Real,
                                  time0::Float64=dd.global_time, grid_name::AbstractString="SOLPS-NN b2")
    sc = scaled_geometry(geo, R)
    crx, cry, vol = sc.crx, sc.cry, sc.vol
    nX, nY = size(vol)              # (nx+2, ny+2)

    # ---- grid slice (grid_ggd is not time-aware at the vector level) ----
    grid_index = length(dd.edge_profiles.grid_ggd) + 1
    resize!(dd.edge_profiles.grid_ggd, grid_index)
    g = dd.edge_profiles.grid_ggd[grid_index]
    g.time = time0
    g.identifier.name = String(grid_name)
    g.identifier.index = 1
    g.identifier.description = "Fixed SOLPS-ITER B2 grid (scaled by R/R_JET) from SOLPS-NN"

    resize!(g.space, 1)
    sp = g.space[1]
    sp.identifier.name = "primary_standard"
    sp.identifier.index = 1
    sp.coordinates_type = [1, 2]   # R, Z
    sp.geometry_type.index = 0     # standard

    resize!(sp.objects_per_dimension, CELLS_DIM)  # [nodes, edges, faces]
    nodes_opd = sp.objects_per_dimension[1]
    faces_opd = sp.objects_per_dimension[3]
    nodes_opd.geometry_content.index = 1          # node coordinates

    # ---- nodes (deduplicated corners) ----
    node_key(r, z) = (round(r; digits=8), round(z; digits=8))
    node_ids = Dict{Tuple{Float64,Float64},Int}()
    node_coords = Vector{Tuple{Float64,Float64}}()
    function node_id!(r, z)
        k = node_key(r, z)
        id = get(node_ids, k, 0)
        id != 0 && return id
        push!(node_coords, (r, z))
        id = length(node_coords)
        node_ids[k] = id
        return id
    end

    ncells = nX * nY
    cell_nodes = Vector{NTuple{4,Int}}(undef, ncells)
    cell_measure = Vector{Float64}(undef, ncells)
    @inbounds for ix in 1:nX, iy in 1:nY
        c = (ix - 1) * nY + iy
        n = ntuple(4) do t
            corner = _CORNER_CCW[t]
            node_id!(crx[ix, iy, corner], cry[ix, iy, corner])
        end
        cell_nodes[c] = n
        cell_measure[c] = vol[ix, iy]
    end

    resize!(nodes_opd.object, length(node_coords))
    @inbounds for (id, (r, z)) in enumerate(node_coords)
        nodes_opd.object[id].geometry = [r, z]
    end

    faces_opd.geometry_content.index = 31   # (ix, iy, volume)
    resize!(faces_opd.object, ncells)
    @inbounds for ix in 1:nX, iy in 1:nY
        c = (ix - 1) * nY + iy
        obj = faces_opd.object[c]
        obj.nodes = collect(cell_nodes[c])
        obj.measure = cell_measure[c]
        obj.geometry = [Float64(ix), Float64(iy), cell_measure[c]]
    end

    # ---- cells subset (implicit: no explicit elements needed) ----
    resize!(g.grid_subset, 1)
    ss = g.grid_subset[1]
    ss.identifier.name = "cells"
    ss.identifier.index = CELLS_SUBSET
    ss.identifier.description = "All cells (2D)"
    ss.dimension = CELLS_DIM

    return grid_index
end

"""
    ggd_time_slice!(dd, time0=dd.global_time)

Return (creating if needed) the `dd.edge_profiles.ggd` element at `time0`.
"""
ggd_time_slice!(dd::IMAS.dd, time0::Float64=dd.global_time) = resize!(dd.edge_profiles.ggd, time0)

"""
    add_ggd_field!(field_vec, field2D; grid_index, subset_index=CELLS_SUBSET)

Attach a 2D `(nx+2, ny+2)` field as a single `generic_grid_scalar` entry on the
cells subset. `field_vec` is a GGD scalar vector such as
`ep.electrons.temperature`, `ep.electrons.density`, or `ep.ion[i].density`.
"""
function add_ggd_field!(field_vec, field2D::AbstractMatrix; grid_index::Integer, subset_index::Integer=CELLS_SUBSET)
    resize!(field_vec, 1)
    e = field_vec[1]
    e.grid_index = grid_index
    e.grid_subset_index = subset_index
    e.values = _flatten_cells(field2D)
    return e
end
