module SOLPS2imasExt

# Package extension: build the edge_profiles GGD from a SOLPS b2fgmtry via
# SOLPS2imas.jl, so the SOLPS-NN surrogate populates the *same* IMAS grid
# (nodes/edges/cells + all physical subsets) that a real SOLPS run would,
# enabling drop-in use with GGDUtils / SOLPS2ctrl downstream tooling.

import SOLPSNN
import SOLPS2imas

"""
    build_edge_profiles_ggd_solps2imas!(dd, b2gmtry; R=nothing, R_JET=nothing, time0=nothing)

See the docstring of `SOLPSNN.build_edge_profiles_ggd_solps2imas!`.

Returns the `grid_index` (the `grid_ggd.identifier.index` that SOLPS2imas assigns
and that its field entries reference). Rescales node coordinates and object
measures by `s = R / R_JET` when both are given (lengths ∝ s, edge areas ∝ s²,
cell volumes ∝ s³, and the toroidal node `measure = 2πr` ∝ s).
"""
function SOLPSNN.build_edge_profiles_ggd_solps2imas!(
    dd, b2gmtry::AbstractString;
    R::Union{Nothing,Real}=nothing,
    R_JET::Union{Nothing,Real}=nothing,
    time0::Union{Nothing,Real}=nothing,
)
    SOLPS2imas.solps2imas(b2gmtry; ids=dd)

    slice = length(dd.edge_profiles.grid_ggd)
    g = dd.edge_profiles.grid_ggd[slice]
    grid_index = g.identifier.index

    if time0 !== nothing
        g.time = Float64(time0)
    end

    if R !== nothing && R_JET !== nothing
        s = Float64(R) / Float64(R_JET)
        sp = g.space[1]
        # solps2imas sets geometry/measure on every node/edge/cell, so scale
        # unconditionally (lengths ∝ s, node 2πr ∝ s, areas ∝ s², volumes ∝ s³).
        for nd in sp.objects_per_dimension[1].object          # nodes
            nd.geometry = nd.geometry .* s
            nd.measure *= s
        end
        for ed in sp.objects_per_dimension[2].object          # edges
            ed.measure *= s^2
        end
        for cl in sp.objects_per_dimension[3].object          # cells
            cl.measure *= s^3
        end
    end

    return grid_index
end

end # module SOLPS2imasExt
