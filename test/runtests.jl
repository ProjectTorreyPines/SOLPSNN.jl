using Test
import SOLPSNN
import NPZ
import IMASdd as IMAS

# CI safety belt: never trigger the (multi-GB, conda-based) TF->ONNX conversion
# from the test suite. Artifact-dependent tests already skip without a local
# cache; this guarantees a stray `load_model` errors fast instead of attempting
# a SURFdrive fetch + conversion on a runner. Respect an explicit override.
get!(ENV, "FUSE_SOLPS_NN_AUTOCONVERT", "0")

const DATA = joinpath(@__DIR__, "data")

# SOLPS2imas is an optional (weak) dependency exercised via the SOLPS2imasExt
# package extension. Load it if available so the extension test can run; skip
# gracefully otherwise (e.g. if the netCDF stack is unavailable).
const HAVE_SOLPS2IMAS = try
    @eval import SOLPS2imas
    true
catch err
    @warn "SOLPS2imas not loadable; skipping SOLPS2imasExt test" err
    false
end

# Build a SOLPSModel with no ONNX sessions, for testing the pure-Julia
# pre/post-processing kernels in isolation (no artifacts needed).
function stub_model(; X_mean, X_std, qt, is_scalar=false, grid=(3, 3))
    return SOLPSNN.SOLPSModel(
        "stub", nothing, "stub", is_scalar, grid,
        Vector{SOLPSNN.ORT.InferenceSession}(), String[], String[],
        Float64.(X_mean), Float64.(X_std), qt)
end

# Synthetic B2 geometry on a regular integer lattice: cell (ix,iy) spans
# [ix-1,ix] x [iy-1,iy], so corners dedup to exactly (nx+1)*(ny+1) nodes.
# crx corner order matches upstream (LL, LR, UL, UR) = indices (1,2,3,4).
function synthetic_geometry(nx::Int, ny::Int; R_JET::Float64=3.0)
    nX, nY = nx + 2, ny + 2
    crx = Array{Float64,3}(undef, nX, nY, 4)
    cry = Array{Float64,3}(undef, nX, nY, 4)
    vol = Array{Float64,2}(undef, nX, nY)
    for ix in 1:nX, iy in 1:nY
        x0, x1 = ix - 1.0, Float64(ix)
        y0, y1 = iy - 1.0, Float64(iy)
        crx[ix, iy, :] .= (x0, x1, x0, x1)   # LL, LR, UL, UR
        cry[ix, iy, :] .= (y0, y0, y1, y1)
        vol[ix, iy] = 10.0 * ix + iy
    end
    return SOLPSNN.SOLPSGeometry(nx, ny, 0, 0, 0, R_JET, crx, cry, vol)
end

# The heavy ONNX artifacts (~1.3 GB) live outside git. Skip artifact-dependent
# tests gracefully when they (or the Python references) are not present.
function artifacts_dir()
    try
        return SOLPSNN.resolve_dir()
    catch
        return nothing
    end
end

@testset "SOLPSNN.jl" begin

    @testset "quantile inverse (unit)" begin
        # Identity-ish check: references = linspace(0,1,N), quantiles column = same
        # grid mapped to a known monotone function; Φ(0)=0.5 must land mid-table.
        refs = collect(range(0.0, 1.0; length=101))
        qcol = collect(range(-3.0, 3.0; length=101))       # values
        qt = SOLPSNN.QuantileTransformer(refs, reshape(qcol, :, 1))
        # y=0 -> Φ=0.5 -> interp at 0.5 -> value 0.0
        @test isapprox(SOLPSNN.inverse_transform_column(qt, 0.0, 1), 0.0; atol=1e-9)
        # monotonicity: larger latent -> larger physical value
        @test SOLPSNN.inverse_transform_column(qt, 1.0, 1) >
              SOLPSNN.inverse_transform_column(qt, -1.0, 1)
        # clamping outside table
        @test SOLPSNN.inverse_transform_column(qt, 50.0, 1) == qcol[end]
        @test SOLPSNN.inverse_transform_column(qt, -50.0, 1) == qcol[1]
    end

    @testset "normal CDF" begin
        @test isapprox(SOLPSNN._normcdf(0.0), 0.5; atol=1e-12)
        @test isapprox(SOLPSNN._normcdf(1.96), 0.975; atol=1e-3)
        @test isapprox(SOLPSNN._normcdf(-1.96), 0.025; atol=1e-3)
    end

    # ------------------------------------------------------------------ #
    # Self-contained regression tests: lock in the behavior of the custom #
    # pure-Julia kernels. These run WITHOUT the (out-of-git) ONNX models. #
    # ------------------------------------------------------------------ #
    @testset "regression: normalize_inputs" begin
        qt = SOLPSNN.QuantileTransformer([0.0, 0.5, 1.0], reshape([0.0, 1.0, 2.0], :, 1))
        X = [6.2 5.3 1.0e8 1.0e22 1.0e20 9.1e21 0.3 1.0]

        # identity scaler -> exposes the raw transform (P/2, log10 of puff/fuel)
        m = stub_model(; X_mean=zeros(8), X_std=ones(8), qt)
        Xn = SOLPSNN.normalize_inputs(m, X)
        @test size(Xn) == (1, 8)
        @test eltype(Xn) == Float32
        expected = Float32[6.2, 5.3, 5.0e7, 22.0, 20.0, log10(9.1e21), 0.3, 1.0]
        @test isapprox(vec(Xn), expected; rtol=1.0f-6)

        # affine scaler -> standardization applied after the raw transform
        m2 = stub_model(; X_mean=fill(1.0, 8), X_std=fill(2.0, 8), qt)
        Xn2 = SOLPSNN.normalize_inputs(m2, X)
        @test isapprox(vec(Xn2), (expected .- 1.0f0) ./ 2.0f0; rtol=1.0f-6)

        # batch handling: two identical rows -> identical normalized rows
        Xb = vcat(X, X)
        @test SOLPSNN.normalize_inputs(m, Xb)[1, :] == SOLPSNN.normalize_inputs(m, Xb)[2, :]
        @test_throws Exception SOLPSNN.normalize_inputs(m, X[:, 1:7])
    end

    @testset "regression: quantile inverse golden values" begin
        # references (CDF grid) -> quantile values (physical). Two features.
        refs = [0.0, 0.5, 1.0]
        quants = [10.0 100.0
                  20.0 200.0
                  40.0 400.0]
        qt = SOLPSNN.QuantileTransformer(refs, quants)
        @test SOLPSNN.nfeatures(qt) == 2

        z25 = -0.6744897501960817   # Φ⁻¹(0.25)
        z75 = 0.6744897501960817    # Φ⁻¹(0.75)

        # column 1: y=0 -> p=0.5 -> node value 20; p=0.25 -> 15; p=0.75 -> 30
        @test isapprox(SOLPSNN.inverse_transform_column(qt, 0.0, 1), 20.0; atol=1e-9)
        @test isapprox(SOLPSNN.inverse_transform_column(qt, z25, 1), 15.0; atol=1e-6)
        @test isapprox(SOLPSNN.inverse_transform_column(qt, z75, 1), 30.0; atol=1e-6)
        # column 2 uses its own quantile column (indexing correctness)
        @test isapprox(SOLPSNN.inverse_transform_column(qt, 0.0, 2), 200.0; atol=1e-9)
        @test isapprox(SOLPSNN.inverse_transform_column(qt, z25, 2), 150.0; atol=1e-5)
        # clamping at the table extremes
        @test SOLPSNN.inverse_transform_column(qt, 40.0, 1) == 40.0
        @test SOLPSNN.inverse_transform_column(qt, -40.0, 2) == 100.0
    end

    @testset "regression: synthetic geometry + GGD (no artifacts)" begin
        geo = synthetic_geometry(2, 3; R_JET=3.0)
        nX, nY = SOLPSNN.grid_size(geo)
        @test (nX, nY) == (4, 5)

        # scaling: lengths ∝ R, volumes ∝ R^3
        sc = SOLPSNN.scaled_geometry(geo, 6.0)   # s = 2
        @test isapprox(sc.crx, geo.crx .* 2)
        @test isapprox(sc.vol, geo.vol .* 8)

        dd = IMAS.dd()
        gi = SOLPSNN.build_edge_profiles_ggd!(dd, geo; R=3.0, time0=0.0)  # s=1, no scaling
        @test gi == 1
        g = dd.edge_profiles.grid_ggd[1]
        faces = g.space[1].objects_per_dimension[3].object
        nodes = g.space[1].objects_per_dimension[1].object
        # regular lattice -> corners dedup to exactly (nX+1)*(nY+1) nodes
        @test length(nodes) == (nX + 1) * (nY + 1)
        @test length(faces) == nX * nY
        # cell measures equal the (unscaled) volumes, in cell order c=(ix-1)*nY+iy
        for (ix, iy) in ((1, 1), (2, 3), (nX, nY))
            c = (ix - 1) * nY + iy
            @test faces[c].measure == geo.vol[ix, iy]
            @test faces[c].geometry == [Float64(ix), Float64(iy), geo.vol[ix, iy]]
        end

        # field flattening matches the cell ordering
        field = reshape(collect(1.0:(nX * nY)), nX, nY)
        ep = SOLPSNN.ggd_time_slice!(dd, 0.0)
        e = SOLPSNN.add_ggd_field!(ep.electrons.temperature, field; grid_index=gi)
        @test length(e.values) == nX * nY
        @test e.grid_index == 1 && e.grid_subset_index == 5
        for (ix, iy) in ((1, 1), (2, 4), (nX, nY))
            @test e.values[(ix - 1) * nY + iy] == field[ix, iy]
        end
    end

    @testset "SOLPS2imas extension: grid build + :solps ordering" begin
        b2 = joinpath(@__DIR__, "..", "convert", "geometry_data", "b2fgmtry")
        if !HAVE_SOLPS2IMAS
            @test_skip "SOLPS2imas not loadable"
        elseif !isfile(b2)
            @warn "b2fgmtry fixture missing; skipping SOLPS2imas extension test" b2
            @test_skip "b2fgmtry fixture missing"
        else
            # the extension supplies the method for the (empty) generic function
            @test !isempty(methods(SOLPSNN.build_edge_profiles_ggd_solps2imas!))
            @test Base.get_extension(SOLPSNN, :SOLPS2imasExt) !== nothing

            dd = IMAS.dd()
            gi = SOLPSNN.build_edge_profiles_ggd_solps2imas!(dd, b2; R=6.2, R_JET=3.0, time0=0.0)
            g = dd.edge_profiles.grid_ggd[end]
            sp = g.space[1]
            cells = sp.objects_per_dimension[3].object
            nodes = sp.objects_per_dimension[1].object
            ncell = length(cells)
            nX = SOLPS2imas.read_b2_output(b2)["dim"]["nx"]
            nY = ncell ÷ nX
            @test g.time == 0.0
            @test ncell == nX * nY
            @test length(nodes) > ncell            # shared corners, not 4*ncell

            # SOLPS2imas builds the full set of physical subsets (cells=5,
            # outer/inner target=13/14, separatrix=16) that the native builder lacks
            subs = Set(s.identifier.index for s in g.grid_subset)
            for idx in (5, 13, 14, 16)
                @test idx in subs
            end

            # R-scaling: node coordinates ∝ s = R/R_JET vs an unscaled build
            dd0 = IMAS.dd()
            SOLPSNN.build_edge_profiles_ggd_solps2imas!(dd0, b2)
            n0 = dd0.edge_profiles.grid_ggd[end].space[1].objects_per_dimension[1].object
            @test isapprox(nodes[7].geometry, n0[7].geometry .* (6.2 / 3.0); rtol=1e-9)

            # :solps cell ordering ic=(iy-1)*nX+ix  ==>  flatten is vec(field)
            ep = SOLPSNN.ggd_time_slice!(dd, 0.0)
            field = reshape(collect(1.0:ncell), nX, nY)
            e = SOLPSNN.add_ggd_field!(ep.electrons.temperature, field;
                                       grid_index=gi, subset_index=5, order=:solps)
            @test e.grid_index == gi && e.grid_subset_index == 5
            @test e.values == vec(field)
            for (ix, iy) in ((1, 1), (3, 4), (nX, nY))
                @test e.values[(iy - 1) * nX + ix] == field[ix, iy]
            end
        end
    end

    dir = artifacts_dir()
    if dir === nothing || !isfile(joinpath(dir, "manifest.json"))
        @warn "SOLPSNN artifacts not found; skipping model/geometry/parity tests. " *
              "Set FUSE_SOLPS_NN_DIR to the convert/ output to enable them."
    else
        @testset "geometry" begin
            geo = SOLPSNN.load_geometry(dir)
            @test SOLPSNN.grid_size(geo) == (geo.nx + 2, geo.ny + 2)
            @test size(geo.crx) == (geo.nx + 2, geo.ny + 2, 4)
            @test size(geo.vol) == (geo.nx + 2, geo.ny + 2)
            sc = SOLPSNN.scaled_geometry(geo, 2 * geo.R_JET)
            @test isapprox(sc.crx, geo.crx .* 2)              # lengths ∝ R
            @test isapprox(sc.vol, geo.vol .* 8)              # volumes ∝ R^3
        end

        @testset "model key routing" begin
            @test SOLPSNN.model_key("te", nothing) == "te"
            @test SOLPSNN.model_key("na", "D1") == "na1"
            @test SOLPSNN.model_key("na", "N7") == "na9"
            @test_throws Exception SOLPSNN.model_key("na", nothing)
            @test_throws Exception SOLPSNN.model_key("bogus", nothing)
        end

        @testset "GGD builder" begin
            geo = SOLPSNN.load_geometry(dir)
            nX, nY = SOLPSNN.grid_size(geo)
            dd = IMAS.dd()
            gi = SOLPSNN.build_edge_profiles_ggd!(dd, geo; R=6.2, time0=0.0)
            @test gi == 1
            g = dd.edge_profiles.grid_ggd[1]
            @test g.time == 0.0
            @test g.space[1].coordinates_type == [1, 2]
            faces = g.space[1].objects_per_dimension[3].object
            nodes = g.space[1].objects_per_dimension[1].object
            @test length(faces) == nX * nY
            @test 0 < length(nodes) <= 4 * nX * nY          # deduped corners
            @test length(faces[1].nodes) == 4
            @test g.grid_subset[1].identifier.index == 5
            @test g.grid_subset[1].dimension == 3

            # field/value alignment: c = (ix-1)*nY + iy
            field = reshape(collect(1.0:(nX * nY)), nX, nY)
            ep = SOLPSNN.ggd_time_slice!(dd, 0.0)
            e = SOLPSNN.add_ggd_field!(ep.electrons.temperature, field; grid_index=gi)
            @test length(e.values) == nX * nY
            @test e.grid_index == 1
            @test e.grid_subset_index == 5
            for (ix, iy) in ((1, 1), (3, 4), (nX, nY))
                @test e.values[(ix - 1) * nY + iy] == field[ix, iy]
            end
        end

        # Parity against upstream sklearn post-processing (test/gen_reference.py)
        Xfile = joinpath(DATA, "X_inputs.npy")
        if isfile(Xfile)
            X = Float64.(NPZ.npzread(Xfile))
            @testset "parity: $item" for item in ("te", "ti", "na1", "pwmxap", "psol")
                reffile = joinpath(DATA, "ref_$item.npy")
                isfile(reffile) || continue
                ref = NPZ.npzread(reffile)
                sp = startswith(item, "na") ? "D1" : nothing
                it = startswith(item, "na") ? "na" : item
                model = SOLPSNN.load_model(it; species=sp, dir=dir)
                got = SOLPSNN.predict(model, X)
                @test size(got) == size(ref)
                # relative match with a small floor for clamped near-zero cells
                @test isapprox(got, ref; rtol=1e-3, atol=1e-3 * maximum(abs, ref))
            end

            @testset "vector input convenience" begin
                geo = SOLPSNN.load_geometry(dir)
                q = SOLPSNN.load_model("pwmxap"; dir=dir)
                @test SOLPSNN.predict(q, X[1, :]) isa Real
                te = SOLPSNN.load_model("te"; dir=dir)
                f = SOLPSNN.predict(te, X[1, :])
                @test size(f) == (geo.nx + 2, geo.ny + 2)
            end
        else
            @warn "no test/data/X_inputs.npy; run test/gen_reference.py to enable parity tests"
        end
    end
end
