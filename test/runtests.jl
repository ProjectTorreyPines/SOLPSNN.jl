using Test
import SOLPSNN
import NPZ
import IMAS

const DATA = joinpath(@__DIR__, "data")

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
