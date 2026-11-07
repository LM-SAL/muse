import shutil
import warnings

import attrs
import dask.array as da
import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.instrument import align_response_and_vdem, response_io
from muse.instrument.response_io import load_and_concat_responses, read_response, save_response
from muse.tests.helpers import fake_response, fake_response_file, fake_vdem
from muse.variables import DEFAULTS_MUSE

pytestmark = [
    pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning"),
    pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning"),
    pytest.mark.filterwarnings("ignore:numpy.ndarray size changed:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning"),
]


# The shipped SG and CI gains are equal, so only a CI table that differs proves the CI table was read.
_CI_GAIN = xr.DataArray(np.array([7.0, 7.0]) * u.electron / u.DN, coords={"channel": [195, 304]}, dims="channel")


def _write(ds: xr.Dataset, path, fmt: str) -> str:
    if fmt == "nc":
        ds.to_netcdf(path)
    else:
        ds.to_zarr(path)
    return str(path)


def _axis(values, name: str) -> xr.DataArray:
    values = np.asarray(values, dtype=float)
    return xr.DataArray(values, dims=name, coords={name: values})


def _slit(n: int) -> xr.DataArray:
    return xr.DataArray(np.arange(n), dims="slit", coords={"slit": np.arange(n)})


def _open(path, fmt: str) -> xr.Dataset:
    if fmt == "zarr":
        return xr.open_zarr(path, consolidated=False)
    return xr.open_dataset(path)


@pytest.fixture
def response_path(tmp_path):
    # For tests reading a pristine legacy file where the on-disk format is irrelevant.
    return _write(fake_response_file(), tmp_path / "resp.nc", "nc")


def _small_response() -> xr.Dataset:
    response = fake_response().isel(line=slice(0, 2), logT=slice(0, 3), slit=slice(0, 4))
    response = response.assign_coords(component_kind=("line", ["line"] * response.sizes["line"]))
    return response.assign_attrs(normalization=1e-27, HISTORY=["create response", "map response"])


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
def test_save_response_roundtrip_uses_default_chunks(tmp_path, fmt) -> None:
    response = _small_response()
    before = response.copy(deep=True)
    path = tmp_path / f"response.{fmt}"

    save_response(response, path)

    with _open(path, fmt) as source:
        encoding = source.detector_response.encoding
        chunks = encoding["chunks" if fmt == "zarr" else "chunksizes"]
        if fmt == "zarr":
            assert encoding["compressors"][0].to_dict() == {
                "name": "blosc",
                "configuration": {
                    "typesize": 8,
                    "cname": "zstd",
                    "clevel": 3,
                    "shuffle": "bitshuffle",
                    "blocksize": 0,
                },
            }
        else:
            assert {name: encoding[name] for name in ("zlib", "complevel", "shuffle")} == {
                "zlib": True,
                "complevel": 1,
                "shuffle": True,
            }
        loaded = source.load()
    assert chunks == (1, min(20, response.sizes["doppler_velocity"]), 1, 4, response.sizes["detector_x_pixel"])
    xr.testing.assert_identical(loaded, before)
    xr.testing.assert_identical(response, before)


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
def test_save_response_then_read_response_round_trips(tmp_path, fmt) -> None:
    # The save tests read back with the raw opener and the read tests are fed raw to_netcdf
    # output, so nothing checked that what save_response writes is what read_response accepts.
    response = _small_response()
    path = tmp_path / f"response.{fmt}"

    save_response(response, path)
    loaded = read_response(path)

    # read_response adds a gain coordinate of its own, so compare the data rather than the
    # whole object.
    assert loaded.detector_response.dims == response.detector_response.dims
    np.testing.assert_allclose(loaded.detector_response.values, response.detector_response.values)
    np.testing.assert_allclose(loaded.line_wavelength.values, response.line_wavelength.values)
    # The injected gain is per line, selected from the per-channel defaults in electron/DN.
    assert loaded.gain.dims == ("line",)
    np.testing.assert_allclose(
        loaded.gain.values,
        u.Quantity(DEFAULTS_MUSE.ccd_gain_SG.sel(channel=loaded.channel).data).to_value(u.electron / u.DN),
    )
    assert loaded.gain.attrs["units"] == str(u.electron / u.DN)


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
def test_save_response_accepts_chunk_overrides(tmp_path, fmt) -> None:
    response = _small_response()
    path = tmp_path / f"response.{fmt}"

    save_response(
        response, path, chunks={"line": 2, "doppler_velocity": 3, "logT": 2, "slit": 2, "detector_x_pixel": 4}
    )

    with _open(path, fmt) as source:
        chunks = source.detector_response.encoding["chunks" if fmt == "zarr" else "chunksizes"]
    assert chunks == (2, 3, 2, 2, 4)


def test_save_response_refuses_to_overwrite(tmp_path) -> None:
    path = tmp_path / "response.nc"
    path.touch()

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        save_response(fake_response(), path)


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("response_type", TypeError, "xarray.Dataset"),
        ("missing_variable", ValueError, "must contain detector_response"),
        ("missing_line_wavelength", ValueError, "line_wavelength is missing"),
        ("unitless_axis", ValueError, "logT must define units"),
        ("bad_suffix", ValueError, "must end in"),
    ],
)
def test_save_response_validates_the_response(tmp_path, case, error, match) -> None:
    response = fake_response()
    path = tmp_path / "response.nc"
    if case == "response_type":
        response = None
    elif case == "missing_variable":
        response = response.drop_vars("detector_response")
    elif case == "missing_line_wavelength":
        response = response.drop_vars("line_wavelength")
    elif case == "unitless_axis":
        response = response.assign_coords(logT=response.logT.assign_attrs({}))
        del response.logT.attrs["units"]
    else:
        path = tmp_path / "response.txt"

    with pytest.raises(error, match=match):
        save_response(response, path)


@pytest.mark.parametrize(
    ("chunks", "error", "match"),
    [
        ([], TypeError, "mapping"),
        ({"missing": 1}, ValueError, "unknown dimension"),
        ({"doppler_velocity": 0}, ValueError, "positive integer"),
    ],
)
def test_save_response_validates_chunk_overrides(tmp_path, chunks, error, match) -> None:
    with pytest.raises(error, match=match):
        save_response(_small_response(), tmp_path / "response.zarr", chunks=chunks)


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
def test_read_response_roundtrip_selects_axes(tmp_path, fmt) -> None:
    path = _write(fake_response_file(), tmp_path / f"resp.{fmt}", fmt)
    logT = _axis(np.linspace(5.2, 6.6, 4), "logT")
    doppler_velocity = _axis([-200.0, -100.0, 0.0, 100.0, 200.0], "doppler_velocity")

    r = read_response(path, logT=logT, doppler_velocity=doppler_velocity, slit=_slit(3), logT_method="nearest")

    assert isinstance(r, xr.Dataset)
    assert "detector_response" in r.data_vars
    assert r.sizes["logT"] == logT.size
    assert r.sizes["doppler_velocity"] == doppler_velocity.size
    assert r.sizes["slit"] == 3  # read_response selects np.arange(slit.max() + 1)
    np.testing.assert_allclose(r.logT.values, logT.values)
    np.testing.assert_allclose(r.doppler_velocity.values, doppler_velocity.values)
    # The file carries its wavelength units and the reader keeps them.
    assert r.line_wavelength.attrs["units"] == str(u.AA)
    assert r.detector_wavelength.attrs["units"] == str(u.AA)
    np.testing.assert_array_equal(
        r.gain.values, u.Quantity(DEFAULTS_MUSE.ccd_gain_SG.sel(channel=r.channel).data).to_value(u.electron / u.DN)
    )
    assert r.attrs["HISTORY"][-1].startswith("read_response(")


def test_read_response_normalizes_slit_labels(tmp_path) -> None:
    # Files with non-0-based slit numbering must come back 0-based so downstream
    # exact joins against reshape_x_to_slit_step output (slit = 0..n-1) align.
    src = fake_response_file()
    src = src.assign_coords(slit=src.slit.values + 1)
    expected = src.detector_response.isel(slit=slice(3)).values
    path = _write(src, tmp_path / "resp.nc", "nc")

    r = read_response(path, slit=_slit(3))

    np.testing.assert_array_equal(r.slit.values, np.arange(3))
    np.testing.assert_allclose(r.detector_response.values, expected)


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
@pytest.mark.parametrize(("dropped", "expected"), [("logT", "logT"), ("doppler_velocity", "doppler_velocity")])
def test_read_response_requires_the_resampling_coordinates(tmp_path, fmt, dropped, expected) -> None:
    # A response without logT or doppler_velocity cannot be resampled or contracted, so it must
    # be rejected on load rather than failing somewhere downstream.
    src = fake_response_file().isel({dropped: 0}, drop=True)
    path = _write(src, tmp_path / f"resp.{fmt}", fmt)

    with pytest.raises(ValueError, match=rf"response\.{expected} is missing"):
        read_response(path)


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
def test_read_response_chunked_stays_lazy_and_matches_eager(tmp_path, fmt) -> None:
    path = _write(fake_response_file(), tmp_path / f"resp.{fmt}", fmt)
    logT = _axis(np.linspace(5.2, 6.6, 4), "logT")
    doppler_velocity = _axis([-150.0, -50.0, 0.0, 50.0, 150.0], "doppler_velocity")
    kwargs = {
        "logT": logT,
        "doppler_velocity": doppler_velocity,
        "slit": _slit(3),
        "logT_method": "nearest",
        "doppler_velocity_method": "linear",
    }

    lazy = read_response(path, chunked=True, **kwargs)
    eager = read_response(path, chunked=False, **kwargs)

    assert isinstance(lazy.detector_response.data, da.Array)
    xr.testing.assert_allclose(lazy.compute(), eager)


def test_load_and_concat_responses_chunked_stays_lazy(tmp_path) -> None:
    for name in ("a.nc", "b.nc"):
        fake_response_file().to_netcdf(tmp_path / name)

    response = load_and_concat_responses(tmp_path, ["a.nc", "b.nc"], channels=[108, 171], chunked=True)

    assert isinstance(response.detector_response.data, da.Array)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("chunked", [False, True])
def test_read_response_reads_a_zipped_zarr_store(tmp_path, zarr_format, chunked) -> None:
    legacy = fake_response_file()
    expected = read_response(_write(legacy, tmp_path / "resp.nc", "nc"))
    legacy.to_zarr(tmp_path / "resp.zarr", zarr_format=zarr_format, consolidated=False)
    # The archives in use were made by zipping a store, so the entries sit at the archive root.
    shutil.make_archive(str(tmp_path / "resp.zarr"), "zip", root_dir=tmp_path / "resp.zarr")
    zipped = read_response(tmp_path / "resp.zarr.zip", chunked=chunked)
    if chunked:
        assert isinstance(zipped.detector_response.data, da.Array)
    np.testing.assert_allclose(zipped.detector_response.values, expected.detector_response.values)
    assert "detector_wavelength" in zipped.coords


def test_read_response_eager_zarr_is_not_dask_backed(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.zarr", "zarr")
    assert read_response(path).detector_response.chunks is None
    assert read_response(path, chunked=True).detector_response.chunks is not None


def test_read_response_resamples_in_the_response_axis_units(tmp_path) -> None:
    # The file's velocity axis is in m/s at -300e3..300e3; its samples sit every 100e3 m/s.
    src = fake_response_file()
    src = src.assign_coords(doppler_velocity=src.doppler_velocity * 1000.0)
    src.doppler_velocity.attrs["units"] = "m / s"
    path = _write(src, tmp_path / "resp.nc", "nc")

    plain = read_response(path, doppler_velocity=xr.DataArray([-100e3, 0.0, 100e3], dims="doppler_velocity"))
    np.testing.assert_array_equal(plain.doppler_velocity.values, [-100e3, 0.0, 100e3])
    assert plain.doppler_velocity.attrs["units"] == "m / s"  # a plain axis is read in the response's units

    in_km_s = xr.DataArray([-100.0, 0.0, 100.0], dims="doppler_velocity", attrs={"units": "km / s", "long_name": "v"})
    converted = read_response(path, doppler_velocity=in_km_s)
    np.testing.assert_array_equal(converted.doppler_velocity.values, [-100e3, 0.0, 100e3])
    assert converted.doppler_velocity.attrs["units"] == "m / s"  # converted into the response's units
    np.testing.assert_array_equal(converted.detector_response.values, plain.detector_response.values)

    with pytest.raises(ValueError, match="not convertible"):
        read_response(path, doppler_velocity=xr.DataArray([0.0], dims="doppler_velocity", attrs={"units": "K"}))


def test_read_response_opens_nonconsolidated_zarr3_without_fallback_warning(tmp_path) -> None:
    path = tmp_path / "response.zarr"
    fake_response_file().to_zarr(path, zarr_format=3, consolidated=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        response = read_response(path)

    assert "detector_response" in response


def test_read_response_linear_interp_hits_grid_and_stays_nonnegative(response_path) -> None:
    # A grid offset from the source logT forces real interpolation rather than nearest selection.
    logT = _axis(np.linspace(5.1, 6.9, 9), "logT")

    r = read_response(response_path, logT=logT, logT_method="linear")

    np.testing.assert_allclose(r.logT.values, logT.values)
    assert bool((r.detector_response >= 0).all())  # interp path clamps negatives to zero


def test_read_response_without_channel_requires_explicit_gain(tmp_path) -> None:
    path = _write(fake_response_file().drop_vars("channel"), tmp_path / "resp.zarr", "zarr")

    with pytest.raises(ValueError, match="pass gain explicitly"):
        read_response(path)


def test_read_response_rejects_missing_line_wavelength(tmp_path) -> None:
    path = _write(fake_response_file().drop_vars("line_wavelength"), tmp_path / "resp.zarr", "zarr")

    with pytest.raises(ValueError, match="line_wavelength is missing"):
        read_response(path)


def test_read_response_keeps_existing_wavelength_units(tmp_path) -> None:
    src = fake_response_file()
    src.line_wavelength.attrs["units"] = "nm"
    src.detector_wavelength.attrs["units"] = "nm"
    path = _write(src, tmp_path / "resp.zarr", "zarr")

    r = read_response(path)

    assert r.line_wavelength.attrs["units"] == "nm"  # present units left untouched
    assert r.detector_wavelength.attrs["units"] == "nm"


def test_read_response_gain_accepts_quantity(response_path) -> None:
    r = read_response(response_path, gain=5.0 * u.electron / u.DN)

    np.testing.assert_array_equal(r.gain.values, [5.0])
    assert r.gain.attrs["units"] == str(u.electron / u.DN)


def test_read_response_uses_channel_gain(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        DEFAULTS_MUSE.ccd_gain_SG,
        "data",
        np.array([8.0, 10.0, 12.0]) * u.electron / u.DN,
    )
    src = fake_response_file().assign_coords(channel=("line", [284]))
    path = _write(src, tmp_path / "resp.zarr", "zarr")

    r = read_response(path)

    np.testing.assert_array_equal(r.gain.values, [12.0])


def test_read_response_gain_rejects_wrong_units(response_path) -> None:
    with pytest.raises(u.UnitsError):
        read_response(response_path, gain=5.0 * u.second)


@pytest.mark.parametrize(
    ("logT", "match"),
    [
        ([], "must not be empty"),
        ([5.0, np.nan], "finite"),
        ([8.0, 8.5], "no overlap"),
    ],
)
def test_read_response_validates_the_logT_axis(response_path, logT, match) -> None:
    with pytest.raises(ValueError, match=match):
        read_response(response_path, logT=xr.DataArray(np.array(logT), dims="logT"))


def test_read_response_names_the_legacy_layout(tmp_path) -> None:
    legacy = fake_response_file().rename({"detector_response": "SG_resp"})
    path = _write(legacy, tmp_path / "resp.nc", "nc")
    with pytest.raises(ValueError, match="legacy MUSE names"):
        read_response(path)


def test_read_response_requires_detector_response(tmp_path) -> None:
    path = _write(fake_response_file().drop_vars("detector_response"), tmp_path / "resp.nc", "nc")
    with pytest.raises(ValueError, match="detector_response"):
        read_response(path)


def test_load_and_concat_responses_concatenates_lines(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        DEFAULTS_MUSE.ccd_gain_SG,
        "data",
        np.array([8.0, 10.0, 12.0]) * u.electron / u.DN,
    )
    first = xr.concat(
        [
            fake_response_file().assign_coords(
                line=("line", ["Fe XIX 108.355"]),
                line_wavelength=("line", [108.355], {"units": "Angstrom"}),
                channel=("line", [108]),
                component_kind=("line", ["line"]),
            ),
            fake_response_file().assign_coords(
                line=("line", ["Fe XXI 108.117"]),
                line_wavelength=("line", [108.117], {"units": "Angstrom"}),
                channel=("line", [108]),
                component_kind=("line", ["line"]),
            ),
            fake_response_file().assign_coords(
                line=("line", ["contaminants"]),
                line_wavelength=("line", [108.355], {"units": "Angstrom"}),
                channel=("line", [108]),
                component_kind=("line", ["contaminants"]),
            ),
        ],
        dim="line",
        data_vars="all",
        join="exact",
    ).assign_attrs(
        HISTORY="first_response()",
        shared_attr="kept",
        conflicting_attr="first",
    )
    second = (
        fake_response_file()
        .assign_coords(
            line=("line", ["Fe XV 284.163"]),
            line_wavelength=("line", [284.163], {"units": "Angstrom"}),
            channel=("line", [284]),
            component_kind=("line", ["line"]),
        )
        .assign_attrs(
            HISTORY="second_response()",
            shared_attr="kept",
            conflicting_attr="second",
        )
    )
    _write(first.drop_vars("channel"), tmp_path / "a.nc", "nc")
    _write(second.drop_vars("channel"), tmp_path / "b.nc", "nc")

    resp = load_and_concat_responses(
        response_directory=tmp_path,
        response_files=["a.nc", "b.nc"],
        logT=_axis(np.linspace(5.2, 6.6, 4), "logT"),
        doppler_velocity=_axis([-100.0, 0.0, 100.0], "doppler_velocity"),
        slit=_slit(3),
        logT_method="nearest",
        channels=[108, 284],
    )

    assert resp.sizes["line"] == 4
    np.testing.assert_array_equal(resp.line, ["Fe XIX 108.355", "Fe XXI 108.117", "contaminants", "Fe XV 284.163"])
    np.testing.assert_array_equal(resp.component_kind, ["line", "line", "contaminants", "line"])
    np.testing.assert_allclose(resp.line_wavelength, [108.355, 108.117, 108.355, 284.163])
    np.testing.assert_array_equal(resp.channel.values, [108, 108, 108, 284])
    np.testing.assert_array_equal(resp.gain.values, [8.0, 8.0, 8.0, 12.0])
    assert "effective_area" not in resp.data_vars  # dropped before concatenation
    assert "wavelength" not in resp.dims
    assert "detector_response" in resp.data_vars
    assert resp.attrs["shared_attr"] == "kept"
    assert "conflicting_attr" not in resp.attrs
    assert resp.attrs["HISTORY"][0] == "first_response()"
    assert "second_response()" in resp.attrs["HISTORY"]
    assert resp.attrs["HISTORY"][-1].startswith("load_and_concat_responses(")


def test_load_and_concat_responses_rejects_misaligned_grids(tmp_path) -> None:
    # join="exact" in the line concat must refuse responses on different
    # logT/doppler_velocity grids (when no target grid is given) instead of silently
    # outer-joining them with NaN fill.
    _write(fake_response_file(), tmp_path / "a.nc", "nc")
    shifted = fake_response_file()
    _write(shifted.assign_coords(doppler_velocity=shifted.doppler_velocity + 5.0), tmp_path / "b.nc", "nc")

    with pytest.raises(ValueError, match=r"align|exact"):
        load_and_concat_responses(
            response_directory=tmp_path,
            response_files=["a.nc", "b.nc"],
            channels=[108, 284],
        )


def test_load_and_concat_responses_channels_length_mismatch_raises(tmp_path) -> None:
    _write(fake_response_file(), tmp_path / "a.zarr", "zarr")
    with pytest.raises(ValueError, match=r"channels .* must match the number of response_files"):
        load_and_concat_responses(
            response_directory=tmp_path,
            response_files=["a.zarr"],
            channels=[171, 284],
        )


def test_read_response_selects_the_ci_gain_for_a_ci_channel(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(response_io, "DEFAULTS_MUSE", attrs.evolve(DEFAULTS_MUSE, ccd_gain_CI=_CI_GAIN))
    legacy = fake_response_file()
    path = _write(legacy.assign_coords(channel=("line", [195] * legacy.sizes["line"])), tmp_path / "ci.nc", "nc")
    np.testing.assert_allclose(read_response(path).gain.values, 7.0)


def test_load_and_concat_responses_selects_the_ci_gain_for_a_ci_channel(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(response_io, "DEFAULTS_MUSE", attrs.evolve(DEFAULTS_MUSE, ccd_gain_CI=_CI_GAIN))
    _write(fake_response_file(), tmp_path / "a.zarr", "zarr")
    response = load_and_concat_responses(tmp_path, ["a.zarr"], channels=[195])
    np.testing.assert_allclose(response.gain.values, 7.0)
    assert response.channel.values.tolist() == [195]


def test_load_and_concat_responses_rejects_unsupported_channel(tmp_path) -> None:
    _write(fake_response_file(), tmp_path / "a.zarr", "zarr")

    with pytest.raises(ValueError, match=r"unsupported MUSE channel\(s\) \[999\]"):
        load_and_concat_responses(tmp_path, ["a.zarr"], channels=[999])


def test_align_response_and_vdem_aligns_nonuniform_coarse_response() -> None:
    response = fake_response().isel(logT=[0, 2, 5, 6], doppler_velocity=[0, 2, 4, 8])
    vdem = fake_vdem()

    matched_response, matched_vdem = align_response_and_vdem(response, vdem)

    xr.testing.assert_equal(matched_response.logT, matched_vdem.logT)
    xr.testing.assert_equal(matched_response.doppler_velocity, matched_vdem.doppler_velocity)
    assert bool(np.isfinite(matched_response.detector_response).all())


def test_align_response_and_vdem_honors_custom_coord_methods() -> None:
    response = fake_response().isel(logT=[0, 2, 5, 6], doppler_velocity=[0, 2, 4, 8])
    vdem = fake_vdem()
    nearest = {"logT": ("nearest", u.dex(u.K)), "doppler_velocity": ("nearest", u.km / u.s)}

    nearest_response, nearest_vdem = align_response_and_vdem(response, vdem, coord_methods=nearest)
    default_response, _ = align_response_and_vdem(response, vdem)

    xr.testing.assert_equal(nearest_response.logT, nearest_vdem.logT)
    xr.testing.assert_equal(nearest_response.doppler_velocity, nearest_vdem.doppler_velocity)
    # Nearest-neighbour velocity resampling must differ from the default linear interpolation.
    assert not np.allclose(nearest_response.detector_response, default_response.detector_response)


def test_align_response_and_vdem_trims_both_outputs_to_overlap() -> None:
    response = fake_response().isel(logT=slice(1, -1), doppler_velocity=slice(1, -1))
    vdem = fake_vdem()

    matched_response, matched_vdem = align_response_and_vdem(response, vdem)

    assert matched_vdem.sizes["logT"] == vdem.sizes["logT"] - 2
    assert matched_vdem.sizes["doppler_velocity"] == vdem.sizes["doppler_velocity"] - 2
    xr.testing.assert_equal(matched_response.logT, matched_vdem.logT)
    xr.testing.assert_equal(matched_response.doppler_velocity, matched_vdem.doppler_velocity)


def test_align_response_and_vdem_normalizes_equivalent_velocity_units() -> None:
    response = fake_response()
    vdem = fake_vdem()
    velocity = xr.DataArray(
        vdem.doppler_velocity.values * 1000,
        dims="doppler_velocity",
        attrs={"units": "m/s"},
    )
    vdem = vdem.assign_coords(doppler_velocity=velocity)

    matched_response, matched_vdem = align_response_and_vdem(response, vdem)

    xr.testing.assert_equal(matched_response.doppler_velocity, matched_vdem.doppler_velocity)
    assert matched_vdem.doppler_velocity.attrs["units"] == str(u.km / u.s)
    assert matched_vdem.sizes["doppler_velocity"] == vdem.sizes["doppler_velocity"]


def test_align_response_and_vdem_preserves_inputs_lineage_and_laziness() -> None:
    response = fake_response().assign_attrs(HISTORY=["response()"]).chunk({"logT": 2})
    vdem = fake_vdem().assign_attrs(HISTORY=["vdem()"]).chunk({"logT": 2})
    response_before = response.copy()
    vdem_before = vdem.copy()

    matched_response, matched_vdem = align_response_and_vdem(response, vdem)

    assert isinstance(matched_response.detector_response.data, da.Array)
    assert isinstance(matched_vdem.vdem.data, da.Array)
    xr.testing.assert_identical(response, response_before)
    xr.testing.assert_identical(vdem, vdem_before)
    assert matched_response.attrs["HISTORY"][:2] == ["response()", "vdem()"]
    assert matched_vdem.attrs["HISTORY"][:2] == ["response()", "vdem()"]
    assert matched_response.attrs["HISTORY"][-1].startswith("align_response_and_vdem(")
    assert matched_vdem.attrs["HISTORY"][-1].startswith("align_response_and_vdem(")


@pytest.mark.parametrize(("dataset", "name"), [("response", "logT"), ("vdem", "doppler_velocity")])
def test_align_response_and_vdem_requires_coordinates(dataset, name) -> None:
    response = fake_response()
    vdem = fake_vdem()
    if dataset == "response":
        response = response.drop_vars(name)
    else:
        vdem = vdem.drop_vars(name)

    with pytest.raises(ValueError, match=rf"{dataset}\.{name} is missing"):
        align_response_and_vdem(response, vdem)


def test_align_response_and_vdem_requires_coordinate_units() -> None:
    response = fake_response()
    del response.doppler_velocity.attrs["units"]

    with pytest.raises(ValueError, match=r"response\.doppler_velocity must define units"):
        align_response_and_vdem(response, fake_vdem())


def test_align_response_and_vdem_rejects_nonoverlapping_coordinates() -> None:
    response = fake_response()
    response = response.assign_coords(logT=response.logT + 10)

    with pytest.raises(ValueError, match="Requested logT axis has no overlap with the response range"):
        align_response_and_vdem(response, fake_vdem())
