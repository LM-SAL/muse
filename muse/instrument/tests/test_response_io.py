import shutil

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.instrument import align_response_and_vdem
from muse.instrument.response.io import read_response, read_responses, save_response, save_spectral_response
from muse.tests.helpers import fake_response, fake_response_file_contents, fake_vdem

pytestmark = [
    pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning"),
    pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning"),
    pytest.mark.filterwarnings("ignore:numpy.ndarray size changed:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning"),
]


def _write(ds: xr.Dataset, path, fmt: str) -> str:
    if fmt == "nc":
        ds.to_netcdf(path)
    else:
        ds.to_zarr(path)
    return str(path)


def _axis(values, name: str, unit: str) -> xr.DataArray:
    values = np.asarray(values, dtype=float)
    return xr.DataArray(values, dims=name, coords={name: values}, attrs={"units": unit})


def _open(path, fmt: str) -> xr.Dataset:
    if fmt == "zarr":
        return xr.open_zarr(path, consolidated=False)
    return xr.open_dataset(path)


@pytest.fixture
def response_path(tmp_path):
    # For tests that read an unmodified response file, whatever its on-disk format.
    return _write(fake_response_file_contents(), tmp_path / "resp.nc", "nc")


def _small_response() -> xr.Dataset:
    response = fake_response().isel(line=slice(0, 2), logT=slice(0, 3), slit=slice(0, 4))
    return response.assign_attrs(
        normalization=1e-27, gain="10.0 electron / DN", history=["create response", "map response"]
    )


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
            configuration = encoding["compressors"][0].to_dict()["configuration"]
            assert {name: configuration[name] for name in ("cname", "clevel", "shuffle")} == {
                "cname": "zstd",
                "clevel": 3,
                "shuffle": "bitshuffle",
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


@pytest.mark.parametrize("fmt", ["nc", "zarr", "zarr.zip"])
def test_save_response_then_read_response_round_trips(tmp_path, fmt) -> None:
    # The save tests read back with the raw opener and the read tests are fed raw to_netcdf
    # output, so nothing checked that what save_response writes is what read_response accepts.
    response = _small_response()
    path = tmp_path / f"response.{fmt}"

    save_response(response, path)
    loaded = read_response(path)

    # a zipped store is one file, with nothing left beside it
    assert [found.name for found in tmp_path.iterdir()] == [path.name]
    assert path.is_dir() == (fmt == "zarr")
    # read_response adds a gain coordinate of its own, so compare the data rather than the
    # whole object.
    assert loaded.detector_response.dims == response.detector_response.dims
    np.testing.assert_allclose(loaded.detector_response.values, response.detector_response.values)
    np.testing.assert_allclose(loaded.line_wavelength.values, response.line_wavelength.values)


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
    made = {"date_created": "2026-01-01T00:00:00Z", "muse_version": "0.1"}
    source = fake_response_file_contents().assign_attrs(made)
    path = _write(source, tmp_path / f"resp.{fmt}", fmt)
    logT = _axis(np.linspace(5.2, 6.6, 4), "logT", "dex(K)")
    doppler_velocity = _axis([-200.0, -100.0, 0.0, 100.0, 200.0], "doppler_velocity", "km / s")

    r = read_response(path, logT=logT, doppler_velocity=doppler_velocity, number_of_slits=3, logT_method="nearest")

    assert r.sizes["slit"] == 3  # the first number_of_slits slits
    np.testing.assert_allclose(r.logT.values, logT.values)
    np.testing.assert_allclose(r.doppler_velocity.values, doppler_velocity.values)
    # The data are the samples nearest to the requested axes, not the file's samples relabelled.
    expected = (
        source.detector_response.sel(logT=logT.values, method="nearest")
        .sel(doppler_velocity=doppler_velocity.values)
        .isel(slit=slice(3))
    )
    np.testing.assert_allclose(r.detector_response.transpose(*expected.dims).values, expected.values)
    assert r.attrs["history"][-1].startswith("read_response(")
    # reading dates nothing: the product keeps when and by which muse it was made
    assert {key: r.attrs[key] for key in made} == made
    assert "date_modified" not in r.attrs


def test_read_response_reads_the_provenance_of_a_file_of_muse_0_3(tmp_path) -> None:
    old = {
        "HISTORY": ["create response"],
        "date created": "01-Mar-2026",
        "version": "0.3.0",
        "chunked": 0,
        "logT_method": "nearest",
    }
    path = _write(fake_response_file_contents().assign_attrs(old), tmp_path / "old.nc", "nc")

    r = read_response(path)

    assert r.attrs["history"][0] == "create response"
    assert (r.attrs["date_created"], r.attrs["muse_version"]) == ("2026-03-01", "0.3.0")
    assert not {"HISTORY", "date created", "version", "chunked", "logT_method"} & set(r.attrs)


def test_read_response_normalizes_slit_labels(tmp_path) -> None:
    # Files with non-0-based slit numbering must come back 0-based so downstream
    # exact joins against reshape_x_to_slit_step output (slit = 0..n-1) align.
    src = fake_response_file_contents()
    src = src.assign_coords(slit=src.slit.values + 1)
    expected = src.detector_response.isel(slit=slice(3)).values
    path = _write(src, tmp_path / "resp.nc", "nc")

    r = read_response(path, number_of_slits=3)

    np.testing.assert_array_equal(r.slit.values, np.arange(3))
    np.testing.assert_allclose(r.detector_response.values, expected)
    for wrong in (0, 5):
        with pytest.raises(ValueError, match="between 1 and the 4 slits"):
            read_response(path, number_of_slits=wrong)
    with pytest.raises(TypeError, match="must be an integer"):
        read_response(path, number_of_slits=True)


@pytest.mark.parametrize("dropped", ["logT", "doppler_velocity"])
def test_read_response_requires_the_resampling_coordinates(tmp_path, dropped) -> None:
    # A response without logT or doppler_velocity cannot be resampled or contracted, so it must
    # be rejected on load rather than failing somewhere downstream.
    src = fake_response_file_contents().isel({dropped: 0}, drop=True)
    path = _write(src, tmp_path / "resp.nc", "nc")

    with pytest.raises(ValueError, match=rf"response\.{dropped} is missing"):
        read_response(path)


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
def test_read_response_chunked_stays_lazy_and_matches_eager(tmp_path, fmt) -> None:
    path = _write(fake_response_file_contents(), tmp_path / f"resp.{fmt}", fmt)
    logT = _axis(np.linspace(5.2, 6.6, 4), "logT", "dex(K)")
    doppler_velocity = _axis([-150.0, -50.0, 0.0, 50.0, 150.0], "doppler_velocity", "km / s")
    kwargs = {
        "logT": logT,
        "doppler_velocity": doppler_velocity,
        "number_of_slits": 3,
        "logT_method": "nearest",
        "doppler_velocity_method": "linear",
    }

    lazy = read_response(path, chunked=True, **kwargs)
    eager = read_response(path, chunked=False, **kwargs)

    assert isinstance(lazy.detector_response.data, da.Array)
    xr.testing.assert_allclose(lazy.compute(), eager)


def test_read_responses_chunked_stays_lazy(tmp_path) -> None:
    for name in ("a.nc", "b.nc"):
        fake_response_file_contents().to_netcdf(tmp_path / name)

    response = read_responses([tmp_path / "a.nc", tmp_path / "b.nc"], chunked=True)

    assert isinstance(response.detector_response.data, da.Array)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("chunked", [False, True])
def test_read_response_reads_a_zipped_zarr_store(tmp_path, zarr_format, chunked) -> None:
    source = fake_response_file_contents()
    expected = read_response(_write(source, tmp_path / "resp.nc", "nc"))
    source.to_zarr(tmp_path / "resp.zarr", zarr_format=zarr_format, consolidated=False)
    # The archives in use were made by zipping a store, so the entries sit at the archive root.
    shutil.make_archive(str(tmp_path / "resp.zarr"), "zip", root_dir=tmp_path / "resp.zarr")
    zipped = read_response(tmp_path / "resp.zarr.zip", chunked=chunked)
    if chunked:
        assert isinstance(zipped.detector_response.data, da.Array)
    np.testing.assert_allclose(zipped.detector_response.values, expected.detector_response.values)
    assert "detector_wavelength" in zipped.coords


def test_read_response_eager_zarr_is_not_dask_backed(tmp_path) -> None:
    path = _write(fake_response_file_contents(), tmp_path / "resp.zarr", "zarr")
    assert read_response(path).detector_response.chunks is None
    assert read_response(path, chunked=True).detector_response.chunks is not None


def test_read_response_resamples_in_the_response_axis_units(tmp_path) -> None:
    # The file's velocity axis is in m/s at -300e3..300e3; its samples sit every 100e3 m/s.
    src = fake_response_file_contents()
    src = src.assign_coords(doppler_velocity=src.doppler_velocity * 1000.0)
    src.doppler_velocity.attrs["units"] = "m / s"
    path = _write(src, tmp_path / "resp.nc", "nc")

    in_m_s = read_response(path, doppler_velocity=[-100e3, 0.0, 100e3] * u.m / u.s)
    np.testing.assert_array_equal(in_m_s.doppler_velocity.values, [-100e3, 0.0, 100e3])

    in_km_s = xr.DataArray([-100.0, 0.0, 100.0], dims="doppler_velocity", attrs={"units": "km / s", "long_name": "v"})
    converted = read_response(path, doppler_velocity=in_km_s)
    np.testing.assert_array_equal(converted.doppler_velocity.values, [-100e3, 0.0, 100e3])
    assert converted.doppler_velocity.attrs["units"] == "m / s"  # converted into the response's units
    np.testing.assert_array_equal(converted.detector_response.values, in_m_s.detector_response.values)

    with pytest.raises(ValueError, match="not convertible"):
        read_response(path, doppler_velocity=[0.0] * u.K)
    # The axes of a response always carry units, so a requested axis must too.
    with pytest.raises(ValueError, match="units attribute"):
        read_response(path, doppler_velocity=xr.DataArray([0.0], dims="doppler_velocity"))


def test_read_response_without_a_gain_attribute_requires_gain(tmp_path) -> None:
    source = fake_response_file_contents()
    del source.attrs["gain"]
    path = _write(source, tmp_path / "resp.zarr", "zarr")

    with pytest.raises(ValueError, match="states no gain, in a gain coordinate or attribute; pass gain"):
        read_response(path)


def test_read_response_rejects_missing_line_wavelength(tmp_path) -> None:
    path = _write(fake_response_file_contents().drop_vars("line_wavelength"), tmp_path / "resp.zarr", "zarr")

    with pytest.raises(ValueError, match="line_wavelength is missing"):
        read_response(path)


def test_read_response_keeps_existing_wavelength_units(tmp_path) -> None:
    src = fake_response_file_contents()
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


def test_read_response_takes_the_gain_of_the_file(tmp_path) -> None:
    path = _write(fake_response_file_contents().assign_attrs(gain="12.0 electron / DN"), tmp_path / "resp.zarr", "zarr")

    np.testing.assert_array_equal(read_response(path).gain.values, [12.0])


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
        read_response(response_path, logT=np.array(logT, dtype=float) * u.dex(u.K))


def test_read_response_names_the_legacy_layout(tmp_path) -> None:
    legacy = fake_response_file_contents().rename({"detector_response": "SG_resp"})
    path = _write(legacy, tmp_path / "resp.nc", "nc")
    with pytest.raises(ValueError, match="legacy MUSE names"):
        read_response(path)


def _line_file(line: str, wavelength: float, channel: int) -> xr.Dataset:
    # A response file of one line, with its channel.
    return fake_response_file_contents().assign_coords(
        line=("line", [line]),
        line_wavelength=("line", [wavelength], {"units": "Angstrom"}),
        channel=("line", [channel]),
    )


def test_read_responses_concatenates_lines(tmp_path) -> None:
    first = xr.concat(
        [
            _line_file("Fe XIX 108.355", 108.355, 108),
            _line_file("Fe XXI 108.117", 108.117, 108),
            _line_file("contaminants", 108.355, 108),
        ],
        dim="line",
        data_vars="all",
        join="exact",
    ).assign_attrs(
        gain="8.0 electron / DN",
        history="first_response()",
        shared_attr="kept",
        conflicting_attr="first",
        muse_version="0.4.0",
        date_created="2026-01-01T00:00:00Z",
    )
    second = _line_file("Fe XV 284.163", 284.163, 284).assign_attrs(
        gain="12.0 electron / DN",
        history="second_response()",
        shared_attr="kept",
        conflicting_attr="second",
        muse_version="0.4.0",
        date_created="2026-02-01T00:00:00Z",
    )
    _write(first, tmp_path / "a.nc", "nc")
    _write(second, tmp_path / "b.nc", "nc")

    resp = read_responses(
        [tmp_path / "a.nc", tmp_path / "b.nc"],
        logT=np.linspace(5.2, 6.6, 4) * u.dex(u.K),
        doppler_velocity=[-100.0, 0.0, 100.0] * u.km / u.s,
        number_of_slits=3,
        logT_method="nearest",
    )

    np.testing.assert_array_equal(resp.line, ["Fe XIX 108.355", "Fe XXI 108.117", "contaminants", "Fe XV 284.163"])
    np.testing.assert_allclose(resp.line_wavelength, [108.355, 108.117, 108.355, 284.163])
    np.testing.assert_array_equal(resp.channel.values, [108, 108, 108, 284])
    np.testing.assert_array_equal(resp.gain.values, [8.0, 8.0, 8.0, 12.0])
    np.testing.assert_array_equal(resp.slit.values, [0, 1, 2])  # number_of_slits reaches every file
    assert "effective_area" not in resp.data_vars  # dropped before concatenation
    assert "wavelength" not in resp.dims
    assert resp.attrs["shared_attr"] == "kept"
    assert "conflicting_attr" not in resp.attrs
    assert resp.attrs["history"][0] == "first_response()"
    assert "second_response()" in resp.attrs["history"]
    assert resp.attrs["history"][-1].startswith("read_responses(")
    # what the files agree on, their muse included, and not a date they do not share, nor the read's
    assert resp.attrs["muse_version"] == "0.4.0"
    assert "date_created" not in resp.attrs


def test_read_responses_rejects_misaligned_grids(tmp_path) -> None:
    # join="exact" in the line concat must refuse responses on different
    # logT/doppler_velocity grids (when no target grid is given) instead of silently
    # outer-joining them with NaN fill.
    _write(fake_response_file_contents(), tmp_path / "a.nc", "nc")
    shifted = fake_response_file_contents()
    _write(shifted.assign_coords(doppler_velocity=shifted.doppler_velocity + 5.0), tmp_path / "b.nc", "nc")

    with pytest.raises(ValueError, match=r"align|exact"):
        read_responses([tmp_path / "a.nc", tmp_path / "b.nc"])


def test_read_response_keeps_a_gain_per_line_and_replaces_the_one_of_the_file_with_a_given_one(tmp_path) -> None:
    per_line = fake_response_file_contents().assign_coords(gain=("line", [12.0], {"units": "electron / DN"}))
    del per_line.attrs["gain"]
    given = read_response(
        _write(fake_response_file_contents(), tmp_path / "b.zarr", "zarr"), gain=[11.0] * u.electron / u.DN
    )

    assert read_response(_write(per_line, tmp_path / "a.zarr", "zarr")).gain.values.tolist() == [12.0]
    assert given.gain.values.tolist() == [11.0]
    assert "gain" not in given.attrs  # not the 10 of the file


def test_read_responses_reads_files_of_one_run_and_one_unit(tmp_path) -> None:
    one_channel = fake_response_file_contents().assign_coords(channel=171)  # one channel for every line
    paths = [_write(one_channel, tmp_path / "a.nc", "nc")]
    paths.append(
        _write(fake_response_file_contents().assign_attrs(instrument_defaults="custom 1"), tmp_path / "b.nc", "nc")
    )
    in_dn = fake_response_file_contents()
    in_dn.detector_response.attrs["units"] = "1e-27 cm5 DN / s"
    paths.append(_write(in_dn, tmp_path / "c.nc", "nc"))

    assert read_responses([paths[0]]).channel.dims == ("line",)
    with pytest.raises(ValueError, match="different instrument defaults"):
        read_responses(paths[:2])
    with pytest.raises(ValueError, match="different units"):
        read_responses([paths[0], paths[2]])


def test_read_responses_takes_a_gain_coordinate_in_electron_per_dn_and_wants_a_file(tmp_path) -> None:
    in_kilo = fake_response_file_contents().assign_coords(gain=("line", [0.012], {"units": "1000 electron / DN"}))

    assert read_responses([_write(in_kilo, tmp_path / "a.zarr", "zarr")]).gain.values.tolist() == [12.0]
    with pytest.raises(ValueError, match="pass at least one response file"):
        read_responses([])


def test_read_responses_refuses_a_file_without_a_channel(tmp_path) -> None:
    path = _write(fake_response_file_contents().drop_vars("channel"), tmp_path / "a.zarr", "zarr")

    with pytest.raises(ValueError, match="has no channel coordinate"):
        read_responses([path])


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

    nearest_response, nearest_vdem = align_response_and_vdem(response, vdem, coordinate_methods=nearest)
    default_response, _ = align_response_and_vdem(response, vdem)

    xr.testing.assert_equal(nearest_response.logT, nearest_vdem.logT)
    xr.testing.assert_equal(nearest_response.doppler_velocity, nearest_vdem.doppler_velocity)
    # Nearest-neighbor velocity resampling must differ from the default linear interpolation.
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
    response = fake_response().assign_attrs(history=["response()"]).chunk({"logT": 2})
    vdem = fake_vdem().assign_attrs(history=["vdem()"]).chunk({"logT": 2})
    response_before = response.copy()
    vdem_before = vdem.copy()

    matched_response, matched_vdem = align_response_and_vdem(response, vdem)

    assert isinstance(matched_response.detector_response.data, da.Array)
    assert isinstance(matched_vdem.vdem.data, da.Array)
    xr.testing.assert_identical(response, response_before)
    xr.testing.assert_identical(vdem, vdem_before)
    assert matched_response.attrs["history"][:2] == ["response()", "vdem()"]
    assert matched_vdem.attrs["history"][:2] == ["response()", "vdem()"]
    assert matched_response.attrs["history"][-1].startswith("align_response_and_vdem(")
    assert matched_vdem.attrs["history"][-1].startswith("align_response_and_vdem(")


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


def test_save_spectral_response_validates_the_response_and_the_destination(tmp_path):
    # The round trip itself is in test_response.py, with a response from create_spectral_response.
    response = xr.Dataset(
        {"spectral_response": (("line", "wavelength_bin"), np.ones((1, 2)))},
        coords={"wavelength_grid": ("wavelength_bin", [170.0, 170.1]), "line_wavelength": ("line", [171.0])},
    )
    existing = tmp_path / "cube.nc"
    existing.touch()
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        save_spectral_response(response, existing)
    with pytest.raises(ValueError, match="missing required variables"):
        save_spectral_response(response.drop_vars("line_wavelength"), tmp_path / "other.nc")
    with pytest.raises(ValueError, match="end in"):
        save_spectral_response(response, tmp_path / "cube.txt")
