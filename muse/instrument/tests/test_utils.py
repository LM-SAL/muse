import warnings

import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.instrument.utils import load_and_concat_responses, read_response, save_response
from muse.tests.helpers import fake_response, fake_response_file
from muse.variables import DEFAULTS_MUSE

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


def _axis(values, name: str) -> xr.DataArray:
    values = np.asarray(values, dtype=float)
    return xr.DataArray(values, dims=name, coords={name: values})


def _slit(n: int) -> xr.DataArray:
    return xr.DataArray(np.arange(n), dims="slit", coords={"slit": np.arange(n)})


def _open(path, fmt: str) -> xr.Dataset:
    if fmt == "zarr":
        return xr.open_zarr(path, consolidated=False)
    return xr.open_dataset(path)


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
    assert chunks == (1, min(20, response.sizes["vdop"]), 1, 4, response.sizes["detector_x_pixel"])
    xr.testing.assert_identical(loaded, before)
    xr.testing.assert_identical(response, before)


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
def test_save_response_accepts_chunk_overrides(tmp_path, fmt) -> None:
    response = _small_response()
    path = tmp_path / f"response.{fmt}"

    save_response(response, path, chunks={"line": 2, "vdop": 3, "logT": 2, "slit": 2, "detector_x_pixel": 4})

    with _open(path, fmt) as source:
        chunks = source.detector_response.encoding["chunks" if fmt == "zarr" else "chunksizes"]
    assert chunks == (2, 3, 2, 2, 4)


def test_save_response_refuses_to_overwrite(tmp_path) -> None:
    path = tmp_path / "response.nc"
    path.touch()

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        save_response(fake_response(), path)


@pytest.mark.parametrize(
    ("chunks", "error", "match"),
    [
        ([], TypeError, "mapping"),
        ({"missing": 1}, ValueError, "unknown dimension"),
        ({"vdop": 0}, ValueError, "positive integer"),
    ],
)
def test_save_response_validates_chunk_overrides(tmp_path, chunks, error, match) -> None:
    with pytest.raises(error, match=match):
        save_response(_small_response(), tmp_path / "response.zarr", chunks=chunks)


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
def test_read_response_roundtrip_selects_axes(tmp_path, fmt) -> None:
    path = _write(fake_response_file(), tmp_path / f"resp.{fmt}", fmt)
    logT = _axis(np.linspace(5.2, 6.6, 4), "logT")
    vdop = _axis([-200.0, -100.0, 0.0, 100.0, 200.0], "vdop")

    r = read_response(path, logT=logT, vdop=vdop, slit=_slit(3), logT_method="nearest")

    assert isinstance(r, xr.Dataset)
    assert "detector_response" in r.data_vars
    assert r.sizes["logT"] == logT.size
    assert r.sizes["vdop"] == vdop.size
    assert r.sizes["slit"] == 3  # read_response selects np.arange(slit.max() + 1)
    np.testing.assert_allclose(r.logT.values, logT.values)
    np.testing.assert_allclose(r.vdop.values, vdop.values)
    # The reader injects Angstrom because the on-disk files carry no wavelength units (for now).
    assert r.line_wavelength.attrs["units"] == str(u.AA)
    assert r.detector_wavelength.attrs["units"] == str(u.AA)
    assert not {"SG_resp", "SG_wvl", "SG_xpixel", "line_wvl"} & set(r.variables)
    np.testing.assert_array_equal(r.gain.values, [DEFAULTS_MUSE.ccd_gain.to_value(u.electron / u.DN)])
    assert r.attrs["HISTORY"][-1].startswith("read_response(")


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
def test_read_response_without_axes_returns_full_resolution(tmp_path, fmt) -> None:
    src = fake_response_file()
    path = _write(src, tmp_path / f"resp.{fmt}", fmt)

    r = read_response(path)

    assert r.sizes["logT"] == src.sizes["logT"]
    assert r.sizes["vdop"] == src.sizes["vdop"]
    assert r.line_wavelength.attrs["units"] == str(u.AA)
    assert "gain" in r.coords


def test_read_response_opens_nonconsolidated_zarr3_without_fallback_warning(tmp_path) -> None:
    path = tmp_path / "response.zarr"
    fake_response_file().to_zarr(path, zarr_format=3, consolidated=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        response = read_response(path)

    assert "detector_response" in response


def test_read_response_linear_interp_hits_grid_and_stays_nonnegative(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.zarr", "zarr")
    # A grid offset from the source logT forces real interpolation rather than nearest selection.
    logT = _axis(np.linspace(5.1, 6.9, 9), "logT")

    r = read_response(path, logT=logT, logT_method="linear")

    np.testing.assert_allclose(r.logT.values, logT.values)
    assert bool((r.detector_response >= 0).all())  # interp path clamps negatives to zero


def test_read_response_expands_line_dim_and_fills_line_wavelength_from_attr(tmp_path) -> None:
    # Drop the line dimension and line_wvl to hit the expand_dims + attribute-fallback branches.
    src = fake_response_file().isel(line=0).drop_vars(["line", "line_wvl", "channel"])
    src.attrs["MAIN_LINE_WVL"] = 171.073
    path = _write(src, tmp_path / "resp.zarr", "zarr")

    r = read_response(path)

    assert "line" in r.dims
    assert float(r.line_wavelength) == pytest.approx(171.073)
    assert r.line_wavelength.attrs["units"] == str(u.AA)


def test_read_response_warns_on_missing_wavelength_units(tmp_path, caplog) -> None:
    # The fixture mirrors the real files, which carry no units on line_wvl/SG_wvl.
    path = _write(fake_response_file(), tmp_path / "resp.zarr", "zarr")

    r = read_response(path)

    assert "missing the 'units' attribute" in caplog.text
    assert r.line_wavelength.attrs["units"] == str(u.AA)  # Angstrom assumed for now
    assert r.detector_wavelength.attrs["units"] == str(u.AA)


def test_read_response_keeps_existing_wavelength_units(tmp_path, caplog) -> None:
    src = fake_response_file()
    src.line_wvl.attrs["units"] = "nm"
    src.SG_wvl.attrs["units"] = "nm"
    path = _write(src, tmp_path / "resp.zarr", "zarr")

    r = read_response(path)

    assert "missing the 'units' attribute" not in caplog.text
    assert r.line_wavelength.attrs["units"] == "nm"  # present units left untouched
    assert r.detector_wavelength.attrs["units"] == "nm"


def test_read_response_prefers_legacy_line_wvl_when_both_names_exist(tmp_path) -> None:
    src = fake_response_file().assign_coords(
        line_wavelength=("line", [999.0], {"units": "Angstrom"}),
    )
    path = _write(src, tmp_path / "resp.nc", "nc")

    r = read_response(path)

    np.testing.assert_allclose(r.line_wavelength, [171.073])
    assert "line_wvl" not in r


def test_read_response_gain_accepts_quantity(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.zarr", "zarr")

    r = read_response(path, gain=5.0 * u.electron / u.DN)

    np.testing.assert_array_equal(r.gain.values, [5.0])
    assert r.gain.attrs["units"] == str(u.electron / u.DN)


def test_read_response_gain_rejects_wrong_units(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.zarr", "zarr")
    with pytest.raises(u.UnitsError):
        read_response(path, gain=5.0 * u.second)


def test_read_response_empty_logT_raises(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.nc", "nc")
    with pytest.raises(ValueError, match="must not be empty"):
        read_response(path, logT=xr.DataArray(np.array([]), dims="logT"))


def test_read_response_nonfinite_logT_raises(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.nc", "nc")
    with pytest.raises(ValueError, match="finite"):
        read_response(path, logT=xr.DataArray(np.array([5.0, np.nan]), dims="logT"))


def test_read_response_out_of_range_logT_raises(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.nc", "nc")
    with pytest.raises(ValueError, match="no overlap"):
        read_response(path, logT=xr.DataArray(np.array([8.0, 8.5]), dims="logT"))


def test_read_response_requires_detector_response(tmp_path) -> None:
    path = _write(fake_response_file().drop_vars("SG_resp"), tmp_path / "resp.nc", "nc")
    with pytest.raises(ValueError, match="detector_response"):
        read_response(path)


def test_read_response_requires_line_wavelength_source(tmp_path) -> None:
    src = fake_response_file().drop_vars(["line_wvl", "channel"])
    path = _write(src, tmp_path / "resp.nc", "nc")

    with pytest.raises(ValueError, match="line_wavelength"):
        read_response(path)


def test_load_and_concat_responses_concatenates_lines(tmp_path) -> None:
    first = xr.concat(
        [
            fake_response_file().assign_coords(
                line=("line", ["Fe XIX 108.355"]),
                line_wvl=("line", [108.355]),
                channel=("line", [108]),
                component_kind=("line", ["line"]),
            ),
            fake_response_file().assign_coords(
                line=("line", ["Fe XXI 108.117"]),
                line_wvl=("line", [108.117]),
                channel=("line", [108]),
                component_kind=("line", ["line"]),
            ),
            fake_response_file().assign_coords(
                line=("line", ["contaminants"]),
                line_wvl=("line", [108.355]),
                channel=("line", [108]),
                component_kind=("line", ["contaminants"]),
            ),
        ],
        dim="line",
        data_vars="all",
        join="exact",
    )
    second = fake_response_file().assign_coords(
        line=("line", ["Fe XV 284.163"]),
        line_wvl=("line", [284.163]),
        channel=("line", [284]),
        component_kind=("line", ["line"]),
    )
    _write(first, tmp_path / "a.nc", "nc")
    _write(second, tmp_path / "b.nc", "nc")

    resp = load_and_concat_responses(
        response_directory=tmp_path,
        response_files=["a.nc", "b.nc"],
        logT=_axis(np.linspace(5.2, 6.6, 4), "logT"),
        vdop=_axis([-100.0, 0.0, 100.0], "vdop"),
        slit=_slit(3),
        logT_method="nearest",
        channels=[108, 284],
    )

    assert resp.sizes["line"] == 4
    np.testing.assert_array_equal(resp.line, ["Fe XIX 108.355", "Fe XXI 108.117", "contaminants", "Fe XV 284.163"])
    np.testing.assert_array_equal(resp.component_kind, ["line", "line", "contaminants", "line"])
    np.testing.assert_allclose(resp.line_wavelength, [108.355, 108.117, 108.355, 284.163])
    np.testing.assert_array_equal(resp.channel.values, [108, 108, 108, 284])
    np.testing.assert_array_equal(resp.gain.values, np.full(4, DEFAULTS_MUSE.ccd_gain.value))
    assert "effective_area" not in resp.data_vars  # dropped before concatenation
    assert "wavelength" not in resp.dims
    assert "detector_response" in resp.data_vars


def test_load_and_concat_responses_rejects_misaligned_grids(tmp_path) -> None:
    # join="exact" in the line concat must refuse responses on different
    # logT/vdop grids (when no target grid is given) instead of silently
    # outer-joining them with NaN fill.
    _write(fake_response_file(), tmp_path / "a.nc", "nc")
    shifted = fake_response_file()
    _write(shifted.assign_coords(vdop=shifted.vdop + 5.0), tmp_path / "b.nc", "nc")

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
