import warnings

import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.instrument.utils import load_and_concat_responses, read_response
from muse.tests.helpers import fake_response_file
from muse.variables import DEFAULTS_MUSE

DEFAULT_GAIN = DEFAULTS_MUSE.ccd_gain.to_value(u.electron / u.DN)


def _write(ds: xr.Dataset, path, fmt: str) -> str:
    # Write-time warnings come from netCDF4/Zarr string-coordinate encoding, not the reader;
    # the read path is exercised under the suite-wide ``filterwarnings = error``.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
def test_read_response_roundtrip_selects_axes(tmp_path, fmt) -> None:
    path = _write(fake_response_file(), tmp_path / f"resp.{fmt}", fmt)
    logT = _axis(np.linspace(5.2, 6.6, 4), "logT")
    vdop = _axis([-200.0, -100.0, 0.0, 100.0, 200.0], "vdop")

    r = read_response(path, logT=logT, vdop=vdop, slit=_slit(3), logT_method="nearest")

    assert isinstance(r, xr.Dataset)
    assert "SG_resp" in r.data_vars
    assert r.sizes["logT"] == logT.size
    assert r.sizes["vdop"] == vdop.size
    assert r.sizes["slit"] == 3  # read_response selects np.arange(slit.max() + 1)
    np.testing.assert_allclose(r.logT.values, logT.values)
    np.testing.assert_allclose(r.vdop.values, vdop.values)
    # The reader injects Angstrom because the on-disk files carry no wavelength units.
    assert r.line_wvl.attrs["units"] == str(u.AA)
    assert r.SG_wvl.attrs["units"] == str(u.AA)
    np.testing.assert_array_equal(r.gain.values, [DEFAULT_GAIN])
    assert r.attrs["HISTORY"][-1].startswith("read_response(")


@pytest.mark.parametrize("fmt", ["nc", "zarr"])
def test_read_response_without_axes_returns_full_resolution(tmp_path, fmt) -> None:
    src = fake_response_file()
    path = _write(src, tmp_path / f"resp.{fmt}", fmt)

    r = read_response(path)

    assert r.sizes["logT"] == src.sizes["logT"]
    assert r.sizes["vdop"] == src.sizes["vdop"]
    assert r.line_wvl.attrs["units"] == str(u.AA)
    assert "gain" in r.coords


def test_read_response_linear_interp_hits_grid_and_stays_nonnegative(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.zarr", "zarr")
    # A grid offset from the source logT forces real interpolation rather than nearest selection.
    logT = _axis(np.linspace(5.1, 6.9, 9), "logT")

    r = read_response(path, logT=logT, logT_method="linear")

    np.testing.assert_allclose(r.logT.values, logT.values)
    assert bool((r.SG_resp >= 0).all())  # interp path clamps negatives to zero


def test_read_response_expands_line_dim_and_fills_line_wvl_from_attr(tmp_path) -> None:
    # Drop the line dimension and line_wvl to hit the expand_dims + attribute-fallback branches.
    src = fake_response_file().isel(line=0).drop_vars(["line", "line_wvl", "channel"])
    src.attrs["MAIN_LINE_WVL"] = 171.073
    path = _write(src, tmp_path / "resp.zarr", "zarr")

    r = read_response(path)

    assert "line" in r.dims
    assert float(r.line_wvl) == pytest.approx(171.073)
    assert r.line_wvl.attrs["units"] == str(u.AA)


def test_read_response_warns_on_missing_wavelength_units(tmp_path, caplog) -> None:
    # The fixture mirrors the real files, which carry no units on line_wvl/SG_wvl.
    path = _write(fake_response_file(), tmp_path / "resp.zarr", "zarr")

    r = read_response(path)

    assert "missing the 'units' attribute" in caplog.text
    assert r.line_wvl.attrs["units"] == str(u.AA)  # Angstrom assumed for now
    assert r.SG_wvl.attrs["units"] == str(u.AA)


def test_read_response_keeps_existing_wavelength_units(tmp_path, caplog) -> None:
    src = fake_response_file()
    src.line_wvl.attrs["units"] = "nm"
    src.SG_wvl.attrs["units"] = "nm"
    path = _write(src, tmp_path / "resp.zarr", "zarr")

    r = read_response(path)

    assert "missing the 'units' attribute" not in caplog.text
    assert r.line_wvl.attrs["units"] == "nm"  # present units left untouched
    assert r.SG_wvl.attrs["units"] == "nm"


def test_read_response_gain_accepts_quantity(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.zarr", "zarr")

    r = read_response(path, gain=5.0 * u.electron / u.DN)

    np.testing.assert_array_equal(r.gain.values, [5.0])
    assert r.gain.attrs["units"] == str(u.electron / u.DN)


def test_read_response_gain_rejects_wrong_units(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.zarr", "zarr")
    with pytest.raises(u.UnitsError):
        read_response(path, gain=5.0 * u.second)


def test_read_response_missing_file_raises(tmp_path) -> None:
    with pytest.raises((AssertionError, ValueError), match="does not exist"):
        read_response(str(tmp_path / "absent.nc"))


def test_read_response_invalid_method_raises(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.nc", "nc")
    with pytest.raises((AssertionError, ValueError), match="Invalid logT_method"):
        read_response(path, logT_method="sinc")


def test_read_response_empty_logT_raises(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.nc", "nc")
    with pytest.raises((AssertionError, ValueError), match="must not be empty"):
        read_response(path, logT=xr.DataArray(np.array([]), dims="logT"))


def test_read_response_nonfinite_logT_raises(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.nc", "nc")
    with pytest.raises((AssertionError, ValueError), match="finite"):
        read_response(path, logT=xr.DataArray(np.array([5.0, np.nan]), dims="logT"))


def test_read_response_out_of_range_logT_raises(tmp_path) -> None:
    path = _write(fake_response_file(), tmp_path / "resp.nc", "nc")
    with pytest.raises(ValueError, match="no overlap"):
        read_response(path, logT=xr.DataArray(np.array([8.0, 8.5]), dims="logT"))


def test_read_response_requires_sg_resp(tmp_path) -> None:
    path = _write(fake_response_file().drop_vars("SG_resp"), tmp_path / "resp.nc", "nc")
    with pytest.raises((AssertionError, ValueError), match="SG_resp"):
        read_response(path)


def test_load_and_concat_responses_concatenates_lines(tmp_path) -> None:
    first = fake_response_file()
    second = fake_response_file().assign_coords(
        line=("line", ["Fe XV 284.163"]),
        line_wvl=("line", [284.163]),
        channel=("line", [284]),
    )
    _write(first, tmp_path / "a.zarr", "zarr")
    _write(second, tmp_path / "b.zarr", "zarr")

    resp = load_and_concat_responses(
        response_directory=tmp_path,
        response_files=["a.zarr", "b.zarr"],
        logT=_axis(np.linspace(5.2, 6.6, 4), "logT"),
        vdop=_axis([-100.0, 0.0, 100.0], "vdop"),
        slit=_slit(3),
        logT_method="nearest",
        channels=[171, 284],
    )

    assert resp.sizes["line"] == 2
    np.testing.assert_array_equal(resp.channel.values, [171, 284])
    assert "effective_area" not in resp.data_vars  # dropped before concatenation
    assert "SG_resp" in resp.data_vars
