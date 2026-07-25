import dask.array as da
import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.tests.helpers import assert_dataset_structure
from muse.transforms.transforms import (
    _spacing_matches,
    match_fov,
    reshape_slit_step_to_x,
    reshape_x_to_slit_step,
)


@pytest.mark.parametrize(
    ("spacing", "to_arcsec", "expected"),
    [
        (None, 1.0, False),  # a size-1 axis has no spacing to measure
        (0.4, 1.0, True),  # exactly the MUSE pixel size
        (0.4 * 1.004, 1.0, True),  # 0.4% off, inside the 0.5% tolerance
        (0.4 * 1.006, 1.0, False),  # 0.6% off, outside it
        (2.0, 0.2, True),  # coordinate in another unit, scaled to 0.4 arcsec
    ],
)
def test_spacing_matches(spacing, to_arcsec, expected) -> None:
    values = np.zeros(1) if spacing is None else np.arange(5.0) * spacing

    assert _spacing_matches(xr.DataArray(values, dims="x"), to_arcsec, 0.4) is expected


def test_spacing_matches_unwraps_a_quantity_coordinate() -> None:
    # Coordinate data may carry a unit; the scaled spacing must be unwrapped before it is
    # compared with a plain float, or the comparison raises.
    coord = xr.DataArray(np.arange(5.0) * 0.4 * u.arcsec, dims="x")

    assert _spacing_matches(coord, 1.0, 0.4) is True


def test_match_fov_returns_input_when_already_muse_resolution(vdem) -> None:
    # The default fixture already sits at the MUSE pixel size, so match_fov takes
    # the early-return path and hands back the same object untouched.
    assert match_fov(vdem) is vdem


def test_match_fov_returns_input_for_single_y_row(vdem) -> None:
    # x matches the MUSE pixel size and there is only one y row, so dy cannot be
    # measured: match_fov accepts on the x match alone and returns the input.
    single_row = vdem.isel(y=[0])
    assert match_fov(single_row) is single_row


def test_match_fov_returns_input_for_single_x_column(vdem) -> None:
    # Only one x column, so dx cannot be measured: match_fov checks dy alone and,
    # since it matches the MUSE pixel size, returns the input.
    single_column = vdem.isel(x=[0])
    assert match_fov(single_column) is single_column


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("y_mismatch", "dy is off the MUSE pixel size, so the x match alone must not accept"),
        ("wrong_x_count", "dx matches but x does not span nslits * nraster, so it must not accept"),
    ],
)
def test_match_fov_does_not_return_input_when_only_one_axis_qualifies(vdem, case, reason) -> None:
    # The negative side of the early-return condition: each of these differs from the
    # accepted case in exactly one term, so a wrong boolean would return the input unchanged.
    source = vdem.assign_coords(y=vdem.y * 3.0) if case == "y_mismatch" else vdem.isel(x=slice(0, 100))

    assert match_fov(source) is not source, reason


def test_match_fov_relabels_single_pixel_input(vdem) -> None:
    # Both axes are size 1: nothing to resample or tile, so match_fov falls through
    # to the tail and returns a copy with the coords relabeled onto the MUSE grid.
    # NOTE: this degenerate relabel-only path may be removed in the future.
    single_pixel = vdem.isel(x=[0], y=[0])
    out = match_fov(single_pixel)
    assert out is not single_pixel
    assert out.sizes["x"] == 1
    assert out.sizes["y"] == 1
    np.testing.assert_array_equal(out.x.values, [0.0])
    np.testing.assert_array_equal(out.y.values, [0.0])
    assert out.x.attrs["units"] == "arcsec"
    assert out.y.attrs["units"] == "arcsec"
    assert "HISTORY" in out.attrs


def test_match_fov_resamples_offgrid_to_muse_grid(vdem_offgrid) -> None:
    out = match_fov(vdem_offgrid)
    assert out is not vdem_offgrid
    assert_dataset_structure(
        out,
        data_vars=("vdem",),
        coords=("logT", "doppler_velocity", "x", "y"),
        sizes={"logT": 7, "doppler_velocity": 9, "x": 385, "y": 62},
        finite_vars=("vdem",),
    )
    # Output spatial axes are remapped onto the MUSE pixel grid.
    assert out.x.attrs["units"] == "arcsec"
    assert out.y.attrs["units"] == "arcsec"
    assert float(out.x[0]) == 0.0
    assert float(out.x[1] - out.x[0]) == pytest.approx(0.4)
    assert float(out.y[1] - out.y[0]) == pytest.approx(0.167)


def test_match_fov_preserves_dimension_order(vdem_offgrid) -> None:
    # The resampling unstack moves the resampled axis last; match_fov must restore
    # the caller's dimension order so 2D plots keep their orientation.
    out = match_fov(vdem_offgrid)
    assert out.vdem.dims == vdem_offgrid.vdem.dims


def test_transforms_keep_dask_inputs_lazy(vdem_offgrid, vdem) -> None:
    # Large VDEMs arrive dask-backed (open_zarr); the FOV/reshape transforms must
    # extend the graph, not silently compute, and match the eager results.
    lazy_fov = match_fov(vdem_offgrid.chunk({"logT": 4, "x": 32}))
    assert isinstance(lazy_fov.vdem.data, da.Array)
    np.testing.assert_allclose(lazy_fov.vdem.compute().values, match_fov(vdem_offgrid).vdem.values, rtol=1e-12)

    lazy_raster = reshape_x_to_slit_step(vdem.chunk({"logT": 4, "x": 64}), nslits=35, nraster=11)
    assert isinstance(lazy_raster.vdem.data, da.Array)
    eager_raster = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    np.testing.assert_array_equal(lazy_raster.vdem.compute().values, eager_raster.vdem.values)

    lazy_x = reshape_slit_step_to_x(lazy_raster)
    assert isinstance(lazy_x.vdem.data, da.Array)
    np.testing.assert_array_equal(lazy_x.vdem.compute().values, reshape_slit_step_to_x(eager_raster).vdem.values)


def test_match_fov_tiles_to_fill_fov(vdem_offgrid) -> None:
    # Coarse pixels under-fill the FOV, so the x axis is padded out to nslits*nraster.
    out = match_fov(vdem_offgrid, dx_pix=1.0 * u.arcsec)
    assert out.x.size == 35 * 11
    assert bool(np.isfinite(out.vdem).all())


def test_match_fov_notile_keeps_resampled_width(vdem_offgrid) -> None:
    out = match_fov(vdem_offgrid, dx_pix=1.0 * u.arcsec, restype="match_res_notile")
    # "notile" suffix skips the pad, so the resampled width is kept as-is (< full FOV).
    assert out.x.size == 308
    assert out.x.size < 35 * 11
    assert bool(np.isfinite(out.vdem).all())


def test_match_fov_downsamples_with_factor_branch(vdem_offgrid) -> None:
    # Very coarse target pixels exercise the integer-factor averaging branch.
    out = match_fov(vdem_offgrid, dx_pix=4.0 * u.arcsec)
    assert out.x.size == 35 * 11
    assert bool(np.isfinite(out.vdem).all())


def test_match_fov_downsamples_y_with_factor_branch(vdem_offgrid) -> None:
    # Coarse y pixels exercise the integer-factor averaging branch on the y axis.
    out = match_fov(vdem_offgrid, dy_pix=2.0 * u.arcsec)
    assert out.y.size == 5
    assert bool(np.isfinite(out.vdem).all())


def test_match_fov_rotate(vdem_offgrid) -> None:
    out = match_fov(vdem_offgrid, rotate=True)
    assert_dataset_structure(
        out,
        data_vars=("vdem",),
        coords=("logT", "doppler_velocity", "x", "y"),
        finite_vars=("vdem",),
    )
    assert out.x.size == 35 * 11


def test_reshape_x_to_slit_step(vdem) -> None:
    out = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    assert_dataset_structure(
        out,
        data_vars=("vdem",),
        coords=("slit", "step", "logT", "doppler_velocity", "y"),
        sizes={"slit": 35, "step": 11, "logT": 7, "doppler_velocity": 9, "y": 32},
        finite_vars=("vdem",),
    )
    value = out.vdem.sel(step=0, logT=5, doppler_velocity=0, y=0, slit=0, method="nearest").values
    assert np.isfinite(value)
    assert value >= 0
    assert out.attrs["step_size units"] == "arcsec"
    np.testing.assert_array_equal(out.vdem.sel(slit=0, step=0), vdem.vdem.isel(x=0))
    np.testing.assert_array_equal(out.vdem.sel(slit=34, step=10), vdem.vdem.isel(x=384))


def test_reshape_x_to_slit_step_does_not_mutate_input_history(vdem) -> None:
    source = vdem.copy(deep=True)
    source.attrs["HISTORY"] = ["match_fov(ds=ds)"]

    out = reshape_x_to_slit_step(source, nslits=35, nraster=11)

    assert source.attrs["HISTORY"] == ["match_fov(ds=ds)"]
    assert out.attrs["HISTORY"] == [
        "match_fov(ds=ds)",
        "reshape_x_to_slit_step(ds=ds, nslits=35, nraster=11)",
    ]


def test_reshape_x_to_slit_step_unstacks_existing_slit(vdem) -> None:
    # When x already carries a (slit, step) MultiIndex, the function unstacks it
    # instead of building a fresh index.
    step, slit = (arr.flatten() for arr in np.meshgrid(range(11), range(35)))
    stacked = vdem.assign_coords(slit=("x", slit), step=("x", step)).set_index(x=("slit", "step"))
    stacked.x.attrs["units"] = "arcsec"

    out = reshape_x_to_slit_step(stacked)
    assert_dataset_structure(
        out,
        data_vars=("vdem",),
        coords=("slit", "step", "logT", "doppler_velocity", "y"),
        sizes={"slit": 35, "step": 11, "logT": 7, "doppler_velocity": 9, "y": 32},
        finite_vars=("vdem",),
    )
    assert out.attrs["HISTORY"] == ["reshape_x_to_slit_step(ds=ds, nslits=35, nraster=11)"]


def test_reshape_slit_step_to_x_round_trips(vdem) -> None:
    reshaped = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    out = reshape_slit_step_to_x(reshaped, nslits=35, nraster=11)
    assert_dataset_structure(
        out,
        data_vars=("vdem",),
        coords=("x", "logT", "doppler_velocity", "y"),
        sizes={"x": 385, "logT": 7, "doppler_velocity": 9, "y": 32},
        finite_vars=("vdem",),
    )
    assert out.x.attrs["units"] == "arcsec"
    np.testing.assert_allclose(out.x.values, vdem.x.values)
    np.testing.assert_array_equal(out.vdem.transpose(*vdem.vdem.dims).values, vdem.vdem.values)


def test_reshape_slit_step_to_x_defaults_step_size_when_missing(vdem) -> None:
    reshaped = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    del reshaped.attrs["step_size"]
    out = reshape_slit_step_to_x(reshaped, nslits=35, nraster=11)
    # Falls back to the MUSE pixel size (0.4 arcsec) for the x spacing.
    np.testing.assert_allclose(out.x[1] - out.x[0], 0.4)


def test_reshape_slit_step_to_x_records_history(vdem) -> None:
    reshaped = reshape_x_to_slit_step(vdem, nslits=35, nraster=11)
    out = reshape_slit_step_to_x(reshaped, nslits=35, nraster=11)
    assert out.attrs["HISTORY"] == [
        "reshape_x_to_slit_step(ds=ds, nslits=35, nraster=11)",
        "reshape_slit_step_to_x(ds=ds, nslits=35, nraster=11)",
    ]


def test_reshape_slit_step_to_x_requires_slit_and_step(vdem) -> None:
    with pytest.raises(ValueError, match="slit coordinate is missing"):
        reshape_slit_step_to_x(vdem)


def test_match_fov_rejects_unknown_restype(vdem) -> None:
    with pytest.raises(ValueError, match="Unsupported restype"):
        match_fov(vdem, restype="match_fov")


def test_match_fov_requires_quantity_pixel_sizes(vdem) -> None:
    with pytest.raises(TypeError, match=r"dx_pix must be an astropy\.units\.Quantity"):
        match_fov(vdem, dx_pix=0.4)


def test_match_fov_requires_coordinate_units(vdem) -> None:
    bad_vdem = vdem.copy(deep=True)
    del bad_vdem.x.attrs["units"]
    with pytest.raises(ValueError, match="x coordinate must define units"):
        match_fov(bad_vdem)


def test_match_fov_requires_x_coordinate(vdem) -> None:
    with pytest.raises(ValueError, match="x coordinate is missing"):
        match_fov(vdem.drop_vars("x"))


def test_reshape_x_to_slit_step_requires_x_units(vdem) -> None:
    bad_vdem = vdem.copy(deep=True)
    del bad_vdem.x.attrs["units"]
    with pytest.raises(ValueError, match="x coordinate must define units"):
        reshape_x_to_slit_step(bad_vdem, nslits=35, nraster=11)


def test_reshape_x_to_slit_step_requires_x_coordinate(vdem) -> None:
    with pytest.raises(ValueError, match="x coordinate is missing"):
        reshape_x_to_slit_step(vdem.drop_vars("x"), nslits=35, nraster=11)
