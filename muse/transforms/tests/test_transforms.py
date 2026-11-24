import dask.array as da
import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.transforms.transforms import (
    _spacing_matches,
    match_fov,
    reshape_slit_step_to_x,
    reshape_x_to_slit_step,
)
from muse.variables import DEFAULTS_MUSE

WIDTH = DEFAULTS_MUSE.number_of_slits_SG * DEFAULTS_MUSE.steps_per_raster_SG


@pytest.mark.parametrize(
    ("spacing", "to_arcsec", "expected"),
    [
        (0.4 * 1.006, 1.0, False),  # 0.6% off, outside the 0.5% tolerance
        (2.0 * 1.004, 0.2, True),  # another unit, 0.4% off once scaled: inside the tolerance
    ],
)
def test_spacing_matches(spacing, to_arcsec, expected) -> None:
    assert _spacing_matches(xr.DataArray(np.arange(5.0) * spacing, dims="x"), to_arcsec, 0.4) is expected


def test_match_fov_returns_input_for_single_y_row(vdem) -> None:
    # x matches the MUSE pixel size and there is only one y row, so dy cannot be
    # measured: match_fov accepts on the x match alone and returns the input.
    single_row = vdem.isel(y=[0])
    assert match_fov(single_row) is single_row


def test_match_fov_without_tiling_returns_input_for_single_x_column(vdem) -> None:
    # Only one x column, so dx cannot be measured: match_fov checks dy alone and,
    # since it matches the MUSE pixel size and needs no tiling, returns the input.
    single_column = vdem.isel(x=[0])
    assert match_fov(single_column, extent="crop") is single_column


def test_match_fov_does_not_return_input_when_only_one_axis_qualifies(vdem) -> None:
    # dy is off the MUSE pixel size, so the x match alone must not accept.
    source = vdem.assign_coords(y=vdem.y * 3.0)

    assert match_fov(source) is not source


def test_match_fov_does_not_return_a_single_pixel_input_as_is(vdem) -> None:
    # Both axes are size 1, so no spacing can be measured: match_fov falls through to
    # the tail and returns a new dataset with the call in its history.
    single_pixel = vdem.isel(x=[0], y=[0])
    out = match_fov(single_pixel, extent="crop")
    assert out is not single_pixel
    assert out.sizes["x"] == 1
    assert out.sizes["y"] == 1
    assert "history" in out.attrs


def test_match_fov_resamples_offgrid_to_muse_grid(vdem_offgrid) -> None:
    out = match_fov(vdem_offgrid)
    assert dict(out.sizes) == {"logT": 7, "doppler_velocity": 9, "x": WIDTH, "y": 62}
    # Output spatial axes are remapped onto the MUSE pixel grid.
    assert out.x.attrs["units"] == "arcsec"
    assert out.y.attrs["units"] == "arcsec"
    assert float(out.x[0]) == float(out.y[0]) == 0.0
    assert float(out.x[1] - out.x[0]) == pytest.approx(DEFAULTS_MUSE.pixel_width_SG.value)
    assert float(out.y[1] - out.y[0]) == pytest.approx(DEFAULTS_MUSE.pixel_height_SG.value)
    # The resampling moves the resampled axis last; the caller's order keeps 2D plots in their orientation.
    assert out.vdem.dims == vdem_offgrid.vdem.dims
    assert out.vdem.attrs == vdem_offgrid.vdem.attrs
    # The first and last rows land on the input's first and last samples, so the field is not flipped.
    np.testing.assert_allclose(out.vdem.isel(x=0, y=[0, -1]).values, vdem_offgrid.vdem.isel(x=0, y=[0, -1]).values)
    # "keep" matches the resolution alone: the field, sampled at twice the pixel width, keeps all its columns.
    assert match_fov(vdem_offgrid, extent="keep").x.size == 2 * vdem_offgrid.x.size


def test_match_fov_keeps_dask_inputs_lazy(vdem_offgrid) -> None:
    # Large VDEMs arrive dask-backed (open_zarr); match_fov must extend the graph,
    # not silently compute, and match the eager result.
    lazy_fov = match_fov(vdem_offgrid.chunk({"logT": 4, "x": 32}))
    assert isinstance(lazy_fov.vdem.data, da.Array)
    np.testing.assert_allclose(lazy_fov.vdem.compute().values, match_fov(vdem_offgrid).vdem.values, rtol=1e-12)


@pytest.mark.parametrize("fraction", [1 / 4, 1 / 2])  # the whole-factor branch, then the oversampled one
def test_match_fov_averages_rows_into_pixels(fraction) -> None:
    # 40 rows fill a whole number of pixels at both fractions.
    y = np.arange(40) * fraction * DEFAULTS_MUSE.pixel_height_SG.value
    alternating = xr.Dataset(
        {"vdem": (("y", "x"), (np.arange(40) % 2.0)[:, np.newaxis])},
        coords={"x": ("x", [0.0], {"units": "arcsec"}), "y": ("y", y, {"units": "arcsec"})},
    )

    np.testing.assert_allclose(match_fov(alternating, extent="keep").vdem.values, 0.5)


def test_match_fov_constant_mode_pads_with_zeros(vdem_offgrid) -> None:
    # The padded columns mean "no emission here", so they must be zero: xarray.pad defaults
    # constant_values to NaN, which would propagate through synthesis into every moment.
    resampled_width = match_fov(vdem_offgrid, pixel_width=1.0 * u.arcsec, extent="crop").x.size

    out = match_fov(vdem_offgrid, pixel_width=1.0 * u.arcsec, pad_mode="constant")

    assert out.x.size == WIDTH
    assert bool(np.isfinite(out.vdem).all())
    np.testing.assert_array_equal(out.vdem.isel(x=slice(resampled_width, None)).values, 0.0)
    # only the padding is zero: the resampled field in front of it keeps its emission
    assert float(out.vdem.isel(x=slice(0, resampled_width)).sum()) > 0


def test_match_fov_without_extent_match_returns_input_at_pixel_size(vdem) -> None:
    # With the width left alone, an input already at MUSE resolution needs no work
    # at all, even one narrow enough that the tiling path would extend it.
    narrower = vdem.isel(x=slice(0, 100))

    assert match_fov(narrower, extent="keep") is narrower


def test_match_fov_rejects_invalid_extent(vdem) -> None:
    with pytest.raises(ValueError, match="Unsupported extent 'foo'"):
        match_fov(vdem, extent="foo")


def test_match_fov_without_tiling_still_crops_to_fov(vdem) -> None:
    wider = vdem.isel(x=np.arange(WIDTH + 15) % WIDTH).assign_coords(
        x=np.arange(WIDTH + 15) * DEFAULTS_MUSE.pixel_width_SG.value
    )
    wider.x.attrs["units"] = "arcsec"

    out = match_fov(wider, extent="crop")

    assert out.x.size == WIDTH
    np.testing.assert_array_equal(out.vdem.values, wider.vdem.isel(x=slice(0, WIDTH)).values)


def test_match_fov_crops_y_to_y_pixels(vdem) -> None:
    out = match_fov(vdem, y_pixels=20)

    assert out.y.size == 20
    np.testing.assert_array_equal(out.vdem.values, vdem.vdem.isel(y=slice(0, 20)).values)


def test_match_fov_extends_y_to_y_pixels_when_tiling(vdem) -> None:
    # Rows are extended as x is: "wrap" repeats the field.
    out = match_fov(vdem, y_pixels=40)

    assert out.y.size == 40
    np.testing.assert_array_equal(out.vdem.isel(y=slice(0, 32)).values, vdem.vdem.values)
    np.testing.assert_array_equal(out.vdem.isel(y=slice(32, None)).values, vdem.vdem.isel(y=slice(0, 8)).values)
    np.testing.assert_allclose(np.diff(out.y.values), np.diff(vdem.y.values)[0], rtol=5e-3)


def test_match_fov_keeps_a_single_row_whatever_y_pixels(vdem) -> None:
    # Narrower than the field, so the input is extended along x and reaches the fit of the rows.
    out = match_fov(vdem.isel(y=[0], x=slice(0, 100)), y_pixels=40)

    assert out.y.size == 1


def test_match_fov_counts_y_pixels_after_the_resampling(vdem_offgrid) -> None:
    # The coarse input is resampled to more rows than it has, and y_pixels counts those.
    resampled_rows = match_fov(vdem_offgrid).y.size
    assert resampled_rows > vdem_offgrid.y.size
    assert match_fov(vdem_offgrid, y_pixels=resampled_rows - 5).y.size == resampled_rows - 5
    # Without "extend", fewer rows are kept.
    assert match_fov(vdem_offgrid, extent="crop", y_pixels=resampled_rows + 5).y.size == resampled_rows


def test_match_fov_rejects_y_pixels_below_one(vdem) -> None:
    with pytest.raises(ValueError, match="y_pixels must be at least 1"):
        match_fov(vdem, y_pixels=0)


def test_match_fov_rotation_skips_resampling_when_resolution_matches(vdem) -> None:
    source = vdem.assign_coords(
        x=np.arange(vdem.x.size) * DEFAULTS_MUSE.pixel_height_SG.value,
        y=np.arange(vdem.y.size) * DEFAULTS_MUSE.pixel_width_SG.value,
    )
    source.x.attrs["units"] = "arcsec"
    source.y.attrs["units"] = "arcsec"

    out = match_fov(source, rotation=90 * u.deg, extent="crop")

    assert out.sizes["x"] == source.sizes["y"]
    assert out.sizes["y"] == source.sizes["x"]
    assert out.vdem.dims == source.vdem.dims  # in the caller's order, so the array is not mirrored
    # A quarter turn counter-clockwise moves the point at (x, y) to (-y, x).
    last_y = source.sizes["y"] - 1
    for x, y in [(0, 0), (5, 3), (384, 31)]:
        np.testing.assert_array_equal(out.vdem.isel(x=last_y - y, y=x).values, source.vdem.isel(x=x, y=y).values)


def test_match_fov_turns_by_any_multiple_of_a_quarter(vdem) -> None:
    # The fixture sits at the MUSE pixel size, so only the turn changes the field.
    half = match_fov(vdem, rotation=180 * u.deg)
    last_x, last_y = vdem.sizes["x"] - 1, vdem.sizes["y"] - 1
    for x, y in [(0, 0), (5, 3)]:
        np.testing.assert_array_equal(
            half.vdem.isel(x=last_x - x, y=last_y - y).values, vdem.vdem.isel(x=x, y=y).values
        )
    xr.testing.assert_identical(match_fov(vdem, rotation=-90 * u.deg).vdem, match_fov(vdem, rotation=270 * u.deg).vdem)
    # a variable along x alone turns with the field, onto y
    turned = match_fov(vdem.assign(row=vdem.vdem.isel(logT=0, doppler_velocity=0, y=0, drop=True)), rotation=90 * u.deg)
    assert turned.row.sizes == {"y": turned.sizes["y"]}
    # The tolerance is absolute: a relative one would take this as a quarter turn.
    with pytest.raises(ValueError, match="multiple of 90 degrees"):
        match_fov(vdem, rotation=90.0001 * u.deg)


def test_match_fov_turns_an_uneven_grid_to_the_mirrored_positions(vdem) -> None:
    # The value of each sample is its y; after a quarter turn it sits at x = y[0] + y[-1] - y.
    y = np.array([0.0, 0.1, 0.3, 0.7, 1.5])
    uneven = vdem.isel(y=slice(0, 5), x=[0]).assign_coords(y=("y", y, {"units": "arcsec"}))
    uneven = uneven.assign(vdem=uneven.vdem * 0 + uneven.y)

    turned = match_fov(uneven, rotation=90 * u.deg, extent="keep", pixel_width=0.25 * u.arcsec)

    np.testing.assert_allclose(turned.vdem.isel(logT=0, doppler_velocity=0, y=0).values, [1.5, 1.2, 0.9, 0.6, 0.3, 0.0])


def test_reshape_slit_step_to_x_round_trips(vdem) -> None:
    vdem = vdem.assign_coords(x=("x", np.arange(vdem.sizes["x"]) * 2.5e7, {"units": "cm"}))
    out = reshape_slit_step_to_x(reshape_x_to_slit_step(vdem))

    assert out.x.attrs["units"] == "cm"
    np.testing.assert_allclose(out.x.values, vdem.x.values)
    np.testing.assert_array_equal(out.vdem.transpose(*vdem.vdem.dims).values, vdem.vdem.values)
    assert [entry.split("(")[0] for entry in out.attrs["history"]] == [
        "reshape_x_to_slit_step",
        "reshape_slit_step_to_x",
    ]


def test_reshape_slit_step_to_x_refuses_a_step_size_without_its_unit(raster) -> None:
    raster.attrs["step_size"] = 0.4  # as muse 0.3 wrote it, beside a separate unit

    with pytest.raises(ValueError, match="has no unit"):
        reshape_slit_step_to_x(raster)


def test_reshape_slit_step_to_x_defaults_step_size_when_missing(raster) -> None:
    del raster.attrs["step_size"]
    out = reshape_slit_step_to_x(raster)
    # Falls back to the MUSE pixel size for the x spacing.
    np.testing.assert_allclose(out.x[1] - out.x[0], DEFAULTS_MUSE.pixel_width_SG.value)


def test_reshape_slit_step_to_x_requires_slit_and_step(vdem) -> None:
    with pytest.raises(ValueError, match="slit coordinate is missing"):
        reshape_slit_step_to_x(vdem)


def test_match_fov_requires_coordinate_units(vdem) -> None:
    bad_vdem = vdem.copy(deep=True)
    del bad_vdem.x.attrs["units"]
    with pytest.raises(ValueError, match="x coordinate must define units"):
        match_fov(bad_vdem)


def test_reshape_x_to_slit_step_requires_x_units(vdem) -> None:
    bad_vdem = vdem.copy(deep=True)
    del bad_vdem.x.attrs["units"]
    with pytest.raises(ValueError, match="x coordinate must define units"):
        reshape_x_to_slit_step(bad_vdem)
