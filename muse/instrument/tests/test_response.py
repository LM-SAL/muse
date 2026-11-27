import dask.array as da
import numpy as np
import pytest
import xarray as xr
from attrs import evolve

import astropy.units as u

from muse.instrument import map_response_to_ci_detector, map_response_to_sg_detector
from muse.instrument.response.io import read_spectral_response, save_spectral_response
from muse.instrument.response.spectral import create_log_wavelength_edges, create_spectral_response
from muse.tests.helpers import fake_line_list
from muse.variables import DEFAULTS_MUSE


def _spectral_response(*, contaminants: bool = False) -> xr.Dataset:
    wavelength = np.arange(169.5, 171.51, 0.01)
    line = ["Fe IX 171.073"]
    line_wavelength = [171.073]
    if contaminants:
        line.append("contaminants")
        line_wavelength.append(np.nan)

    values = np.broadcast_to(
        wavelength,
        (len(line), 1, 1, wavelength.size),
    ).copy()
    return xr.Dataset(
        {
            "spectral_response": (
                ("line", "logT", "doppler_velocity", "wavelength_bin"),
                values,
                {"units": "1e-27 erg cm5 / (Angstrom s sr)"},
            )
        },
        coords={
            "line": line,
            "logT": [6.0],
            "doppler_velocity": ("doppler_velocity", [0.0], {"units": "km / s"}),
            "wavelength_grid": ("wavelength_bin", wavelength, {"units": "Angstrom"}),
            "line_wavelength": ("line", line_wavelength, {"units": "Angstrom"}),
        },
        attrs={"normalization": 1e-27},
    )


def _ci_spectral_response() -> xr.Dataset:
    response = _spectral_response().isel(wavelength_bin=slice(0, 3))
    values = np.broadcast_to([1.0, 2.0, 4.0], response.spectral_response.shape).copy()
    return response.assign(
        spectral_response=(response.spectral_response.dims, values, {"units": "1e-27 cm5 ph / (Angstrom s)"}),
    ).assign_coords(
        wavelength_grid=("wavelength_bin", [190.0, 191.0, 194.0], {"units": "Angstrom"}),
        line_wavelength=("line", [195.0], {"units": "Angstrom"}),
    )


def test_map_response_to_sg_detector_geometry_and_units():
    response = _spectral_response()
    original = response.copy(deep=True)

    mapped = map_response_to_sg_detector(
        response,
        number_of_slits=2,
        dispersion=0.1 * u.AA / u.pix,
        pixels_between_slits=2 * u.pix,
        detector_x_pixels=4,
        wavelength_start=170 * u.AA,
    )

    expected_wavelength = (170.0 + np.arange(4) * 0.1)[np.newaxis, :] - np.array([[0.0], [0.2]])
    wavelength = response.wavelength_grid.values
    expected_response = np.stack([np.interp(row, wavelength, wavelength) for row in expected_wavelength]) * 0.1

    assert mapped.detector_response.dims == ("line", "logT", "doppler_velocity", "slit", "detector_x_pixel")
    assert mapped.detector_wavelength.dims == ("detector_x_pixel", "slit")
    np.testing.assert_allclose(
        mapped.detector_wavelength.transpose("slit", "detector_x_pixel"),
        expected_wavelength,
    )
    np.testing.assert_allclose(mapped.detector_response.isel(line=0, logT=0, doppler_velocity=0), expected_response)
    assert u.Unit(mapped.detector_response.attrs["units"]) == u.Unit("1e-27 erg cm5 / (s sr)")
    assert mapped.detector_wavelength.attrs["units"] == "Angstrom"
    assert mapped.line_wavelength.attrs["units"] == "Angstrom"
    assert "channel" not in mapped.coords
    assert "wavelength_bin" not in mapped.dims
    assert mapped.attrs["history"][-1].startswith("map_response_to_sg_detector(")
    xr.testing.assert_identical(response, original)


def test_map_response_to_sg_detector_uses_muse_defaults():
    mapped = map_response_to_sg_detector(_spectral_response(), channel=17.1 * u.nm)

    assert mapped.channel.item() == 171
    assert mapped.channel.attrs["units"] == "Angstrom"


def test_map_response_to_sg_detector_preserves_invalid_physical_line_wavelength():
    # one slit, so no pixels_between_slits
    mapped = map_response_to_sg_detector(
        _spectral_response(contaminants=True),
        number_of_slits=1,
        dispersion=0.01 * u.AA / u.pix,
        detector_x_pixels=1,
        wavelength_start=171 * u.AA,
    )

    np.testing.assert_allclose(mapped.line_wavelength, [171.073, np.nan])


def test_map_response_to_sg_detector_requires_effective_area():
    response = _spectral_response()
    response.spectral_response.attrs["units"] = "1e-27 erg cm3 / (Angstrom s sr)"

    with pytest.raises(ValueError, match="convertible"):
        map_response_to_sg_detector(response, channel=171 * u.AA)


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("channel", ValueError, "unsupported MUSE SG channel"),
        ("channel_shape", ValueError, "scalar wavelength"),
        ("channel_nonfinite", ValueError, "finite, positive wavelength"),
        ("schema", ValueError, "missing required variables"),
        ("normalization", ValueError, "normalization"),
        ("wavelength_grid_empty", ValueError, "wavelength_grid"),
        ("wavelength_grid_positive", ValueError, "positive"),
        ("pixels_between_slits", ValueError, "pixels_between_slits"),
        ("geometry", ValueError, "number_of_slits"),
    ],
)
def test_map_response_to_sg_detector_rejects_invalid_inputs(case, error, match):
    response = _spectral_response()
    channel = 171 * u.AA
    kwargs = {}
    if case == "channel":
        channel = 195 * u.AA
    elif case == "channel_shape":
        channel = [171] * u.AA
    elif case == "channel_nonfinite":
        channel = np.nan * u.AA
    elif case == "schema":
        response = response.drop_vars("line_wavelength")
    elif case == "normalization":
        response.attrs["normalization"] = 0
    elif case == "wavelength_grid_empty":
        response = response.isel(wavelength_bin=slice(0, 0))
    elif case == "wavelength_grid_positive":
        response = response.assign_coords(wavelength_grid=response.wavelength_grid - 200)
    elif case == "pixels_between_slits":
        kwargs["pixels_between_slits"] = 0 * u.pix
    else:
        kwargs["number_of_slits"] = 0

    with pytest.raises(error, match=match):
        map_response_to_sg_detector(response, channel=channel, **kwargs)


def test_map_response_to_sg_detector_requires_explicit_geometry_without_channel():
    with pytest.raises(
        ValueError,
        match="missing: number_of_slits, dispersion, detector_x_pixels, wavelength_start",
    ):
        map_response_to_sg_detector(_spectral_response())


def test_map_response_to_sg_detector_refuses_slits_without_their_spacing():
    no_spacing = evolve(DEFAULTS_MUSE, pixels_between_slits_SG=None)

    with pytest.raises(ValueError, match="missing: pixels_between_slits"):
        map_response_to_sg_detector(
            _spectral_response(), channel=171 * u.AA, dispersion=0.1 * u.AA / u.pix, instrument_defaults=no_spacing
        )


def test_a_mapped_response_states_how_far_apart_its_slits_fall():
    # given a dispersion, the separation is that of its pixels between slits, without the noise of 0.01 * 41
    explicit = map_response_to_sg_detector(
        _spectral_response(),
        number_of_slits=2,
        dispersion=0.01 * u.AA / u.pix,
        pixels_between_slits=41 * u.pix,
        detector_x_pixels=4,
        wavelength_start=170 * u.AA,
    )
    assert explicit.attrs["spectral_slit_separation"] == "410 mAngstrom"


def test_map_response_to_ci_detector_integrates_nonuniform_grid_without_detector_axis():
    response = _ci_spectral_response().chunk({"wavelength_bin": 2})
    original = response.copy(deep=True)

    with xr.set_options(keep_attrs=False):
        mapped = map_response_to_ci_detector(response, channel=171 * u.AA)

    assert isinstance(mapped.detector_response.data, da.Array)
    assert mapped.detector_response.dims == ("line", "logT", "doppler_velocity")
    assert "wavelength_bin" not in mapped.dims
    assert mapped.detector_wavelength.dims == ("line",)
    assert mapped.detector_wavelength.item() == 171
    assert mapped.detector_wavelength.attrs["units"] == "Angstrom"
    assert mapped.line_wavelength.item() == 195
    assert mapped.channel.item() == 171
    assert mapped.channel.attrs["units"] == "Angstrom"
    assert mapped.attrs["normalization"] == 1e-27
    assert mapped.attrs["history"][-1].startswith("map_response_to_ci_detector(")
    assert u.Unit(mapped.detector_response.attrs["units"]) == u.Unit("1e-27 cm5 ph / s")
    np.testing.assert_allclose(mapped.detector_response.compute(), 10.5)
    xr.testing.assert_identical(response, original)


def test_map_response_to_ci_detector_rejects_invalid_units():
    response = _ci_spectral_response()
    response = response.assign(
        spectral_response=response.spectral_response.assign_attrs(units="erg cm3 / (s sr)"),
    )

    with pytest.raises(ValueError, match="convertible"):
        map_response_to_ci_detector(response, channel=171 * u.AA)


def test_map_response_to_ci_detector_needs_two_wavelengths_to_integrate():
    with pytest.raises(ValueError, match="at least two points"):
        map_response_to_ci_detector(_ci_spectral_response().isel(wavelength_bin=slice(0, 1)), channel=195 * u.AA)
    with pytest.raises(ValueError, match="finite, positive"):
        map_response_to_ci_detector(_ci_spectral_response(), channel=-171 * u.AA)


def test_ci_mapping_sums_the_bins_when_the_response_carries_bounds():
    line_list = fake_line_list(1)
    response = create_spectral_response(
        line_list,
        main_lines=[line_list.full_name[0].item()],
        doppler_velocity=np.array([-200.0, 0.0, 200.0]) * u.km / u.s,
        wavelength_edges=create_log_wavelength_edges([100, 300] * u.AA, 0.01),
        bin_method="average",
    )
    mapped = map_response_to_ci_detector(response, channel=171 * u.AA)
    expected = line_list.gofnt.isel(transition=0) / response.attrs["normalization"]
    np.testing.assert_allclose((mapped.detector_response.isel(line=0) / expected).values, 1.0, rtol=1e-12)
    assert "bound" not in mapped.dims
    assert u.Unit(mapped.detector_response.attrs["units"]) == u.Unit(response.spectral_response.attrs["units"]) * u.AA


def test_a_saved_spectral_response_maps_to_an_imager_as_the_one_in_memory(tmp_path):
    """
    A netCDF file keeps the units of the bin bounds on the grid alone: the bounds read
    back without them must still give the bin widths.
    """
    response = create_spectral_response(
        fake_line_list(),
        wavelength_edges=np.linspace(170.0, 172.0, 9) * u.AA,
        main_lines=None,
        include_contaminants=True,
        effective_area=xr.DataArray(2.0, attrs={"units": "cm2"}),
    )
    save_spectral_response(response, tmp_path / "cube.nc")

    back = read_spectral_response(tmp_path / "cube.nc")

    # the cube and its bin bounds read back as they were saved
    xr.testing.assert_allclose(back.spectral_response, response.spectral_response)
    np.testing.assert_array_equal(back.wavelength_bounds.values, response.wavelength_bounds.values)
    xr.testing.assert_allclose(
        map_response_to_ci_detector(back, channel=171 * u.AA).detector_response,
        map_response_to_ci_detector(response, channel=171 * u.AA).detector_response,
    )


def test_sg_mapping_one_slit_at_a_time_is_the_interpolation_of_all_slits_at_once():
    """
    A response in memory is interpolated one slit at a time and a chunked one in one
    lazy call: both give the same detector response.
    """
    response = _spectral_response(contaminants=True)
    kwargs = {
        "number_of_slits": 3,
        "dispersion": 0.02 * u.AA / u.pix,
        "pixels_between_slits": 5 * u.pix,
        "detector_x_pixels": 60,
        "wavelength_start": 170.6 * u.AA,
    }

    in_memory = map_response_to_sg_detector(response, **kwargs)
    chunked = map_response_to_sg_detector(response.chunk({"logT": 1}), **kwargs)

    assert isinstance(in_memory.detector_response.data, np.ndarray)
    assert isinstance(chunked.detector_response.data, da.Array)
    xr.testing.assert_identical(chunked.compute().drop_attrs(), in_memory.drop_attrs())
    assert chunked.detector_response.attrs == in_memory.detector_response.attrs
    # the attributes of the spectral response stay on the mapped one, with the new units
    assert in_memory.detector_response.attrs == {
        **response.spectral_response.attrs,
        "units": in_memory.detector_response.attrs["units"],
    }
    assert float(in_memory.detector_response.max()) > 0
    # the last pixels fall beyond the wavelength grid
    assert (in_memory.detector_response.isel(detector_x_pixel=-1) == 0).all()
