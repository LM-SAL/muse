from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import astropy.constants as const
import astropy.units as u

from muse.instrument import linelist as linelist_module
from muse.instrument.linelist import create_chianti_line_list
from muse.instrument.response import map_response_to_sg_detector
from muse.instrument.spectral import create_spectral_response
from muse.instrument.utils import read_response
from muse.synthesis.synthesis import vdem_synthesis
from muse.variables import DEFAULTS_MUSE


def _spectral_response(*, contaminants: bool = False) -> xr.Dataset:
    wavelength = np.arange(169.5, 171.51, 0.01)
    line = ["Fe IX 171.073"]
    line_wavelength = [171.073]
    component_kind = ["line"]
    if contaminants:
        line.append("contaminants")
        line_wavelength.append(np.nan)
        component_kind.append("contaminants")

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
            "component_kind": ("line", component_kind),
        },
        attrs={"normalization": 1e-27},
    )


def test_map_response_to_sg_detector_geometry_and_units():
    response = _spectral_response()
    original = response.copy(deep=True)

    mapped = map_response_to_sg_detector(
        response,
        171,
        number_of_slits=2,
        dispersion=0.1 * u.AA / u.pix,
        slit_spacing=2 * u.pix,
        detector_pixels=4,
        wavelength_start=170 * u.AA,
        pixel_width=1 * u.arcsec,
        pixel_height=2 * u.arcsec,
    )

    expected_wavelength = (170.0 + np.arange(4) * 0.1)[np.newaxis, :] - np.array([[0.0], [0.2]])
    solid_angle = (1 * u.arcsec).to_value(u.rad) * (2 * u.arcsec).to_value(u.rad)
    wavelength = response.wavelength_grid.values
    photon_energy = (const.h * const.c / (wavelength * u.AA)).to_value(u.erg)
    photon_response = wavelength * solid_angle / photon_energy
    expected_response = np.stack([np.interp(row, wavelength, photon_response) for row in expected_wavelength]) * 0.1

    assert mapped.detector_response.dims == ("line", "logT", "vdop", "slit", "detector_x_pixel")
    assert mapped.detector_wavelength.dims == ("detector_x_pixel", "slit")
    np.testing.assert_allclose(
        mapped.detector_wavelength.transpose("slit", "detector_x_pixel"),
        expected_wavelength,
    )
    np.testing.assert_allclose(np.diff(mapped.detector_wavelength.isel(slit=0)), 0.1)
    np.testing.assert_allclose(mapped.detector_response.isel(line=0, logT=0, vdop=0), expected_response)
    assert u.Unit(mapped.detector_response.attrs["units"]) == u.Unit("1e-27 ph cm5 / s")
    assert mapped.detector_wavelength.attrs["units"] == "Angstrom"
    assert mapped.line_wavelength.attrs["units"] == "Angstrom"
    assert mapped.line_wavelength.item() == pytest.approx(171.073)
    assert mapped.channel.item() == 171
    assert "spectral_response" not in mapped
    assert "wavelength_bin" not in mapped.dims
    assert mapped.attrs["HISTORY"][-1].startswith("map_response_to_sg_detector(")
    xr.testing.assert_identical(response, original)


def test_map_response_to_sg_detector_keeps_chunked_input_lazy():
    response = _spectral_response()
    kwargs = {
        "number_of_slits": 2,
        "dispersion": 0.1 * u.AA / u.pix,
        "slit_spacing": 2 * u.pix,
        "detector_pixels": 4,
        "wavelength_start": 170 * u.AA,
        "pixel_width": 1 * u.arcsec,
        "pixel_height": 2 * u.arcsec,
    }

    lazy = map_response_to_sg_detector(response.chunk({"doppler_velocity": 1}), 171, **kwargs)
    eager = map_response_to_sg_detector(response, 171, **kwargs)

    assert isinstance(lazy.detector_response.data, da.Array)
    xr.testing.assert_allclose(lazy.compute(), eager)


def test_map_response_to_sg_detector_preserves_constant_photon_density_integral():
    response = _spectral_response()
    solid_angle = (1 * u.arcsec).to_value(u.rad) ** 2
    photon_energy = (const.h * const.c / (response.wavelength_grid.values * u.AA)).to_value(u.erg)
    response["spectral_response"] = (
        response.spectral_response.dims,
        np.broadcast_to(photon_energy / solid_angle, response.spectral_response.shape),
        response.spectral_response.attrs,
    )

    mapped = map_response_to_sg_detector(
        response,
        171,
        number_of_slits=1,
        dispersion=0.01 * u.AA / u.pix,
        detector_pixels=100,
        wavelength_start=170 * u.AA,
        pixel_width=1 * u.arcsec,
        pixel_height=1 * u.arcsec,
    )

    np.testing.assert_allclose(mapped.detector_response.sum("detector_x_pixel"), 1.0, rtol=1e-12)


def test_map_response_to_sg_detector_uses_muse_defaults():
    mapped = map_response_to_sg_detector(_spectral_response(), 171)

    assert mapped.sizes["slit"] == DEFAULTS_MUSE.number_of_slits_SG
    assert mapped.sizes["detector_x_pixel"] == DEFAULTS_MUSE.pixels_SG.to_value(u.pix)
    expected_start = u.Quantity(DEFAULTS_MUSE.initial_wavelength_SG.sel(channel=171).data).to_value(u.AA)
    assert mapped.detector_wavelength.isel(slit=0, detector_x_pixel=0).item() == pytest.approx(expected_start)
    dispersion = (
        2
        * DEFAULTS_MUSE.spectral_slit_separation_SG
        / DEFAULTS_MUSE.pixels_between_slits
        / DEFAULTS_MUSE.channel_spectral_order.sel(channel=171).item()
    ).to_value(u.AA / u.pix)
    assert mapped.detector_wavelength.isel(slit=0, detector_x_pixel=-1).item() == pytest.approx(
        expected_start + (DEFAULTS_MUSE.pixels_SG.to_value(u.pix) - 1) * dispersion
    )


def test_map_response_to_sg_detector_gives_contaminants_a_line_reference():
    mapped = map_response_to_sg_detector(
        _spectral_response(contaminants=True),
        171,
        number_of_slits=1,
        detector_pixels=1,
        wavelength_start=171 * u.AA,
    )

    np.testing.assert_allclose(mapped.line_wavelength, [171.073, 171.073])
    np.testing.assert_array_equal(mapped.component_kind, ["line", "contaminants"])


@pytest.mark.parametrize("component_kind", [None, ["line", "line"]])
def test_map_response_to_sg_detector_preserves_invalid_physical_line_wavelength(component_kind):
    response = _spectral_response(contaminants=True)
    if component_kind is None:
        response = response.drop_vars("component_kind")
    else:
        response = response.assign_coords(component_kind=("line", component_kind))

    mapped = map_response_to_sg_detector(response, 171, number_of_slits=1, detector_pixels=1)

    np.testing.assert_allclose(mapped.line_wavelength, [171.073, np.nan])


def test_map_response_to_sg_detector_preserves_input_nan():
    response = _spectral_response()
    response.spectral_response.data[..., response.sizes["wavelength_bin"] // 2] = np.nan

    mapped = map_response_to_sg_detector(
        response,
        171,
        number_of_slits=1,
        dispersion=0.01 * u.AA / u.pix,
        detector_pixels=response.sizes["wavelength_bin"],
        wavelength_start=response.wavelength_grid[0].item() * u.AA,
    )

    assert bool(mapped.detector_response.isnull().any())
    assert mapped.detector_response.isel(detector_x_pixel=-1).item() == 0


def test_map_response_to_sg_detector_requires_effective_area():
    response = _spectral_response()
    response.spectral_response.attrs["units"] = "1e-27 erg cm3 / (Angstrom s sr)"

    with pytest.raises(ValueError, match="convertible"):
        map_response_to_sg_detector(response, 171)


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("response_type", TypeError, "xarray.Dataset"),
        ("channel", ValueError, "unsupported MUSE SG channel"),
        ("schema", ValueError, "missing required variables"),
        ("normalization", ValueError, "normalization"),
        ("wavelength_grid", ValueError, "strictly increasing"),
        ("wavelength_grid_empty", ValueError, "wavelength_grid"),
        ("wavelength_grid_positive", ValueError, "positive"),
        ("slit_spacing", ValueError, "slit_spacing"),
        ("geometry", ValueError, "number_of_slits"),
    ],
)
def test_map_response_to_sg_detector_rejects_invalid_inputs(case, error, match):
    response = _spectral_response()
    channel = 171
    kwargs = {}
    if case == "response_type":
        response = None
    elif case == "channel":
        channel = 195
    elif case == "schema":
        response = response.drop_vars("line_wavelength")
    elif case == "normalization":
        response.attrs["normalization"] = 0
    elif case == "wavelength_grid":
        response = response.assign_coords(wavelength_grid=response.wavelength_grid[::-1])
    elif case == "wavelength_grid_empty":
        response = response.isel(wavelength_bin=slice(0, 0))
    elif case == "wavelength_grid_positive":
        response = response.assign_coords(wavelength_grid=response.wavelength_grid - 200)
    elif case == "slit_spacing":
        kwargs["slit_spacing"] = 0 * u.pix
    else:
        kwargs["number_of_slits"] = 0

    with pytest.raises(error, match=match):
        map_response_to_sg_detector(response, channel, **kwargs)


@pytest.mark.filterwarnings(
    "ignore:numpy.ndarray size changed:RuntimeWarning",
    "ignore:Setting the shape on a NumPy array has been deprecated in NumPy:DeprecationWarning: ",
)
def test_public_response_workflow_maps_directly_into_synthesis(monkeypatch, tmp_path):
    generated_line_list = xr.Dataset(
        {
            "wavelength": ("trans_index", [171.073], {"units": "Angstrom"}),
            "atomic_number": ("trans_index", [26]),
            "gofnt": (
                ("logT", "trans_index"),
                [[1e-25]],
                {"units": "erg cm3 / (s sr)"},
            ),
            "full_name": ("trans_index", ["Fe IX 171.073"]),
        },
        coords={"logT": [6.0]},
    )

    fake_chianti = SimpleNamespace(
        bunch=lambda *_args, **_kwargs: SimpleNamespace(AbundanceName="test.abund"),
    )
    monkeypatch.setattr(linelist_module, "_initialize_chianti", lambda: ("test", fake_chianti))
    monkeypatch.setattr(
        linelist_module,
        "_chianti_bunch_to_dataset",
        lambda *_args, **_kwargs: generated_line_list.copy(deep=True),
    )
    line_list = create_chianti_line_list(
        xr.DataArray([1e6] * u.K, dims="logT"),
        pressure=xr.DataArray([3e15] * u.K / u.cm**3, dims="pressure"),
        wavelength_range=[170.0, 172.0] * u.AA,
        ion_list=["fe_9"],
    )
    effective_area = xr.DataArray(
        [10.0, 10.0],
        dims="wavelength",
        coords={"wavelength": ("wavelength", [170.0, 172.0], {"units": "Angstrom"})},
        attrs={"units": "cm2"},
    )
    spectral = create_spectral_response(
        line_list,
        np.linspace(170.5, 171.5, 101) * u.AA,
        main_lines=["Fe IX 171.073"],
        doppler_velocity=[0.0] * u.km / u.s,
        effective_area=effective_area,
    )
    assert "slit" not in spectral.dims
    response = map_response_to_sg_detector(
        spectral,
        171,
        number_of_slits=2,
        dispersion=0.01 * u.AA / u.pix,
        slit_spacing=2 * u.pix,
        detector_pixels=5,
        wavelength_start=171.05 * u.AA,
    )
    raster = xr.Dataset(
        {
            "vdem": (
                ("logT", "vdop", "slit"),
                np.ones((1, 1, 2)),
                {"units": "1e27 / cm5"},
            )
        },
        coords={"logT": response.logT, "vdop": response.vdop, "slit": response.slit},
    )
    path = tmp_path / "response.nc"
    response.to_netcdf(path)
    loaded_response = read_response(path).load()
    loaded_response.close()

    synthesized = vdem_synthesis(raster, loaded_response)

    assert synthesized.flux.dims == ("line", "detector_x_pixel")
    assert np.isfinite(synthesized.flux).all()
    assert bool((synthesized.flux > 0).any())
    assert synthesized.line_wavelength.item() == pytest.approx(171.073)
