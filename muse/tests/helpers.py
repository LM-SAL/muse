from pathlib import Path
from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import astropy
import astropy.units as u

from muse.variables import DEFAULTS_MUSE

__all__ = [
    "assert_dataset_structure",
    "fake_continuum",
    "fake_effective_area",
    "fake_line_list",
    "fake_response",
    "fake_response_file_contents",
    "fake_vdem",
    "fake_vdem_offgrid",
    "fake_vdem_single_doppler_velocity",
]


def fake_line_list(n_lines=2, wavelength=None, logT=None):
    """
    Return a minimal deterministic iron line list.
    """
    wavelength = np.linspace(170.6, 171.4, n_lines) if wavelength is None else np.asarray(wavelength, dtype=float)
    n_lines = wavelength.size
    logT = np.array([5.8, 6.0, 6.2]) if logT is None else np.asarray(logT, dtype=float)
    peaks = np.linspace(1.0, 0.5, n_lines)
    gofnt = peaks[np.newaxis, :] * np.exp(-((logT[:, np.newaxis] - 6.0) ** 2) / 0.02) * 1e-25
    return xr.Dataset(
        {
            "wavelength": ("transition", wavelength, {"units": "Angstrom"}),
            "atomic_number": ("transition", np.full(n_lines, 26)),
            "gofnt": (("logT", "transition"), gofnt, {"units": "erg cm3 / (s sr)"}),
            "full_name": ("transition", [f"Fake Fe {i} {value:.3f}" for i, value in enumerate(wavelength)]),
        },
        coords={"logT": ("logT", logT, {"units": "dex(K)"})},
    )


def fake_continuum(
    temperature, wavelength_grid, *, density=None, pressure=None, abundance=None, minimum_abundance=None, **_kwargs
):
    """
    Return a flat continuum with the call signature and the layout of the continuum
    builders: free-free 1e-24, free-bound half of it, and a two-photon part that is 1e-40
    times the density or the pressure of each plasma point.
    """
    from muse.instrument.atomic.backends import (  # noqa: PLC0415
        _broadcast_plasma_grid,
        _validate_plasma_inputs,
    )
    from muse.instrument.atomic.continuum import _continuum_dataset  # noqa: PLC0415

    temperature, plasma = _validate_plasma_inputs(temperature, density, pressure)
    grid = wavelength_grid.to_value(u.AA)
    temperature_broadcast, _, plasma_coords = _broadcast_plasma_grid(
        temperature, plasma, by_density=density is not None
    )
    base = 1e-24 * np.ones((temperature.size, grid.size))
    two_photon = 1e-40 * plasma.data[:, None, None] * np.ones((plasma.size, temperature.size, grid.size))
    return _continuum_dataset(
        base,
        0.5 * base,
        two_photon,
        temperature=temperature,
        temperature_broadcast=temperature_broadcast,
        plasma_coords=plasma_coords,
        wavelength_grid=grid,
        attrs={"abundance": abundance, "minimum_abundance": minimum_abundance, "fiasco_version": "0.0.test"},
    )


def fake_effective_area(
    values=(1.0, 1.0, 1.0),
    wavelength=(169.0, 171.0, 173.0),
    *,
    area_units="cm2",
    wavelength_units="Angstrom",
):
    """
    Return a minimal deterministic effective area.
    """
    area_attrs = {} if area_units is None else {"units": area_units}
    wavelength_attrs = {} if wavelength_units is None else {"units": wavelength_units}
    return xr.DataArray(
        np.asarray(values),
        dims="wavelength",
        coords={"wavelength": ("wavelength", np.asarray(wavelength), wavelength_attrs)},
        attrs=area_attrs,
    )


def _clean_version(version: str) -> str:
    """
    Collapse dev/rc versions to "dev", otherwise strip the dots.
    """
    return "dev" if ("dev" in version or "rc" in version) else version.replace(".", "")


def get_hash_library_name():
    """
    Generate the hash library name for this env.
    """
    ft2_version = mpl.ft2font.__freetype_version__.replace(".", "")
    mpl_version = _clean_version(mpl.__version__)
    astropy_version = _clean_version(astropy.__version__)
    return f"figure_hashes_mpl_{mpl_version}_ft_{ft2_version}_astropy_{astropy_version}.json"


def figure_test(test_function):
    """
    A decorator for a test that verifies the hash of the current figure or the returned
    figure, with the name of the test function as the hash identifier in the library. A
    PNG is also created in the 'result_image' directory, which is created on the current
    path.

    All such decorated tests are marked with `pytest.mark.mpl_image` for convenient filtering.

    Examples
    --------
    @figure_test
    def test_simple_plot():
        plt.plot([0,1])
    """
    hash_library_name = get_hash_library_name()
    hash_library_file = Path(__file__).parent / hash_library_name

    @pytest.mark.mpl_image_compare(
        hash_library=hash_library_file, savefig_kwargs={"metadata": {"Software": None}}, style="default"
    )
    @wraps(test_function)
    def test_wrapper(*args, **kwargs):
        ret = test_function(*args, **kwargs)
        if ret is None:
            ret = plt.gcf()
        return ret

    return test_wrapper


N_SLITS = 35
N_STEPS = 11
N_X = N_STEPS * N_SLITS
N_Y = 32
N_PIXELS = 32
LOGT_AXIS = np.asarray([4.4, 4.9, 5.4, 5.9, 6.4, 6.9, 7.4])
DOPPLER_VELOCITY_AXIS = np.arange(-400.0, 401.0, 100.0)
N_LOGT = np.size(LOGT_AXIS)
N_DOPPLER_VELOCITY = np.size(DOPPLER_VELOCITY_AXIS)
X_AXIS = np.linspace(0.0, 154.0, N_X)
Y_AXIS = np.arange(N_Y) * (67.468 / 403)
LINE_NAMES = np.asarray(
    [
        DEFAULTS_MUSE.main_lines_by_channel_SG[0][0],
        DEFAULTS_MUSE.main_lines_by_channel_SG[0][1],
        "108 remaining 10000 lines",
        DEFAULTS_MUSE.main_lines_by_channel_SG[1][0],
        "171 remaining 10000 lines",
        DEFAULTS_MUSE.main_lines_by_channel_SG[2][0],
        "284 remaining 10000 lines",
    ]
)

N_LINES = np.size(LINE_NAMES)
LINE_CHANNELS = np.asarray([108, 108, 108, 171, 171, 284, 284])
# The wavelengths in the names of the lines above.
LINE_WAVELENGTHS = np.asarray([108.355, 108.117, 108.355, 171.073, 171.073, 284.163, 284.163])
SLIT_INDICES = np.arange(0, N_SLITS, 1)
# The real response functions have 1024 detector pixels. The fixture samples 32
# of them for speed, so line peaks land on the nearest sampled pixel. At slit=17
# and doppler_velocity=0 km/s, sampled peak offsets are about -0.021 A (Fe XIX/108 rem.),
# +0.217 A (Fe XXI), +0.204 A (Fe IX/171 rem.), and +0.161 A (Fe XV/284 rem.).
DETECTOR_X_PIXEL = np.linspace(0, 1023, N_PIXELS, dtype=int)
INITIAL_WAVELENGTH_SG = {108: 107.68034, 171: 170.62314, 284: 283.01608}
SPECTRAL_DISPERSION_SG = {108: 0.014714709071675623, 171: 0.014714709071675623, 284: 0.029429418143351246}
SPECTRAL_SLIT_OFFSET_SG = {108: 0.39, 171: 0.39, 284: 0.78}
RESPONSE_LOGT_CENTER = np.asarray([6.9, 7.1, 6.2, 5.9, 6.0, 6.4, 6.1])
RESPONSE_LOGT_WIDTH = np.asarray([0.22, 0.22, 0.45, 0.25, 0.45, 0.28, 0.45])
RESPONSE_AMPLITUDE = np.asarray([1.0, 0.85, 0.08, 0.65, 0.06, 0.75, 0.07])
RESPONSE_SPECTRAL_WIDTH = np.asarray([0.35, 0.35, 0.8, 0.35, 0.8, 0.55, 1.1])


def assert_dataset_structure(
    ds: xr.Dataset,
    *,
    data_vars: tuple[str, ...],
    coords: tuple[str, ...],
    sizes: dict[str, int] | None = None,
    finite_vars: tuple[str, ...] = (),
) -> None:
    actual_data_vars = set(ds.data_vars)
    expected_data_vars = set(data_vars)
    if actual_data_vars != expected_data_vars:
        msg = f"data vars differ: expected {expected_data_vars}, got {actual_data_vars}"
        raise AssertionError(msg)

    expected_coords = set(coords)
    actual_coords = set(ds.coords)
    if not expected_coords <= actual_coords:
        msg = f"coords missing: expected at least {expected_coords}, got {actual_coords}"
        raise AssertionError(msg)

    if sizes is not None and dict(ds.sizes) != sizes:
        msg = f"sizes differ: expected {sizes}, got {dict(ds.sizes)}"
        raise AssertionError(msg)

    for name in finite_vars:
        if name not in ds.data_vars:
            msg = f"{name!r} is not a data variable"
            raise AssertionError(msg)
        if not bool(np.isfinite(ds[name]).all()):
            msg = f"{name!r} contains non-finite values"
            raise AssertionError(msg)


def detector_wavelength_of_line(line_index):
    line_channel = int(LINE_CHANNELS[line_index])
    return (
        INITIAL_WAVELENGTH_SG[line_channel]
        + SPECTRAL_DISPERSION_SG[line_channel] * DETECTOR_X_PIXEL[np.newaxis, :]
        - SPECTRAL_SLIT_OFFSET_SG[line_channel] * SLIT_INDICES[:, np.newaxis]
    )


def fake_vdem():
    logT_grid = LOGT_AXIS[:, np.newaxis, np.newaxis, np.newaxis]
    doppler_velocity_grid = DOPPLER_VELOCITY_AXIS[np.newaxis, :, np.newaxis, np.newaxis]
    y_grid = Y_AXIS[np.newaxis, np.newaxis, :, np.newaxis]
    x_grid = X_AXIS[np.newaxis, np.newaxis, np.newaxis, :]

    x_window = 1 / (1 + np.exp(-(x_grid - 12.0) / 5.0)) * 1 / (1 + np.exp((x_grid - 145.0) / 5.0))
    y_ridge = 3.0 + 0.7 * np.sin(2 * np.pi * x_grid / X_AXIS[-1])
    flare_core = (
        480.0
        * np.exp(
            -(((logT_grid - 6.8) / 0.22) ** 2)
            - ((doppler_velocity_grid + 90.0) / 85.0) ** 2
            - ((y_grid - y_ridge) / 0.48) ** 2
        )
        * x_window
    )
    warm_arcade = 18.0 * np.exp(
        -(((logT_grid - 6.1) / 0.35) ** 2)
        - (doppler_velocity_grid / 220.0) ** 2
        - ((x_grid - 78.0) / 48.0) ** 2
        - ((y_grid - 2.2) / 2.6) ** 2
    )
    transition_region = (
        0.05 * np.exp(-(((logT_grid - 5.4) / 0.35) ** 2)) * (1.0 + 0.2 * np.sin(2 * np.pi * x_grid / X_AXIS[-1]))
    )
    table = flare_core + warm_arcade + transition_region
    table = np.where(table < 1e-6, 0.0, table)
    ds = xr.Dataset(
        data_vars={"vdem": (["logT", "doppler_velocity", "y", "x"], table)},
        coords={"logT": LOGT_AXIS, "doppler_velocity": DOPPLER_VELOCITY_AXIS, "y": Y_AXIS, "x": X_AXIS},
        attrs={
            "description": "DEM(T,vel,x,y)",
        },
    )
    ds.vdem.attrs["description"] = "DEM(T,vel,x,y)"
    ds.vdem.attrs["time"] = 555291
    ds.vdem.attrs["units"] = "1e27 / cm5"
    ds.logT.attrs["long_name"] = "log$_{10}$(T)"
    ds.logT.attrs["units"] = "dex(K)"
    ds.doppler_velocity.attrs["long_name"] = "v$_{Doppler}$"
    ds.doppler_velocity.attrs["units"] = "km/s"
    ds.x.attrs["units"] = "arcsec"
    ds.y.attrs["units"] = "arcsec"
    return ds


def fake_vdem_offgrid():
    """
    `fake_vdem` with the spatial grid stretched so the x/y spacing no longer matches the
    MUSE pixel size.

    This forces `match_fov` down its resample/tile path instead of the early return for
    MUSE-sized pixels.
    """
    ds = fake_vdem()
    ds = ds.assign_coords(x=ds.x.values * 2.0, y=ds.y.values * 2.0)
    ds.x.attrs["units"] = "arcsec"
    ds.y.attrs["units"] = "arcsec"
    return ds


def fake_vdem_single_doppler_velocity(doppler_velocity_kms=0.0):
    """
    Return `fake_vdem` with all emission collapsed into the doppler_velocity bin nearest
    ``doppler_velocity_kms``.

    This isolates one Doppler velocity so synthesized line centroids shift by the
    classical ``lambda * v / c``.
    """
    ds = fake_vdem()
    doppler_velocity_index = int(np.argmin(np.abs(ds.doppler_velocity.values - doppler_velocity_kms)))
    collapsed = np.zeros_like(ds.vdem.values)
    collapsed[:, doppler_velocity_index] = ds.vdem.values.sum(axis=1)  # all doppler_velocity emission -> one bin
    ds["vdem"] = (ds.vdem.dims, collapsed, dict(ds.vdem.attrs))
    return ds


def fake_response():
    detector_response = np.zeros((N_LINES, N_DOPPLER_VELOCITY, N_LOGT, N_SLITS, N_PIXELS))
    detector_wavelength = np.asarray([detector_wavelength_of_line(line_index) for line_index in range(N_LINES)])
    speed_of_light_kms = 299792.458
    slit_throughput = 1.0 + 0.08 * np.cos(np.pi * SLIT_INDICES / N_SLITS)

    for line_index in range(N_LINES):
        shifted_line_wavelength = LINE_WAVELENGTHS[line_index] * (
            1.0 + DOPPLER_VELOCITY_AXIS[:, np.newaxis, np.newaxis] / speed_of_light_kms
        )
        spectral_profile = np.exp(
            -(
                (
                    (detector_wavelength[line_index][np.newaxis, :, :] - shifted_line_wavelength)
                    / RESPONSE_SPECTRAL_WIDTH[line_index]
                )
                ** 2
            )
        )
        temperature_profile = np.exp(
            -(((LOGT_AXIS - RESPONSE_LOGT_CENTER[line_index]) / RESPONSE_LOGT_WIDTH[line_index]) ** 2)
        )
        detector_response[line_index] = (
            RESPONSE_AMPLITUDE[line_index]
            * spectral_profile[:, np.newaxis, :, :]
            * temperature_profile[np.newaxis, :, np.newaxis, np.newaxis]
            * slit_throughput[np.newaxis, np.newaxis, :, np.newaxis]
        )

    response = xr.Dataset(
        data_vars={
            "detector_response": (["line", "doppler_velocity", "logT", "slit", "detector_x_pixel"], detector_response),
        },
        coords={
            "logT": LOGT_AXIS,
            "doppler_velocity": DOPPLER_VELOCITY_AXIS,
            "line": LINE_NAMES,
            "slit": SLIT_INDICES,
            "detector_x_pixel": DETECTOR_X_PIXEL,
        },
        attrs={"description": "No attributes"},
    )
    response = response.assign_coords(
        line_wavelength=("line", LINE_WAVELENGTHS),
        detector_wavelength=(["line", "slit", "detector_x_pixel"], detector_wavelength),
    )
    response.line_wavelength.attrs["units"] = "Angstrom"
    response.logT.attrs["long_name"] = "log$_{10}$(T)"
    response.logT.attrs["units"] = "dex(K)"
    response.doppler_velocity.attrs["long_name"] = "v$_{Doppler}$"
    response.doppler_velocity.attrs["units"] = "km/s"
    response.detector_response.attrs["units"] = "1e-27 ph cm5 / s"
    response.detector_wavelength.attrs["units"] = "Angstrom"
    return response.assign_coords(channel=("line", LINE_CHANNELS))


def fake_response_file_contents():
    """
    Response dataset shaped like the on-disk MUSE response files, for IO tests.

    Carries the public names and units, as `muse.instrument.save_response` writes them,
    and the ``pressure``, ``abundance`` and ``effective_area`` extras the files have, in
    the files' dimension order (pressure, logT, line, doppler_velocity, abundance, slit,
    detector_x_pixel).
    """
    n_logT, n_doppler_velocity, n_slit, n_pixel, n_wave = 5, 7, 4, 8, 6
    logT_axis = np.linspace(5.0, 7.0, n_logT)
    doppler_velocity_axis = np.linspace(-300.0, 300.0, n_doppler_velocity)
    slit_axis = np.arange(n_slit)
    detector_x_pixel = np.arange(n_pixel)
    wavelength = np.linspace(160.0, 180.0, n_wave)

    # Smooth, non-negative response: Gaussian in logT x velocity, gently modulated by slit/pixel.
    logT_profile = np.exp(-(((logT_axis - 6.0) / 0.4) ** 2))
    doppler_velocity_profile = np.exp(-((doppler_velocity_axis / 150.0) ** 2))
    slit_profile = 1.0 + 0.05 * np.cos(np.pi * slit_axis / n_slit)
    pixel_profile = np.exp(-(((detector_x_pixel - n_pixel / 2.0) / 3.0) ** 2))
    detector_response = (
        logT_profile[None, :, None, None, None, None, None]
        * doppler_velocity_profile[None, None, None, :, None, None, None]
        * slit_profile[None, None, None, None, None, :, None]
        * pixel_profile[None, None, None, None, None, None, :]
    )
    detector_wavelength = 170.62314 + 0.014714709 * detector_x_pixel[:, np.newaxis] - 0.39 * slit_axis[np.newaxis, :]
    effective_area = np.linspace(0.1, 1.0, n_wave)[np.newaxis, :]

    response = xr.Dataset(
        data_vars={
            "detector_response": (
                ["pressure", "logT", "line", "doppler_velocity", "abundance", "slit", "detector_x_pixel"],
                detector_response,
                {"units": "1e-27 cm5 ph / s"},
            ),
            "effective_area": (["pressure", "wavelength"], effective_area, {"units": "cm2"}),
        },
        coords={
            "logT": ("logT", logT_axis, {"units": "dex(K)"}),
            "doppler_velocity": ("doppler_velocity", doppler_velocity_axis, {"units": "km / s"}),
            "slit": slit_axis,
            "detector_x_pixel": detector_x_pixel,
            "wavelength": ("wavelength", wavelength, {"units": "Angstrom"}),
            "pressure": [3.0e15],
            "abundance": ["sun_coronal_2021_chianti"],
            "line": ["Fe IX 171.073"],
        },
        attrs={"gain": "10.0 electron / DN"},  # as the generators write it
    )
    return response.assign_coords(
        line_wavelength=("line", [171.073], {"units": "Angstrom"}),
        channel=("line", [171]),
        detector_wavelength=(["detector_x_pixel", "slit"], detector_wavelength, {"units": "Angstrom"}),
    )
