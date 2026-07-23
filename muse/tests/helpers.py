import warnings
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
    "fake_response",
    "fake_response_file",
    "fake_vdem",
    "fake_vdem_offgrid",
    "fake_vdem_single_vdop",
    "warnings_as_errors",
]


@pytest.fixture
def warnings_as_errors():
    warnings.simplefilter("error")
    yield
    warnings.resetwarnings()


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


nslit = 35
steps = 11
nx = steps * nslit
ny = 32
npixel = 32
lgtaxis = np.asarray([4.4, 4.9, 5.4, 5.9, 6.4, 6.9, 7.4])
dopaxis = np.arange(-400.0, 401.0, 100.0)
nlgtaxis = np.size(lgtaxis)
ndopaxis = np.size(dopaxis)
x_axis = np.linspace(0.0, 154.0, nx)
y_axis = np.arange(ny) * (67.468 / 403)
line = np.asarray(
    [
        DEFAULTS_MUSE.main_lines_SG[0][0],
        DEFAULTS_MUSE.main_lines_SG[0][1],
        "108 remaining 10000 lines",
        DEFAULTS_MUSE.main_lines_SG[1][0],
        "171 remaining 10000 lines",
        DEFAULTS_MUSE.main_lines_SG[2][0],
        "284 remaining 10000 lines",
    ]
)

nline = np.size(line)
channel = np.asarray([108, 108, 108, 171, 171, 284, 284])
line_wavelength = np.asarray(
    [
        DEFAULTS_MUSE.main_lines_SG_wavelength["Fe XIX 108.355"].to_value(u.AA),
        DEFAULTS_MUSE.main_lines_SG_wavelength["Fe XXI 108.117"].to_value(u.AA),
        DEFAULTS_MUSE.main_lines_SG_wavelength["Fe XIX 108.355"].to_value(u.AA),
        DEFAULTS_MUSE.main_lines_SG_wavelength["Fe IX 171.073"].to_value(u.AA),
        DEFAULTS_MUSE.main_lines_SG_wavelength["Fe IX 171.073"].to_value(u.AA),
        DEFAULTS_MUSE.main_lines_SG_wavelength["Fe XV 284.163"].to_value(u.AA),
        DEFAULTS_MUSE.main_lines_SG_wavelength["Fe XV 284.163"].to_value(u.AA),
    ]
)
slit = np.arange(0, nslit, 1)
# The real response functions have 1024 detector pixels. The fixture samples 32
# of them for speed, so line peaks land on the nearest sampled pixel. At slit=17
# and vdop=0 km/s, sampled peak offsets are about -0.021 A (Fe XIX/108 rem.),
# +0.217 A (Fe XXI), +0.204 A (Fe IX/171 rem.), and +0.161 A (Fe XV/284 rem.).
SG_XPIXEL = np.linspace(0, 1023, npixel, dtype=int)
initial_wavelength_sg = {108: 107.68034, 171: 170.62314, 284: 283.01608}
spectral_dispersion_sg = {108: 0.014714709071675623, 171: 0.014714709071675623, 284: 0.029429418143351246}
spectral_slit_offset_sg = {108: 0.39, 171: 0.39, 284: 0.78}
response_logt_center = np.asarray([6.9, 7.1, 6.2, 5.9, 6.0, 6.4, 6.1])
response_logt_width = np.asarray([0.22, 0.22, 0.45, 0.25, 0.45, 0.28, 0.45])
response_amplitude = np.asarray([1.0, 0.85, 0.08, 0.65, 0.06, 0.75, 0.07])
response_spectral_width = np.asarray([0.35, 0.35, 0.8, 0.35, 0.8, 0.55, 1.1])


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


def calculate_sgwvl(line_index):
    line_channel = int(channel[line_index])
    return (
        initial_wavelength_sg[line_channel]
        + spectral_dispersion_sg[line_channel] * SG_XPIXEL[np.newaxis, :]
        - spectral_slit_offset_sg[line_channel] * slit[:, np.newaxis]
    )


def fake_vdem():
    logt_grid = lgtaxis[:, np.newaxis, np.newaxis, np.newaxis]
    vdop_grid = dopaxis[np.newaxis, :, np.newaxis, np.newaxis]
    y_grid = y_axis[np.newaxis, np.newaxis, :, np.newaxis]
    x_grid = x_axis[np.newaxis, np.newaxis, np.newaxis, :]

    x_window = 1 / (1 + np.exp(-(x_grid - 12.0) / 5.0)) * 1 / (1 + np.exp((x_grid - 145.0) / 5.0))
    y_ridge = 3.0 + 0.7 * np.sin(2 * np.pi * x_grid / x_axis[-1])
    flare_core = (
        480.0
        * np.exp(
            -(((logt_grid - 6.8) / 0.22) ** 2) - ((vdop_grid + 90.0) / 85.0) ** 2 - ((y_grid - y_ridge) / 0.48) ** 2
        )
        * x_window
    )
    warm_arcade = 18.0 * np.exp(
        -(((logt_grid - 6.1) / 0.35) ** 2)
        - (vdop_grid / 220.0) ** 2
        - ((x_grid - 78.0) / 48.0) ** 2
        - ((y_grid - 2.2) / 2.6) ** 2
    )
    transition_region = (
        0.05 * np.exp(-(((logt_grid - 5.4) / 0.35) ** 2)) * (1.0 + 0.2 * np.sin(2 * np.pi * x_grid / x_axis[-1]))
    )
    table = flare_core + warm_arcade + transition_region
    table = np.where(table < 1e-6, 0.0, table)
    ds = xr.Dataset(
        data_vars={"vdem": (["logT", "vdop", "y", "x"], table)},
        coords={"logT": lgtaxis, "vdop": dopaxis, "y": y_axis, "x": x_axis},
        attrs={
            "description": "DEM(T,vel,x,y)",
        },
    )
    ds.vdem.attrs["description"] = "DEM(T,vel,x,y)"
    ds.vdem.attrs["time"] = 555291
    ds.vdem.attrs["units"] = "1e27 / cm5"
    ds.logT.attrs["long_name"] = "log$_{10}$(T)"
    ds.logT.attrs["units"] = "log$_{10}$ (K)"
    ds.vdop.attrs["long_name"] = "v$_{Doppler}$"
    ds.vdop.attrs["units"] = "km/s"
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


def fake_vdem_single_vdop(vdop_kms=0.0):
    """
    Return `fake_vdem` with all emission collapsed into the vdop bin nearest
    ``vdop_kms``.

    This isolates one Doppler velocity so synthesized line centroids shift by the
    classical ``lambda * v / c``.
    """
    ds = fake_vdem()
    vdop_index = int(np.argmin(np.abs(ds.vdop.values - vdop_kms)))
    collapsed = np.zeros_like(ds.vdem.values)
    collapsed[:, vdop_index] = ds.vdem.values.sum(axis=1)  # all vdop emission -> one bin
    ds["vdem"] = (ds.vdem.dims, collapsed, dict(ds.vdem.attrs))
    return ds


def fake_response():
    table_resp = np.zeros((nline, ndopaxis, nlgtaxis, nslit, npixel))
    table_sgwvl = np.asarray([calculate_sgwvl(line_index) for line_index in range(nline)])
    speed_of_light_kms = 299792.458
    slit_throughput = 1.0 + 0.08 * np.cos(np.pi * slit / nslit)

    for line_index in range(nline):
        shifted_line_wavelength = line_wavelength[line_index] * (
            1.0 + dopaxis[:, np.newaxis, np.newaxis] / speed_of_light_kms
        )
        spectral_profile = np.exp(
            -(
                (
                    (table_sgwvl[line_index][np.newaxis, :, :] - shifted_line_wavelength)
                    / response_spectral_width[line_index]
                )
                ** 2
            )
        )
        temperature_profile = np.exp(
            -(((lgtaxis - response_logt_center[line_index]) / response_logt_width[line_index]) ** 2)
        )
        table_resp[line_index] = (
            response_amplitude[line_index]
            * spectral_profile[:, np.newaxis, :, :]
            * temperature_profile[np.newaxis, :, np.newaxis, np.newaxis]
            * slit_throughput[np.newaxis, np.newaxis, :, np.newaxis]
        )

    response = xr.Dataset(
        data_vars={
            "detector_response": (["line", "vdop", "logT", "slit", "detector_x_pixel"], table_resp),
        },
        coords={"logT": lgtaxis, "vdop": dopaxis, "line": line, "slit": slit, "detector_x_pixel": SG_XPIXEL},
        attrs={"description": "No attributes"},
    )
    response = response.assign_coords(
        line_wavelength=("line", line_wavelength),
        detector_wavelength=(["line", "slit", "detector_x_pixel"], table_sgwvl),
    )
    response.line_wavelength.attrs["units"] = "Angstrom"
    response.logT.attrs["long_name"] = "log$_{10}$(T)"
    response.logT.attrs["units"] = "log$_{10}$ (K)"
    response.vdop.attrs["long_name"] = "v$_{Doppler}$"
    response.vdop.attrs["units"] = "km/s"
    response.detector_response.attrs["units"] = "1e-27 ph cm5 / s"
    response.detector_wavelength.attrs["units"] = "Angstrom"
    return response.assign_coords(channel=("line", channel))


def fake_response_file():
    """
    Response dataset shaped like the real on-disk MUSE response files, for IO tests.
    """
    n_logT, n_vdop, n_slit, n_pixel, n_wave = 5, 7, 4, 8, 6
    logT_axis = np.linspace(5.0, 7.0, n_logT)
    vdop_axis = np.linspace(-300.0, 300.0, n_vdop)
    slit_axis = np.arange(n_slit)
    sg_xpixel = np.arange(n_pixel)
    wavelength = np.linspace(160.0, 180.0, n_wave)

    # Smooth, non-negative response: Gaussian in logT x vdop, gently modulated by slit/pixel.
    logT_profile = np.exp(-(((logT_axis - 6.0) / 0.4) ** 2))
    vdop_profile = np.exp(-((vdop_axis / 150.0) ** 2))
    slit_profile = 1.0 + 0.05 * np.cos(np.pi * slit_axis / n_slit)
    pixel_profile = np.exp(-(((sg_xpixel - n_pixel / 2.0) / 3.0) ** 2))
    # Dimension order matches the real files:
    # (pressure, logT, line, vdop, abundance, slit, SG_xpixel).
    sg_resp = (
        logT_profile[None, :, None, None, None, None, None]
        * vdop_profile[None, None, None, :, None, None, None]
        * slit_profile[None, None, None, None, None, :, None]
        * pixel_profile[None, None, None, None, None, None, :]
    )
    sg_wvl = 170.62314 + 0.014714709 * sg_xpixel[:, np.newaxis] - 0.39 * slit_axis[np.newaxis, :]
    effective_area = np.linspace(0.1, 1.0, n_wave)[np.newaxis, :]

    response = xr.Dataset(
        data_vars={
            "SG_resp": (["pressure", "logT", "line", "vdop", "abundance", "slit", "SG_xpixel"], sg_resp),
            "effective_area": (["pressure", "wavelength"], effective_area),
        },
        coords={
            "logT": logT_axis,
            "vdop": vdop_axis,
            "slit": slit_axis,
            "SG_xpixel": sg_xpixel,
            "wavelength": wavelength,
            "pressure": [3.0e15],
            "abundance": ["sun_coronal_2021_chianti"],
            "line": ["Fe IX 171.073"],
        },
    )
    response = response.assign_coords(
        line_wvl=("line", [171.073]),
        channel=("line", [171]),
        SG_wvl=(["SG_xpixel", "slit"], sg_wvl),
    )
    # Real files carry no units on line_wvl/SG_wvl; the reader is expected to inject Å.
    response.SG_resp.attrs["units"] = "1e-27 cm5 ph / s"
    return response
