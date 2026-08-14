import pickle

import attrs
import numpy as np
import pytest
import xarray as xr
from attrs.exceptions import FrozenInstanceError

import astropy.units as u
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.units import imperial

from muse.variables import DEFAULTS_MUSE
from muse.variables_schema import InstrumentDefaults


def test_instrument_defaults_reject_top_level_reassignment():
    with pytest.raises(FrozenInstanceError, match="can't set attribute"):
        DEFAULTS_MUSE.ccd_gain_SG = 20 * u.electron / u.DN


def test_instrument_mapping_fields_are_read_only_and_copied():
    mesh_transmission = {284: 0.81}
    defaults = InstrumentDefaults(mesh_transmission=mesh_transmission)

    mesh_transmission[284] = 0.5

    assert defaults.mesh_transmission[284] == 0.81
    with pytest.raises(TypeError, match="FrozenDict is read-only"):
        defaults.mesh_transmission[284] = 0.5


def test_instrument_array_fields_are_read_only_and_copied():
    bands = np.array([108, 171, 284]) * u.AA
    defaults = InstrumentDefaults(bands_SG=bands)

    bands[0] = 999 * u.AA

    np.testing.assert_array_equal(defaults.bands_SG.value, [108, 171, 284])
    with pytest.raises(ValueError, match="read-only"):
        defaults.bands_SG[0] = 999 * u.AA


def test_instrument_nested_array_mappings_are_read_only_and_copied():
    target_logt = np.array([4.8, 5.0, 5.2])
    defaults = InstrumentDefaults(target_logT={"QS": target_logt})

    target_logt[0] = 99

    np.testing.assert_array_equal(defaults.target_logT["QS"], [4.8, 5.0, 5.2])
    with pytest.raises(ValueError, match="read-only"):
        defaults.target_logT["QS"][0] = 99


def test_instrument_quantity_mappings_are_read_only_and_copied():
    exposure_times = np.array([2.0, 6.0]) * u.s
    defaults = InstrumentDefaults(exposure_times_SG={"QS": exposure_times})

    exposure_times[0] = 99 * u.s

    np.testing.assert_array_equal(defaults.exposure_times_SG["QS"].value, [2.0, 6.0])
    with pytest.raises(ValueError, match="read-only"):
        defaults.exposure_times_SG["QS"][0] = 99 * u.s


def test_instrument_nested_sequences_are_converted_to_tuples():
    main_lines = [["Fe XIX 108.355", "Fe XXI 108.117"]]
    target_logt = [1, 2, 3]
    defaults = InstrumentDefaults(main_lines_SG=main_lines, target_logT={"QS": target_logt})

    main_lines[0][0] = "changed"
    target_logt[0] = 99

    assert defaults.main_lines_SG == (("Fe XIX 108.355", "Fe XXI 108.117"),)
    assert defaults.target_logT["QS"] == (1, 2, 3)
    with pytest.raises(TypeError, match=r"tuple.*does not support item assignment"):
        defaults.main_lines_SG[0][0] = "changed"


def test_instrument_defaults_use_evolve_for_overrides():
    updated = attrs.evolve(DEFAULTS_MUSE, dx_pixel_CI=1 * u.arcmin)

    assert updated.dx_pixel_CI == 60 * u.arcsec
    assert DEFAULTS_MUSE.dx_pixel_CI == 0.143 * u.arcsec
    assert updated != DEFAULTS_MUSE


def test_ci_calibration_defaults_cover_both_channels():
    np.testing.assert_array_equal(DEFAULTS_MUSE.ccd_gain_CI.channel, [195, 304])
    np.testing.assert_allclose(DEFAULTS_MUSE.ccd_gain_CI.data.value, [10.0, 10.0])
    np.testing.assert_array_equal(DEFAULTS_MUSE.main_line_effective_area_CI.channel, [195, 304])
    np.testing.assert_allclose(DEFAULTS_MUSE.main_line_effective_area_CI.data.value, [4.55, 4.55])


def test_instrumental_width_sg():
    width = DEFAULTS_MUSE.instrumental_width_SG

    assert DEFAULTS_MUSE.instrumental_fwhm_SG == 0.0815 * u.AA
    np.testing.assert_allclose(width.sel(channel=284).data.to_value(u.AA), 0.0815 / gaussian_sigma_to_fwhm)
    np.testing.assert_allclose(width.sel(channel=108).data.to_value(u.AA), 0.0815 / gaussian_sigma_to_fwhm / 2)


def test_instrumental_fwhm_sg_normalizes_to_angstrom():
    defaults = attrs.evolve(DEFAULTS_MUSE, instrumental_fwhm_SG=0.00815 * u.nm)

    assert defaults.instrumental_fwhm_SG.unit == u.AA
    np.testing.assert_allclose(defaults.instrumental_fwhm_SG.value, 0.0815)


def test_instrumental_width_sg_requires_fwhm():
    defaults = attrs.evolve(DEFAULTS_MUSE, instrumental_fwhm_SG=None)

    with pytest.raises(ValueError, match="requires instrumental_fwhm_SG"):
        defaults.instrumental_width_SG  # NOQA: B018


def test_instrument_defaults_pickle_round_trip():
    loaded = pickle.loads(pickle.dumps(DEFAULTS_MUSE))  # NOQA: S301

    assert loaded == DEFAULTS_MUSE


def test_instrument_int_and_bool_fields_accept_numpy_scalars():
    defaults = InstrumentDefaults(number_of_slits_SG=np.int64(35), sum_lines=np.True_)

    assert defaults.number_of_slits_SG == 35
    assert isinstance(defaults.number_of_slits_SG, int)
    assert defaults.sum_lines


def test_instrument_int_fields_reject_floats():
    with pytest.raises(TypeError, match="number_of_slits_SG"):
        InstrumentDefaults(number_of_slits_SG=35.0)


def test_instrument_int_fields_reject_bools():
    with pytest.raises(TypeError, match="integer"):
        InstrumentDefaults(number_of_slits_SG=True)
    with pytest.raises(TypeError, match="integer"):
        InstrumentDefaults(oversample_x_SG=np.False_)


def test_instrument_defaults_are_unhashable():
    with pytest.raises(TypeError, match="unhashable"):
        hash(DEFAULTS_MUSE)


def test_instrument_quantity_converter_preserves_dtype_for_matching_unit():
    assert DEFAULTS_MUSE.pixels_SG.dtype.kind == "i"


def test_instrument_quantity_converter_normalizes_units():
    defaults = InstrumentDefaults(
        spectral_slit_separation_SG=390.0 * u.mAA,
        main_lines_SG_wavelength={"Fe IX 171.073": 171073 * u.mAA},
        target_doppler_velocity={"QS": np.array([1000, 2000]) * u.m / u.s},
        initial_wavelength_SG=xr.DataArray(
            [107680.34, 170623.14, 283016.08] * u.mAA,
            coords={"channel": [108, 171, 284]},
            dims="channel",
        ),
        main_line_effective_area_CI=xr.DataArray(
            [4.55e-4] * u.m**2,
            coords={"channel": [195]},
            dims="channel",
        ),
        pair_creation_energy_CI=xr.DataArray(
            [0.00365, 0.00365] * u.keV / u.electron,
            coords={"channel": [195, 304]},
            dims="channel",
        ),
        lpi={284: 70 / imperial.inch},
    )

    assert defaults.spectral_slit_separation_SG.unit == u.AA
    np.testing.assert_allclose(defaults.spectral_slit_separation_SG.value, 0.39)
    assert defaults.main_lines_SG_wavelength["Fe IX 171.073"].unit == u.AA
    assert defaults.target_doppler_velocity["QS"].unit == u.km / u.s
    assert defaults.initial_wavelength_SG.data.unit == u.AA
    assert defaults.main_line_effective_area_CI.data.unit == u.cm**2
    assert defaults.pair_creation_energy_CI.data.unit == u.eV / u.electron
    np.testing.assert_allclose(defaults.main_line_effective_area_CI.data.value, [4.55])
    np.testing.assert_allclose(defaults.pair_creation_energy_CI.data.value, [3.65, 3.65])
    assert defaults.lpi[284].unit == 1 / imperial.inch


def test_instrument_quantity_fields_reject_unitless_physical_values():
    with pytest.raises(u.UnitsError, match="arcsec"):
        InstrumentDefaults(psf_fwhm=0.5)

    unitless_wavelength = xr.DataArray([107.68034, 170.62314], coords={"channel": [108, 171]}, dims="channel")
    with pytest.raises(u.UnitsError, match="DataArray values must have units convertible to Angstrom"):
        InstrumentDefaults(initial_wavelength_SG=unitless_wavelength)


def test_instrument_defaults_validate_spectral_channels():
    initial = xr.DataArray([107.68034, 170.62314] * u.AA, coords={"channel": [108, 171]}, dims="channel")
    mismatched_order = xr.DataArray([2, 1], coords={"channel": [108, 284]}, dims="channel")

    with pytest.raises(ValueError, match="matching channel coordinates"):
        InstrumentDefaults(initial_wavelength_SG=initial, channel_spectral_order_SG=mismatched_order)

    order = xr.DataArray([2, 1], coords={"channel": [108, 171]}, dims="channel")
    with pytest.raises(ValueError, match="bands_SG unique channels"):
        InstrumentDefaults(initial_wavelength_SG=initial, channel_spectral_order_SG=order, bands_SG=[108, 284] * u.AA)

    # Every per-channel DataArray field is checked, not just the two spectral ones.
    mismatched_pair_energy = xr.DataArray([3.65] * u.eV / u.electron, coords={"channel": [284]}, dims="channel")
    with pytest.raises(ValueError, match="matching channel coordinates"):
        InstrumentDefaults(initial_wavelength_SG=initial, pair_creation_energy_SG=mismatched_pair_energy)


@pytest.mark.parametrize(
    ("field", "unit"),
    [
        ("ccd_gain_CI", u.electron / u.DN),
        ("main_line_effective_area_CI", u.cm**2),
        ("pair_creation_energy_CI", u.eV / u.electron),
    ],
)
def test_ci_calibrations_require_channel_dimension(field, unit):
    calibration = xr.DataArray([1] * unit, coords={"wavelength": [195]}, dims="wavelength")

    with pytest.raises(ValueError, match=rf"{field} must have a 'channel' dimension"):
        InstrumentDefaults(**{field: calibration})


def test_instrument_defaults_validate_spectral_lines():
    with pytest.raises(ValueError, match="one entry for each line"):
        InstrumentDefaults(main_lines_SG=[["Fe IX 171.073"], ["Fe XV 284.163"]], bands_SG=[171] * u.AA)

    with pytest.raises(ValueError, match="main_lines_SG_wavelength is missing entries"):
        InstrumentDefaults(
            main_lines_SG=[["Fe IX 171.073"]],
            main_lines_SG_wavelength={"Fe XV 284.163": 284.163 * u.AA},
        )


def test_instrument_defaults_validate_related_mapping_keys():
    with pytest.raises(ValueError, match="target_logT and target_doppler_velocity must use matching keys"):
        InstrumentDefaults(target_logT={"QS": [4.8, 4.9]}, target_doppler_velocity={"AR": [-200, -100] * u.km / u.s})

    with pytest.raises(ValueError, match="lpi and mesh_transmission must use matching keys"):
        InstrumentDefaults(lpi={171: 70 / imperial.inch}, mesh_transmission={284: 0.81})
