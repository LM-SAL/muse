import pickle

import numpy as np
import pytest
import xarray as xr
from attrs.exceptions import FrozenInstanceError

import astropy.units as u

from muse.variables import DEFAULTS_MUSE
from muse.variables_schema import InstrumentDefaults


def test_instrument_defaults_reject_top_level_reassignment():
    with pytest.raises(FrozenInstanceError, match="can't set attribute"):
        DEFAULTS_MUSE.gain_SG = 20 * u.electron / u.DN


def test_instrument_array_fields_are_read_only_and_copied():
    velocities = np.array([-100.0, 0.0, 100.0]) * u.km / u.s
    defaults = InstrumentDefaults(response_doppler_velocity=velocities)

    velocities[0] = 999 * u.km / u.s

    np.testing.assert_array_equal(defaults.response_doppler_velocity.value, [-100.0, 0.0, 100.0])
    with pytest.raises(ValueError, match="read-only"):
        defaults.response_doppler_velocity[0] = 999 * u.km / u.s


def test_instrument_nested_sequences_are_converted_to_tuples():
    main_lines = [["Fe XIX 108.355", "Fe XXI 108.117"]]
    defaults = InstrumentDefaults(main_lines_by_channel_SG=main_lines)

    main_lines[0][0] = "changed"

    assert defaults.main_lines_by_channel_SG == (("Fe XIX 108.355", "Fe XXI 108.117"),)


def test_instrument_defaults_pickle_round_trip():
    loaded = pickle.loads(pickle.dumps(DEFAULTS_MUSE))  # NOQA: S301

    assert loaded == DEFAULTS_MUSE


def test_instrument_int_fields_accept_numpy_scalars():
    defaults = InstrumentDefaults(number_of_slits_SG=np.int64(35))

    assert defaults.number_of_slits_SG == 35
    assert isinstance(defaults.number_of_slits_SG, int)


def test_instrument_int_fields_reject_floats_and_bools():
    with pytest.raises(TypeError, match="number_of_slits_SG must be an integer"):
        InstrumentDefaults(number_of_slits_SG=35.0)
    with pytest.raises(TypeError, match="number_of_slits_SG must be an integer"):
        InstrumentDefaults(number_of_slits_SG=True)


def test_instrument_quantity_converter_preserves_dtype_for_matching_unit():
    assert DEFAULTS_MUSE.detector_x_pixels_SG.dtype.kind == "i"


def test_instrument_quantity_converter_normalizes_units():
    # one field of each converter: a DataArray and a Quantity
    defaults = InstrumentDefaults(
        main_line_effective_area_CI=xr.DataArray([4.55e-4] * u.m**2, coords={"channel": [195]}, dims="channel"),
        pixel_width_SG=1 * u.arcmin,
    )

    assert defaults.main_line_effective_area_CI.data.unit == u.cm**2
    np.testing.assert_allclose(defaults.main_line_effective_area_CI.data.value, [4.55])
    assert defaults.pixel_width_SG.unit == u.arcsec
    np.testing.assert_allclose(defaults.pixel_width_SG.value, 60.0)


def test_instrument_quantity_fields_reject_unitless_physical_values():
    with pytest.raises(u.UnitsError, match="arcsec"):
        InstrumentDefaults(pixel_width_SG=0.5)

    unitless_wavelength = xr.DataArray([107.68034, 170.62314], coords={"channel": [108, 171]}, dims="channel")
    with pytest.raises(u.UnitsError, match="DataArray values must have units convertible to Angstrom"):
        InstrumentDefaults(wavelength_start_SG=unitless_wavelength)


def test_instrument_defaults_validate_spectral_channels():
    initial = xr.DataArray([107.68034, 170.62314] * u.AA, coords={"channel": [108, 171]}, dims="channel")
    # Every per-channel DataArray field is checked, not just the two spectral ones.
    mismatched_pair_energy = xr.DataArray([3.65] * u.eV / u.electron, coords={"channel": [284]}, dims="channel")
    with pytest.raises(ValueError, match="matching channel coordinates"):
        InstrumentDefaults(wavelength_start_SG=initial, pair_energy_SG=mismatched_pair_energy)


def test_ci_calibrations_require_channel_dimension():
    # every field whose name ends in _CI is checked alike; one stands for them
    calibration = xr.DataArray([1] * u.electron / u.DN, coords={"wavelength": [195]}, dims="wavelength")

    with pytest.raises(ValueError, match="gain_CI must have a 'channel' dimension"):
        InstrumentDefaults(gain_CI=calibration)
