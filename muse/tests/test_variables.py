import attrs
import numpy as np
import pytest
from attrs.exceptions import FrozenInstanceError

import astropy.units as u

from muse.variables import DEFAULTS_INSTRUMENT
from muse.variables_schema import InstrumentDefaults


def test_instrument_defaults_are_frozen():
    with pytest.raises(FrozenInstanceError):
        DEFAULTS_INSTRUMENT.ccd_gain = 20 * u.electron / u.DN


def test_instrument_mapping_fields_are_immutable_and_copied():
    mesh_transmission = {284: 0.81}
    defaults = InstrumentDefaults(mesh_transmission=mesh_transmission)

    mesh_transmission[284] = 0.5

    assert defaults.mesh_transmission[284] == 0.81
    with pytest.raises(TypeError):
        defaults.mesh_transmission[284] = 0.5


def test_instrument_array_fields_are_read_only_and_copied():
    bands = np.array([108, 171, 284])
    defaults = InstrumentDefaults(bands_SG=bands)

    bands[0] = 999

    np.testing.assert_array_equal(defaults.bands_SG, [108, 171, 284])
    with pytest.raises(ValueError, match="read-only"):
        defaults.bands_SG[0] = 999


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


def test_instrument_defaults_use_evolve_for_overrides():
    updated = attrs.evolve(DEFAULTS_INSTRUMENT, dx_pixel_CI=1 * u.arcmin)

    assert updated.dx_pixel_CI == 60 * u.arcsec
    assert DEFAULTS_INSTRUMENT.dx_pixel_CI == 0.143 * u.arcsec
    assert updated != DEFAULTS_INSTRUMENT
