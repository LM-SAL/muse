import attrs
import numpy as np
import pytest
from attrs.exceptions import FrozenInstanceError

import astropy.units as u

from muse.variables import DEFAULTS_AIA, DEFAULTS_INSTRUMENT
from muse.variables_schema import GFATDefaults, InstrumentDefaults, SDCBenchmarkDefaults, SDCDefaults


def test_instrument_defaults_are_frozen():
    with pytest.raises(FrozenInstanceError, match="can't set attribute"):
        DEFAULTS_INSTRUMENT.ccd_gain = 20 * u.electron / u.DN


def test_instrument_mapping_fields_are_immutable_and_copied():
    mesh_transmission = {284: 0.81}
    defaults = InstrumentDefaults(mesh_transmission=mesh_transmission)

    mesh_transmission[284] = 0.5

    assert defaults.mesh_transmission[284] == 0.81
    with pytest.raises(TypeError, match=r"mappingproxy.*does not support item assignment"):
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


def test_instrument_nested_sequences_are_converted_to_tuples():
    main_lines = [["Fe XIX 108.355", "Fe XXI 108.117"]]
    target_vdop = [1, 2, 3]
    defaults = InstrumentDefaults(main_lines_SG=main_lines, target_vdop={"QS": target_vdop})

    main_lines[0][0] = "changed"
    target_vdop[0] = 99

    assert defaults.main_lines_SG == (("Fe XIX 108.355", "Fe XXI 108.117"),)
    assert defaults.target_vdop["QS"] == (1, 2, 3)
    with pytest.raises(TypeError, match=r"tuple.*does not support item assignment"):
        defaults.main_lines_SG[0][0] = "changed"


def test_instrument_defaults_use_evolve_for_overrides():
    updated = attrs.evolve(DEFAULTS_INSTRUMENT, dx_pixel_CI=1 * u.arcmin)

    assert updated.dx_pixel_CI == 60 * u.arcsec
    assert DEFAULTS_INSTRUMENT.dx_pixel_CI == 0.143 * u.arcsec
    assert updated != DEFAULTS_INSTRUMENT


def test_array_defaults_are_read_only():
    gfat_defaults = GFATDefaults()
    benchmark_defaults = SDCBenchmarkDefaults(order=[2, 2, 1])

    with pytest.raises(ValueError, match="read-only"):
        gfat_defaults.sg_channels[0] = 999
    with pytest.raises(ValueError, match="read-only"):
        gfat_defaults.sg_main_lines[0] = "changed"
    with pytest.raises(ValueError, match="read-only"):
        benchmark_defaults.order[0] = 999


def test_instrumental_width_sg_requires_channel_spectral_order():
    with pytest.raises(ValueError, match="requires channel_spectral_order"):
        _ = DEFAULTS_AIA.instrumental_width_sg


def test_data_driven_mask_keyword_defaults_are_copied():
    sdc_defaults = SDCDefaults()
    benchmark_defaults = SDCBenchmarkDefaults()

    assert sdc_defaults.data_driven_mask_keyword == {"fill_value": 1.0, "threshold": 2.24}
    assert benchmark_defaults.data_driven_mask_keyword == sdc_defaults.data_driven_mask_keyword
    assert benchmark_defaults.data_driven_mask_keyword is not sdc_defaults.data_driven_mask_keyword

    sdc_defaults.data_driven_mask_keyword["threshold"] = 99

    assert benchmark_defaults.data_driven_mask_keyword["threshold"] == 2.24
