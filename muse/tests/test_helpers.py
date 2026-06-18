import numpy as np
import pytest

from muse.tests.helpers import assert_dataset_structure


def test_fake_response_matches_response_zarr_contract(response) -> None:
    assert response.SG_resp.dims == ("line", "vdop", "logT", "slit", "SG_xpixel")
    assert response.SG_resp.sizes == {"line": 7, "vdop": 9, "logT": 7, "slit": 35, "SG_xpixel": 32}
    np.testing.assert_array_equal(response.channel.values, [108, 108, 108, 171, 171, 284, 284])
    np.testing.assert_array_equal(
        response.line.values,
        [
            "Fe XIX 108.355",
            "Fe XXI 108.117",
            "108 remaining 10000 lines",
            "Fe IX 171.073",
            "171 remaining 10000 lines",
            "Fe XV 284.163",
            "284 remaining 10000 lines",
        ],
    )
    assert response.SG_wvl.dims == ("line", "slit", "SG_xpixel")
    assert response.SG_wvl.attrs["units"] == "Angstrom"
    assert response.SG_resp.attrs["units"] == "1e-27 ph cm5 / s"


def test_fake_vdem_has_expected_axes(vdem) -> None:
    assert_dataset_structure(
        vdem,
        data_vars=("vdem",),
        coords=("logT", "vdop", "x", "y"),
        sizes={"logT": 7, "vdop": 9, "y": 32, "x": 385},
        finite_vars=("vdem",),
    )
    assert vdem.x.attrs["units"] == "arcsec"


def test_assert_dataset_structure_rejects_wrong_data_vars(vdem) -> None:
    with pytest.raises(AssertionError, match="data vars differ"):
        assert_dataset_structure(vdem, data_vars=("flux",), coords=("x",))
