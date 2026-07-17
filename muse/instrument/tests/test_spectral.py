"""
Tests for CHIANTI-line Gaussian spectral responses.
"""

import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.instrument.spectral import create_response_function

NORM = 1e-27
ORDERS = xr.DataArray([1, 2], dims="order", coords={"order": [1, 2]})
VDOP = np.array([-200.0, 0.0, 200.0]) * u.km / u.s
WIDE_BAND_LINES = {"wavelength": np.linspace(105.0, 195.0, 30), "logT": np.linspace(4.6, 7.6, 15)}


def synthetic_line_list(n_lines=2, wavelength=None, logT=None):
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
            "wavelength": ("trans_index", wavelength),
            "atomic_number": ("trans_index", np.full(n_lines, 26)),
            "gofnt": (("logT", "trans_index"), gofnt),
            "full_name": ("trans_index", [f"Fake Fe {i} {value:.3f}" for i, value in enumerate(wavelength)]),
        },
        coords={"logT": logT},
    )


class TestCreateResponseFunctionScalar:
    def test_integral_matches_gofnt(self):
        ll = synthetic_line_list(1)
        response = create_response_function(
            ll,
            vdop=VDOP,
            instrumental_width=0.02,
            wavelength_range=[170.0, 172.0],
            wavelength_step_mA=2.0,
            num_lines_keep=1,
        )
        dlam = float(response.wavelength[1] - response.wavelength[0])
        integral = (response.response.sel(vdop=0).isel(line=0) * dlam).sum("wavelength")
        expected = ll.gofnt.isel(trans_index=0) / NORM
        np.testing.assert_allclose(integral.values, expected.values, rtol=1e-3)

    def test_peak_follows_doppler_shift(self):
        ll = synthetic_line_list(1)
        response = create_response_function(
            ll,
            vdop=VDOP,
            instrumental_width=0.02,
            wavelength_range=[170.0, 172.0],
            wavelength_step_mA=2.0,
            num_lines_keep=1,
        )
        speed_of_light_kms = 299792.458
        for velocity in (-200.0, 200.0):
            peak_wavelength = float(
                response.wavelength[response.response.sel(vdop=velocity).isel(line=0, logT=1).argmax("wavelength")]
            )
            expected = float(ll.wavelength[0]) * (1 + velocity / speed_of_light_kms)
            assert abs(peak_wavelength - expected) < 3e-3

    def test_scalar_velocity_inputs(self):
        ll = synthetic_line_list(1)
        response = create_response_function(
            ll,
            vdop=0.0 * u.km / u.s,
            nonthermal_velocity=0.0,
            instrumental_width=0.02,
            wavelength_range=[170.0, 172.0],
            wavelength_step_mA=2.0,
            num_lines_keep=1,
        )
        assert response.response.sizes["vdop"] == 1
        assert response.response.sizes["nonthermal_velocity"] == 1

    def test_missing_line_list_fields_raises(self):
        line_list = synthetic_line_list(1).drop_vars("atomic_number")
        with pytest.raises(ValueError, match="atomic_number"):
            create_response_function(line_list, wavelength_range=[170.0, 172.0], num_lines_keep=1)


class TestCreateResponseFunctionOrderDims:
    def test_instrumental_width_order_dim(self):
        line_list = synthetic_line_list(2)
        width = xr.DataArray([0.01, 0.05], dims="order", coords={"order": [1, 2]})
        response = create_response_function(
            line_list,
            vdop=VDOP,
            instrumental_width=width,
            wavelength_range=[170.0, 172.0],
            wavelength_step_mA=2.0,
            num_lines_keep=2,
        )
        assert response.response.sizes["order"] == 2
        assert not np.allclose(response.response.isel(order=0), response.response.isel(order=1))

    def test_equal_widths_give_identical_slices(self):
        line_list = synthetic_line_list(1)
        width = xr.DataArray([0.02, 0.02], dims="order", coords={"order": [1, 2]})
        response = create_response_function(
            line_list,
            vdop=VDOP,
            instrumental_width=width,
            wavelength_range=[170.0, 172.0],
            wavelength_step_mA=2.0,
            num_lines_keep=1,
        )
        np.testing.assert_array_equal(
            response.response.isel(order=0).values,
            response.response.isel(order=1).values,
        )

    def test_wavelength_range_order_dependent(self):
        line_list = synthetic_line_list(2)
        wavelength_range = [171.0 * 2 / ORDERS - 1.0, 171.0 * 2 / ORDERS + 1.0]
        response = create_response_function(
            line_list,
            vdop=VDOP,
            instrumental_width=0.02,
            wavelength_range=wavelength_range,
            num_wavelength_bins=64,
            num_lines_keep=2,
        )
        assert set(response.wavelength_grid.dims) == {"wavelength", "order"}
        assert response.response.sizes["wavelength"] == 64

    def test_wavelength_range_order_dependent_requires_num_bins(self):
        line_list = synthetic_line_list(1)
        wavelength_range = [171.0 * 2 / ORDERS - 1.0, 171.0 * 2 / ORDERS + 1.0]
        with pytest.raises(ValueError, match="num_wavelength_bins"):
            create_response_function(
                line_list,
                vdop=VDOP,
                instrumental_width=0.02,
                wavelength_range=wavelength_range,
                num_lines_keep=1,
            )

    def test_window_requires_one_dimensional_wavelength_grid(self):
        line_list = synthetic_line_list(3)
        wavelength_range = [171.0 * 2 / ORDERS - 1.0, 171.0 * 2 / ORDERS + 1.0]
        with pytest.raises(ValueError, match="one-dimensional"):
            create_response_function(
                line_list,
                wavelength_range=wavelength_range,
                num_wavelength_bins=64,
                num_lines_keep=0,
                window_sigma=8.0,
            )

    def test_contaminant_sum_with_order_dim(self):
        line_list = synthetic_line_list(3)
        width = xr.DataArray([0.01, 0.05], dims="order", coords={"order": [1, 2]})
        response = create_response_function(
            line_list,
            vdop=VDOP,
            instrumental_width=width,
            wavelength_range=[170.0, 172.0],
            wavelength_step_mA=2.0,
            num_lines_keep=0,
        )
        assert response.response.sizes["order"] == 2
        assert response.response.sizes["line"] == 1

    def test_effective_area_band_order(self):
        line_list = synthetic_line_list(2)
        wavelength = np.linspace(168.0, 174.0, 40)
        effective_area = xr.DataArray(
            np.full((40, 1), 10.0),
            dims=("wavelength", "band"),
            coords={"wavelength": wavelength, "band": [171]},
        )
        effective_area = effective_area * xr.DataArray([1.0, 0.5], dims="order", coords={"order": [1, 2]})
        response = create_response_function(
            line_list,
            vdop=VDOP,
            instrumental_width=0.02,
            wavelength_range=[170.0, 172.0],
            wavelength_step_mA=2.0,
            num_lines_keep=2,
            effective_area=effective_area,
        )
        assert {"band", "order", "line", "logT", "vdop", "wavelength"} <= set(response.response.dims)
        np.testing.assert_allclose(
            response.response.sel(order=2).values,
            0.5 * response.response.sel(order=1).values,
            rtol=1e-12,
        )


class TestWindowedContaminants:
    def test_window_matches_full_grid_all_contaminants(self):
        line_list = synthetic_line_list(**WIDE_BAND_LINES)
        kwargs = {
            "vdop": np.arange(-200.0, 210.0, 40.0),
            "instrumental_width": 0.0,
            "wavelength_range": [100.0, 200.0],
            "num_lines_keep": 0,
        }
        full = create_response_function(line_list, **kwargs)
        windowed = create_response_function(line_list, window_sigma=8.0, **kwargs)
        peak = float(np.abs(full["response"]).max())
        xr.testing.assert_allclose(full["response"], windowed["response"], atol=peak * 1e-9)

    def test_window_with_kept_lines(self):
        line_list = synthetic_line_list(**WIDE_BAND_LINES)
        kwargs = {
            "vdop": np.arange(-100.0, 110.0, 50.0),
            "instrumental_width": 0.01,
            "wavelength_range": [100.0, 200.0],
            "num_lines_keep": 2,
        }
        full = create_response_function(line_list, **kwargs)
        windowed = create_response_function(line_list, window_sigma=8.0, **kwargs)
        peak = float(np.abs(full["response"]).max())
        xr.testing.assert_allclose(full["response"], windowed["response"], atol=peak * 1e-9)

    def test_window_off_grid_contaminants_are_zero(self):
        line_list = synthetic_line_list(1, wavelength=[500.0])
        response = create_response_function(
            line_list,
            wavelength_range=[100.0, 101.0],
            num_lines_keep=0,
            window_sigma=8.0,
        )
        assert response.response.dtype.kind != "O"
        assert not response.response.any()
