"""
Tests for CHIANTI-line Gaussian spectral responses.
"""

from functools import partial

import numpy as np
import pytest
import xarray as xr

import astropy.units as u

from muse.instrument.spectral import _create_wavelength_response as _create_wavelength_response_impl
from muse.instrument.spectral import create_spectral_response

RESPONSE_NORMALIZATION = 1e-27
DEFAULT_WAVELENGTH_GRID = np.arange(170.0, 172.002, 0.002) * u.AA
DOPPLER_VELOCITY = np.array([-200.0, 0.0, 200.0]) * u.km / u.s
_create_wavelength_response = partial(_create_wavelength_response_impl, wavelength_grid=DEFAULT_WAVELENGTH_GRID)


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
            "wavelength": ("trans_index", wavelength, {"units": "Angstrom"}),
            "atomic_number": ("trans_index", np.full(n_lines, 26)),
            "gofnt": (("logT", "trans_index"), gofnt, {"units": "erg cm3 / (s sr)"}),
            "full_name": ("trans_index", [f"Fake Fe {i} {value:.3f}" for i, value in enumerate(wavelength)]),
        },
        coords={"logT": logT},
    )


def synthetic_effective_area(
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


def test_public_contract_records_history_and_excludes_unselected_lines():
    line_list = synthetic_line_list(2)
    main_line = line_list.full_name[0].item()

    response = create_spectral_response(
        line_list,
        DEFAULT_WAVELENGTH_GRID,
        main_lines=[main_line],
    )

    assert response.line.values.tolist() == [main_line]
    assert "component_kind" not in response.coords
    assert response.attrs["HISTORY"][0].startswith("create_spectral_response(")
    assert response.attrs["normalization"] == RESPONSE_NORMALIZATION


def test_integral_matches_gofnt():
    ll = synthetic_line_list(1)
    response = _create_wavelength_response(
        ll,
        doppler_velocity=DOPPLER_VELOCITY,
        instrumental_width=0.02 * u.AA,
    )
    dlam = float(response.wavelength_grid[1] - response.wavelength_grid[0])
    integral = (response.spectral_response.sel(doppler_velocity=0).isel(line=0) * dlam).sum("wavelength_bin")
    expected = ll.gofnt.isel(trans_index=0) / RESPONSE_NORMALIZATION
    np.testing.assert_allclose(integral.values, expected.values, rtol=1e-3)
    assert u.Unit(response.spectral_response.attrs["units"]) == u.Unit("1e-27 erg cm3 / (Angstrom s sr)")


def test_peak_follows_doppler_shift():
    ll = synthetic_line_list(1)
    response = _create_wavelength_response(
        ll,
        doppler_velocity=DOPPLER_VELOCITY,
        instrumental_width=0.02 * u.AA,
    )
    speed_of_light_kms = 299792.458
    for velocity in (-200.0, 200.0):
        peak_wavelength = float(
            response.wavelength_grid[
                response.spectral_response.sel(doppler_velocity=velocity).isel(line=0, logT=1).argmax("wavelength_bin")
            ]
        )
        expected = float(ll.wavelength[0]) * (1 + velocity / speed_of_light_kms)
        assert abs(peak_wavelength - expected) < 3e-3


def test_scalar_velocity_inputs():
    ll = synthetic_line_list(1)
    response = _create_wavelength_response(
        ll,
        doppler_velocity=0.0 * u.km / u.s,
        nonthermal_velocity=0.0 * u.km / u.s,
        instrumental_width=0.02 * u.AA,
    )
    assert response.spectral_response.sizes["doppler_velocity"] == 1
    assert response.spectral_response.sizes["nonthermal_velocity"] == 1


def test_multiline_nonthermal_uses_each_line_wavelength():
    line_list = synthetic_line_list(2)
    kwargs = {
        "doppler_velocity": np.array([0.0]) * u.km / u.s,
        "nonthermal_velocity": np.array([0.0, 50.0]) * u.km / u.s,
        "instrumental_width": 0.02 * u.AA,
    }
    combined = _create_wavelength_response(line_list, **kwargs)

    assert "trans_index" not in combined.spectral_response.dims
    for index in range(line_list.sizes["trans_index"]):
        single = _create_wavelength_response(line_list.isel(trans_index=[index]), **kwargs)
        xr.testing.assert_allclose(
            combined.spectral_response.isel(line=index, drop=True),
            single.spectral_response.isel(line=0, drop=True),
        )


def test_missing_line_list_fields_raises():
    line_list = synthetic_line_list(1).drop_vars("atomic_number")
    with pytest.raises(ValueError, match="atomic_number"):
        _create_wavelength_response(line_list)


def test_line_list_units_convert_without_mutating_input():
    canonical = synthetic_line_list(1)
    gofnt_unit = u.erg * u.cm**3 / (u.s * u.sr)
    alternate_gofnt_unit = u.J * u.m**3 / (u.s * u.sr)
    alternate = canonical.assign(
        wavelength=(canonical.wavelength / 10).assign_attrs(units="nm"),
        gofnt=xr.DataArray(
            (canonical.gofnt.data * gofnt_unit).to_value(alternate_gofnt_unit),
            dims=canonical.gofnt.dims,
            coords=canonical.gofnt.coords,
            attrs={"units": str(alternate_gofnt_unit)},
        ),
    )
    before = alternate.copy(deep=True)
    kwargs = {}

    canonical_response = _create_wavelength_response(canonical, **kwargs)
    alternate_response = _create_wavelength_response(alternate, **kwargs)

    xr.testing.assert_allclose(canonical_response, alternate_response)
    xr.testing.assert_identical(alternate, before)


@pytest.mark.parametrize(
    ("name", "units"),
    [
        ("wavelength", None),
        ("wavelength", "km / s"),
        ("gofnt", None),
        ("gofnt", "Angstrom"),
    ],
)
def test_line_list_requires_physical_units(name, units):
    line_list = synthetic_line_list(1)
    if units is None:
        del line_list[name].attrs["units"]
    else:
        line_list[name].attrs["units"] = units

    with pytest.raises(ValueError, match=rf"line_list\.{name}"):
        _create_wavelength_response(line_list)


def test_velocity_axes_convert_to_kilometers_per_second():
    kwargs = {
        "line_list": synthetic_line_list(1),
        "instrumental_width": 0.02 * u.AA,
    }
    response_km = _create_wavelength_response(
        doppler_velocity=np.array([-20.0, 20.0]) * u.km / u.s,
        nonthermal_velocity=np.array([0.0, 5.0]) * u.km / u.s,
        **kwargs,
    )
    response_m = _create_wavelength_response(
        doppler_velocity=np.array([-20000.0, 20000.0]) * u.m / u.s,
        nonthermal_velocity=np.array([0.0, 5000.0]) * u.m / u.s,
        **kwargs,
    )

    xr.testing.assert_allclose(response_km, response_m)
    assert "vdop" not in response_km.dims
    assert response_km.doppler_velocity.attrs["units"] == "km / s"
    assert response_km.nonthermal_velocity.attrs["units"] == "km / s"


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("doppler_velocity", [-20.0, 20.0], TypeError),
        ("doppler_velocity", 0.02 * u.AA, u.UnitsError),
        ("nonthermal_velocity", [5.0], TypeError),
    ],
)
def test_velocity_axes_require_velocity_units(name, value, error):
    with pytest.raises(error, match=name):
        _create_wavelength_response(
            synthetic_line_list(1),
            **{name: value},
        )


def test_wavelength_inputs_convert_to_angstrom():
    kwargs = {"line_list": synthetic_line_list(1), "instrumental_width": 0.02 * u.AA}
    wavelength_grid_nm = DEFAULT_WAVELENGTH_GRID.to(u.nm)

    response_AA = _create_wavelength_response(wavelength_grid=DEFAULT_WAVELENGTH_GRID, **kwargs)
    response_nm = _create_wavelength_response(wavelength_grid=wavelength_grid_nm, **kwargs)

    xr.testing.assert_allclose(response_AA, response_nm)
    assert response_AA.wavelength_grid.attrs["units"] == "Angstrom"
    assert response_AA.wavelength_grid.dims == ("wavelength_bin",)
    assert "wavelength" not in response_AA.coords


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ([170.0, 172.0], TypeError),
        (np.array([170.0, 172.0]) * u.s, u.UnitsError),
        (170.0 * u.AA, ValueError),
        (np.array([170.0, np.nan]) * u.AA, ValueError),
        (np.array([172.0, 170.0]) * u.AA, ValueError),
    ],
)
def test_wavelength_grid_is_validated(value, error):
    with pytest.raises(error, match="wavelength_grid"):
        _create_wavelength_response(synthetic_line_list(1), wavelength_grid=value)


def test_effective_area_units_convert_without_mutating_input():
    wavelength_AA = np.linspace(169.0, 173.0, 20)
    canonical = synthetic_effective_area(
        values=np.full(20, 10.0),
        wavelength=wavelength_AA,
    )
    alternate = synthetic_effective_area(
        values=np.full(20, 0.001),
        wavelength=wavelength_AA / 10,
        area_units="m2",
        wavelength_units="nm",
    )
    before = alternate.copy(deep=True)
    kwargs = {
        "line_list": synthetic_line_list(1),
    }

    canonical_response = _create_wavelength_response(effective_area=canonical, **kwargs)
    alternate_response = _create_wavelength_response(effective_area=alternate, **kwargs)

    xr.testing.assert_allclose(canonical_response, alternate_response)
    xr.testing.assert_identical(alternate, before)


@pytest.mark.parametrize(
    ("area_units", "wavelength_units"),
    [
        (None, "Angstrom"),
        ("s", "Angstrom"),
        ("cm2", None),
        ("cm2", "s"),
    ],
)
def test_effective_area_requires_physical_units(area_units, wavelength_units):
    effective_area = synthetic_effective_area(
        area_units=area_units,
        wavelength_units=wavelength_units,
    )

    with pytest.raises(ValueError, match="effective_area"):
        _create_wavelength_response(
            synthetic_line_list(1),
            effective_area=effective_area,
        )


def test_line_list_must_be_dataset():
    with pytest.raises(TypeError, match=r"xarray\.Dataset"):
        _create_wavelength_response("not-a-dataset")


def test_line_list_must_not_be_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        _create_wavelength_response(synthetic_line_list(0))


def test_line_list_variables_require_transition_dimension():
    line_list = synthetic_line_list(1)
    line_list["wavelength"] = xr.DataArray(
        [170.6],
        dims="other",
        attrs={"units": "Angstrom"},
    )

    with pytest.raises(ValueError, match="trans_index"):
        _create_wavelength_response(line_list)


@pytest.mark.parametrize("name", ["wavelength", "atomic_number", "gofnt", "logT"])
def test_line_list_numeric_values_must_be_finite(name):
    line_list = synthetic_line_list(1)
    if name == "logT":
        line_list = line_list.assign_coords(logT=[5.8, np.nan, 6.2])
    else:
        values = line_list[name].data.astype(float)
        values.flat[0] = np.nan
        line_list[name].data = values

    with pytest.raises(ValueError, match=rf"line_list\.{name}"):
        _create_wavelength_response(line_list)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("wavelength", 0.0),
        ("wavelength", -1.0),
        ("gofnt", -1.0),
    ],
)
def test_line_list_values_must_be_physical(name, value):
    line_list = synthetic_line_list(1)
    values = line_list[name].data.copy()
    values.flat[0] = value
    line_list[name].data = values

    with pytest.raises(ValueError, match=rf"line_list\.{name}"):
        _create_wavelength_response(line_list)


def test_atomic_number_must_be_supported():
    line_list = synthetic_line_list(1).assign(atomic_number=("trans_index", [999]))

    with pytest.raises(ValueError, match=r"line_list\.atomic_number must be between"):
        _create_wavelength_response(line_list)


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("instrumental_width", -0.01 * u.AA, ValueError),
        ("instrumental_width", np.nan * u.AA, ValueError),
        ("doppler_velocity", np.array([0.0, np.nan]) * u.km / u.s, ValueError),
        ("nonthermal_velocity", np.array([-1.0, 0.0]) * u.km / u.s, ValueError),
    ],
)
def test_numeric_controls_are_validated(name, value, error):
    with pytest.raises(error, match=name):
        _create_wavelength_response(synthetic_line_list(1), **{name: value})


@pytest.mark.parametrize("value", [-1.0, np.nan])
def test_effective_area_must_be_finite_and_non_negative(value):
    effective_area = synthetic_effective_area(values=[1.0, value, 1.0])

    with pytest.raises(ValueError, match="effective_area"):
        _create_wavelength_response(synthetic_line_list(1), effective_area=effective_area)


@pytest.mark.parametrize(
    "wavelength",
    [
        [169.0, np.nan, 173.0],
        [169.0, 171.0, 171.0],
        [171.0, 169.0, 173.0],
    ],
)
def test_effective_area_wavelength_must_increase(wavelength):
    effective_area = synthetic_effective_area(wavelength=wavelength)

    with pytest.raises(ValueError, match="effective_area wavelength coordinate"):
        _create_wavelength_response(synthetic_line_list(1), effective_area=effective_area)


def test_effective_area_must_be_one_dimensional():
    effective_area = synthetic_effective_area().expand_dims(order=[1, 2])

    with pytest.raises(ValueError, match="one-dimensional"):
        _create_wavelength_response(synthetic_line_list(1), effective_area=effective_area)


def test_effective_area_is_zero_outside_coverage():
    line_list = synthetic_line_list(wavelength=[171.0])
    main_lines = [line_list.full_name.item()]
    effective_area = synthetic_effective_area(values=[2.0, 2.0], wavelength=[170.5, 171.5])
    kwargs = {"line_list": line_list, "wavelength_grid": DEFAULT_WAVELENGTH_GRID, "main_lines": main_lines}

    unscaled = create_spectral_response(**kwargs)
    scaled = create_spectral_response(**kwargs, effective_area=effective_area)
    inside = (scaled.wavelength_grid >= 170.5) & (scaled.wavelength_grid <= 171.5)
    outside = ~inside

    xr.testing.assert_allclose(
        scaled.spectral_response.where(inside, drop=True),
        2 * unscaled.spectral_response.where(inside, drop=True),
    )
    assert bool((scaled.spectral_response.where(outside, drop=True) == 0).all())
    assert bool(np.isfinite(scaled.spectral_response).all())
    assert bool((scaled.spectral_response >= 0).all())


def test_selection_preserves_order_groups_duplicates_and_hides_contaminants_by_default():
    line_list = synthetic_line_list(3)
    line_list = xr.concat([line_list, line_list.isel(trans_index=[1])], dim="trans_index")
    main_lines = [line_list.full_name[1].item(), line_list.full_name[0].item()]
    kwargs = {
        "instrumental_width": 0.02 * u.AA,
    }

    response = _create_wavelength_response(line_list, main_lines=main_lines, **kwargs)
    reordered = _create_wavelength_response(
        line_list.isel(trans_index=[3, 2, 0, 1]),
        main_lines=main_lines,
        **kwargs,
    )
    with_contaminants = _create_wavelength_response(
        line_list,
        main_lines=main_lines,
        include_contaminants=True,
        **kwargs,
    )
    all_summed = _create_wavelength_response(
        line_list,
        main_lines=[],
        include_contaminants=True,
        **kwargs,
    )
    single = _create_wavelength_response(line_list.isel(trans_index=[1]), **kwargs)

    assert list(response.line.values) == main_lines
    assert list(response.component_kind.values) == ["line", "line"]
    xr.testing.assert_allclose(response.spectral_response, reordered.spectral_response)
    xr.testing.assert_allclose(
        response.spectral_response.sel(line=main_lines[0]),
        2 * single.spectral_response.sel(line=main_lines[0]),
    )
    assert list(with_contaminants.line.values) == [*main_lines, "contaminants"]
    assert list(with_contaminants.component_kind.values) == ["line", "line", "contaminants"]
    assert np.isnan(with_contaminants.line_wavelength.sel(line="contaminants"))
    assert with_contaminants.line_wavelength.attrs["units"] == "Angstrom"
    xr.testing.assert_allclose(
        with_contaminants.spectral_response.sum("line"),
        all_summed.spectral_response.isel(line=0, drop=True),
    )


def test_empty_main_lines_require_contaminant_opt_in():
    with pytest.raises(ValueError, match="include_contaminants=True"):
        _create_wavelength_response(
            synthetic_line_list(2),
            main_lines=[],
        )


def test_instrumental_width_converts_to_angstrom():
    kwargs = {
        "line_list": synthetic_line_list(1),
    }

    response_AA = _create_wavelength_response(instrumental_width=0.02 * u.AA, **kwargs)
    response_nm = _create_wavelength_response(instrumental_width=0.002 * u.nm, **kwargs)

    xr.testing.assert_allclose(response_AA, response_nm)


@pytest.mark.parametrize(
    ("instrumental_width", "error"),
    [
        (0.02, TypeError),
        (5 * u.km / u.s, u.UnitsError),
        (np.array([0.01, 0.02]) * u.AA, ValueError),
    ],
)
def test_instrumental_width_requires_wavelength_units(instrumental_width, error):
    with pytest.raises(error, match="instrumental_width"):
        _create_wavelength_response(
            synthetic_line_list(1),
            instrumental_width=instrumental_width,
        )


@pytest.mark.parametrize(
    ("main_lines", "error", "match"),
    [
        ("not-a-sequence", TypeError, "sequence"),
        (["Fake Fe 0 170.600", "Fake Fe 0 170.600"], ValueError, "unique"),
        (["missing"], ValueError, "not found"),
    ],
)
def test_main_lines_validation(main_lines, error, match):
    with pytest.raises(error, match=match):
        _create_wavelength_response(
            synthetic_line_list(2),
            main_lines=main_lines,
        )
