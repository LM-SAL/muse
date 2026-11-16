"""
Tests for CHIANTI-line Gaussian spectral responses.
"""

import numpy as np
import pytest
import xarray as xr

import astropy.constants as const
import astropy.units as u
from astropy.stats import gaussian_sigma_to_fwhm

import muse.instrument.response.spectral as spectral_module
from muse.instrument import create_effective_area
from muse.instrument.response.spectral import _build_spectral_response as _build_spectral_response_impl
from muse.instrument.response.spectral import create_log_wavelength_edges, create_spectral_response
from muse.tests.helpers import fake_effective_area, fake_line_list

pytest.importorskip("numexpr")
pytest.importorskip("periodictable")

RESPONSE_NORMALIZATION = 1e-27
DEFAULT_WAVELENGTH_GRID = np.arange(170.0, 172.002, 0.002) * u.AA
DOPPLER_VELOCITY = np.array([-200.0, 0.0, 200.0]) * u.km / u.s


def _build_spectral_response(line_list, wavelength_grid=DEFAULT_WAVELENGTH_GRID, **kwargs):
    """
    The builder on the default grid, with every line as a main line unless the test
    names them.
    """
    names = line_list.full_name.values if isinstance(line_list, xr.Dataset) else ()
    kwargs.setdefault("main_lines", [str(name) for name in names])
    return _build_spectral_response_impl(line_list, wavelength_grid, **kwargs)


def test_public_contract_records_history_and_excludes_unselected_lines():
    line_list = fake_line_list(2).assign_attrs(
        history=["create_chianti_line_list()"],
        chianti_version="10.1",
        normalization=2e-27,
    )
    main_line = line_list.full_name[0].item()

    response = create_spectral_response(
        line_list,
        DEFAULT_WAVELENGTH_GRID,
        main_lines=[main_line],
    )

    assert response.line.values.tolist() == [main_line]
    assert response.attrs["history"][0] == "create_chianti_line_list()"
    assert response.attrs["history"][1].startswith("create_spectral_response(")
    assert response.attrs["chianti_version"] == "10.1"
    assert response.attrs["normalization"] == RESPONSE_NORMALIZATION
    assert u.Unit(response.spectral_response.attrs["units"]) == u.Unit("1e-27 erg cm3 / (Angstrom s sr)")


def test_off_grid_contaminants_are_skipped(monkeypatch):
    calls = []
    evaluate = spectral_module._evaluate_gaussian_response
    monkeypatch.setattr(
        spectral_module,
        "_evaluate_gaussian_response",
        lambda *args, **kwargs: (calls.append(1), evaluate(*args, **kwargs))[1],
    )
    ll = fake_line_list(wavelength=[171.0, 171.5, 200.0, 300.0])
    main = str(ll.full_name.values[0])

    response = _build_spectral_response(ll, main_lines=[main], include_contaminants=True)

    assert len(calls) == 2  # main line + the one contaminant whose window reaches the grid
    assert response.line.values.tolist() == [main, "contaminants"]

    calls.clear()
    response = _build_spectral_response(ll.isel(transition=[0, 2, 3]), main_lines=[main], include_contaminants=True)

    assert len(calls) == 2  # one off-grid line is kept so the zero component still exists
    assert response.line.values.tolist() == [main, "contaminants"]
    assert float(response.spectral_response.sel(line="contaminants").max()) == 0.0


def _sigma_of_an_iron_line_at_1_MK(**kwargs) -> float:
    """
    The measured sigma, in Angstrom, of one Fe line at 171 Angstrom and 1 MK.
    """
    line_list = fake_line_list(1, wavelength=[171.0], logT=[6.0])
    # sampled at the bin centers: a bin average would add the width of a bin to the line's
    response = create_spectral_response(
        line_list,
        DEFAULT_WAVELENGTH_GRID,
        main_lines=line_list.full_name.values.tolist(),
        bin_method="center",
        **kwargs,
    )
    profile = response.spectral_response.squeeze().values
    offset = response.wavelength_grid.values - 171.0
    return float(np.sqrt(np.sum(profile * offset**2) / np.sum(profile)))


def test_thermal_width_takes_the_element_mass_in_atomic_mass_units():
    # Fe at 1 MK: sigma = wavelength * sqrt(k T / (55.845 u)) / c. The proton mass in place of u makes
    # the line 0.36 % too narrow.
    thermal_velocity = np.sqrt(const.k_B * 1e6 * u.K / (55.845 * const.u)).to_value(u.m / u.s)
    np.testing.assert_allclose(_sigma_of_an_iron_line_at_1_MK(), 171.0 * thermal_velocity / const.c.value, rtol=1e-4)


def test_instrumental_fwhm_adds_its_sigma_in_quadrature():
    # 0.05 Angstrom FWHM is a sigma of 0.05 / (2 sqrt(2 ln 2)), added in quadrature to the thermal sigma.
    thermal = _sigma_of_an_iron_line_at_1_MK()
    broadened = _sigma_of_an_iron_line_at_1_MK(instrumental_fwhm=0.05 * u.AA)
    np.testing.assert_allclose(broadened**2 - thermal**2, (0.05 / gaussian_sigma_to_fwhm) ** 2, rtol=1e-3)


def test_nonthermal_velocity_is_the_1e_velocity():
    # xi is the 1/e velocity of the EIS/IRIS literature: it adds (lambda xi / c)^2 / 2 to sigma^2.
    thermal = _sigma_of_an_iron_line_at_1_MK()
    broadened = _sigma_of_an_iron_line_at_1_MK(nonthermal_velocity=20 * u.km / u.s)
    np.testing.assert_allclose(broadened**2 - thermal**2, (171.0 * 20 / 299792.458) ** 2 / 2, rtol=1e-3)


def test_peak_follows_doppler_shift():
    ll = fake_line_list(1)
    response = _build_spectral_response(
        ll,
        doppler_velocity=DOPPLER_VELOCITY,
        instrumental_fwhm=0.02 * u.AA,
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
    ll = fake_line_list(1)
    response = _build_spectral_response(
        ll,
        doppler_velocity=0.0 * u.km / u.s,
        nonthermal_velocity=0.0 * u.km / u.s,
        instrumental_fwhm=0.02 * u.AA,
    )
    assert response.spectral_response.sizes["doppler_velocity"] == 1
    assert response.spectral_response.sizes["nonthermal_velocity"] == 1


def test_multiline_nonthermal_uses_each_line_wavelength():
    line_list = fake_line_list(2)
    kwargs = {
        "doppler_velocity": np.array([0.0]) * u.km / u.s,
        "nonthermal_velocity": np.array([0.0, 50.0]) * u.km / u.s,
        "instrumental_fwhm": 0.02 * u.AA,
    }
    combined = _build_spectral_response(line_list, **kwargs)

    assert "transition" not in combined.spectral_response.dims
    for index in range(line_list.sizes["transition"]):
        single = _build_spectral_response(line_list.isel(transition=[index]), **kwargs)
        xr.testing.assert_allclose(
            combined.spectral_response.isel(line=index, drop=True),
            single.spectral_response.isel(line=0, drop=True),
        )


def test_missing_line_list_fields_raises():
    line_list = fake_line_list(1).drop_vars("atomic_number")
    with pytest.raises(ValueError, match="atomic_number"):
        _build_spectral_response(line_list)


def test_line_list_units_convert_without_mutating_input():
    canonical = fake_line_list(1)
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

    canonical_response = _build_spectral_response(canonical, **kwargs)
    alternate_response = _build_spectral_response(alternate, **kwargs)

    xr.testing.assert_allclose(canonical_response, alternate_response)
    xr.testing.assert_identical(alternate, before)


@pytest.mark.parametrize(("name", "units"), [("wavelength", "km / s"), ("gofnt", "Angstrom"), ("logT", "Angstrom")])
def test_line_list_requires_physical_units(name, units):
    line_list = fake_line_list(1)
    line_list[name].attrs["units"] = units

    with pytest.raises(ValueError, match=rf"line_list\.{name}"):
        _build_spectral_response(line_list)


def test_velocity_axes_convert_to_kilometers_per_second():
    kwargs = {
        "line_list": fake_line_list(1),
        "instrumental_fwhm": 0.02 * u.AA,
    }
    response_km = _build_spectral_response(
        doppler_velocity=np.array([-20.0, 20.0]) * u.km / u.s,
        nonthermal_velocity=np.array([0.0, 5.0]) * u.km / u.s,
        **kwargs,
    )
    response_m = _build_spectral_response(
        doppler_velocity=np.array([-20000.0, 20000.0]) * u.m / u.s,
        nonthermal_velocity=np.array([0.0, 5000.0]) * u.m / u.s,
        **kwargs,
    )

    xr.testing.assert_allclose(response_km, response_m)
    assert response_km.doppler_velocity.attrs["units"] == "km / s"
    assert response_km.nonthermal_velocity.attrs["units"] == "km / s"


def test_wavelength_inputs_convert_to_angstrom():
    kwargs = {"line_list": fake_line_list(1), "instrumental_fwhm": 0.02 * u.AA}
    wavelength_grid_nm = DEFAULT_WAVELENGTH_GRID.to(u.nm)

    response_AA = _build_spectral_response(wavelength_grid=DEFAULT_WAVELENGTH_GRID, **kwargs)
    response_nm = _build_spectral_response(wavelength_grid=wavelength_grid_nm, **kwargs)

    xr.testing.assert_allclose(response_AA, response_nm)
    assert response_AA.wavelength_grid.attrs["units"] == "Angstrom"
    assert response_AA.wavelength_grid.dims == ("wavelength_bin",)
    assert "wavelength" not in response_AA.coords


@pytest.mark.parametrize("value", [170.0 * u.AA, np.array([170.0, np.nan]) * u.AA])
def test_wavelength_grid_is_validated(value):
    with pytest.raises(ValueError, match="wavelength_grid"):
        _build_spectral_response(fake_line_list(1), wavelength_grid=value)


def test_effective_area_units_convert_without_mutating_input():
    wavelength_AA = np.linspace(169.0, 173.0, 20)
    canonical = fake_effective_area(
        values=np.full(20, 10.0),
        wavelength=wavelength_AA,
    )
    alternate = fake_effective_area(
        values=np.full(20, 0.001),
        wavelength=wavelength_AA / 10,
        area_units="m2",
        wavelength_units="nm",
    )
    before = alternate.copy(deep=True)
    kwargs = {
        "line_list": fake_line_list(1),
    }

    canonical_response = _build_spectral_response(effective_area=canonical, **kwargs)
    alternate_response = _build_spectral_response(effective_area=alternate, **kwargs)

    xr.testing.assert_allclose(canonical_response, alternate_response)
    xr.testing.assert_identical(alternate, before)


def test_scalar_effective_area_matches_flat_curve():
    # A 0-d DataArray (e.g. DEFAULTS_MUSE.main_line_effective_area_SG.sel(channel=...))
    # must scale the response exactly like a flat curve.
    line_list = fake_line_list(1)
    flat = fake_effective_area(values=np.full(3, 4.3))

    from_curve = _build_spectral_response(line_list, effective_area=flat)
    from_selection = _build_spectral_response(line_list, effective_area=xr.DataArray(4.3 * u.cm**2))
    # Plain data with units declared in attrs (the units-are-attrs convention), incl. conversion.
    from_attrs_m2 = _build_spectral_response(line_list, effective_area=xr.DataArray(4.3e-4, attrs={"units": "m2"}))

    xr.testing.assert_allclose(from_selection, from_curve)
    xr.testing.assert_allclose(from_attrs_m2, from_curve)


def test_an_effective_area_dataset_counts_as_its_curve():
    line_list = fake_line_list(1)
    curve = fake_effective_area([1.0, 2.0, 3.0], [169.0, 171.0, 173.0])
    dataset = create_effective_area(curve.values * u.cm**2, curve.wavelength.values * u.AA, channel=171 * u.AA)

    from_dataset = _build_spectral_response(line_list, effective_area=dataset)

    # identical, coordinates included: the Dataset's scalar channel does not come along
    xr.testing.assert_identical(from_dataset, _build_spectral_response(line_list, effective_area=curve))


def test_scalar_effective_area_is_validated():
    line_list = fake_line_list(1)
    with pytest.raises(TypeError, match=r"effective_area must be an xarray\.DataArray"):
        _build_spectral_response(line_list, effective_area=4.3 * u.cm**2)
    with pytest.raises(ValueError, match="effective_area must be convertible to cm2"):
        _build_spectral_response(line_list, effective_area=xr.DataArray(3.0 * u.s))
    with pytest.raises(ValueError, match="finite, non-negative"):
        _build_spectral_response(line_list, effective_area=xr.DataArray(-1.0 * u.cm**2))
    with pytest.raises(ValueError, match="valid astropy unit"):
        _build_spectral_response(line_list, effective_area=xr.DataArray(4.3, attrs={"units": "bogus"}))
    with pytest.raises(ValueError, match="effective_area must be convertible to cm2"):
        _build_spectral_response(line_list, effective_area=xr.DataArray(4.3))  # no units anywhere
    with pytest.raises(ValueError, match="contradict the data unit"):
        # Both the payload and the attrs declare a unit and they disagree; neither can be
        # trusted, so this must not be silently resolved in favor of one of them.
        _build_spectral_response(line_list, effective_area=xr.DataArray(4.3 * u.cm**2, attrs={"units": "s"}))


@pytest.mark.parametrize(("area_units", "wavelength_units"), [("s", "Angstrom"), ("cm2", "s")])
def test_effective_area_requires_physical_units(area_units, wavelength_units):
    effective_area = fake_effective_area(
        area_units=area_units,
        wavelength_units=wavelength_units,
    )

    with pytest.raises(ValueError, match="effective_area"):
        _build_spectral_response(
            fake_line_list(1),
            effective_area=effective_area,
        )


def test_line_list_must_be_dataset():
    with pytest.raises(TypeError, match=r"xarray\.Dataset"):
        _build_spectral_response("not-a-dataset")


def test_a_line_list_must_lie_along_transition():
    with pytest.raises(ValueError, match=r"lies along trans_index, as muse 0\.3 wrote line lists"):
        _build_spectral_response(fake_line_list(2).rename({"transition": "trans_index"}))
    with pytest.raises(ValueError, match="must lie along a transition dimension"):
        _build_spectral_response_impl(fake_line_list(1).squeeze("transition"), DEFAULT_WAVELENGTH_GRID, main_lines=[])


def test_line_list_must_not_be_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        _build_spectral_response(fake_line_list(0))


def test_line_list_variables_require_transition_dimension():
    line_list = fake_line_list(1)
    line_list["wavelength"] = xr.DataArray(
        [170.6],
        dims="other",
        attrs={"units": "Angstrom"},
    )

    with pytest.raises(ValueError, match="transition"):
        _build_spectral_response(line_list)


@pytest.mark.parametrize("name", ["wavelength", "gofnt", "logT"])
def test_line_list_numeric_values_must_be_finite(name):
    line_list = fake_line_list(1)
    if name == "logT":
        line_list = line_list.assign_coords(logT=[5.8, np.nan, 6.2])
    else:
        values = line_list[name].data.astype(float)
        values.flat[0] = np.nan
        line_list[name].data = values

    with pytest.raises(ValueError, match=rf"line_list\.{name} must contain finite"):
        _build_spectral_response(line_list)


@pytest.mark.parametrize(("name", "value"), [("wavelength", 0.0), ("gofnt", -1.0)])
def test_line_list_values_must_be_physical(name, value):
    line_list = fake_line_list(1)
    values = line_list[name].data.copy()
    values.flat[0] = value
    line_list[name].data = values

    with pytest.raises(ValueError, match=rf"line_list\.{name}"):
        _build_spectral_response(line_list)


def test_atomic_number_must_be_supported():
    line_list = fake_line_list(1).assign(atomic_number=("transition", [999]))

    with pytest.raises(ValueError, match=r"line_list\.atomic_number must be between"):
        _build_spectral_response(line_list)


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("instrumental_fwhm", -0.01 * u.AA, ValueError),
        ("instrumental_fwhm", np.nan * u.AA, ValueError),
        ("doppler_velocity", np.array([0.0, np.nan]) * u.km / u.s, ValueError),
        ("nonthermal_velocity", np.array([-1.0, 0.0]) * u.km / u.s, ValueError),
    ],
)
def test_numeric_controls_are_validated(name, value, error):
    with pytest.raises(error, match=name):
        _build_spectral_response(fake_line_list(1), **{name: value})


@pytest.mark.parametrize("value", [-1.0, np.nan])
def test_effective_area_must_be_finite_and_non_negative(value):
    effective_area = fake_effective_area(values=[1.0, value, 1.0])

    with pytest.raises(ValueError, match="effective_area"):
        _build_spectral_response(fake_line_list(1), effective_area=effective_area)


@pytest.mark.parametrize("wavelength", [[169.0, np.nan, 173.0], [169.0, 171.0, 171.0]])
def test_effective_area_wavelength_must_increase(wavelength):
    effective_area = fake_effective_area(wavelength=wavelength)

    with pytest.raises(ValueError, match="effective_area wavelength coordinate"):
        _build_spectral_response(fake_line_list(1), effective_area=effective_area)


def test_effective_area_must_be_one_dimensional():
    effective_area = fake_effective_area().expand_dims(order=[1, 2])

    with pytest.raises(ValueError, match="one-dimensional"):
        _build_spectral_response(fake_line_list(1), effective_area=effective_area)


def test_effective_area_is_zero_outside_coverage():
    line_list = fake_line_list(wavelength=[171.0])
    main_lines = [line_list.full_name.item()]
    effective_area = fake_effective_area(values=[2.0, 2.0], wavelength=[170.5, 171.5])
    # at the bin centers the curve is resampled onto the grid, which lies partly beyond the curve
    kwargs = {
        "line_list": line_list,
        "wavelength_grid": DEFAULT_WAVELENGTH_GRID,
        "main_lines": main_lines,
        "bin_method": "center",
    }

    unscaled = create_spectral_response(**kwargs)
    scaled = create_spectral_response(**kwargs, effective_area=effective_area)
    inside = (scaled.wavelength_grid >= 170.5) & (scaled.wavelength_grid <= 171.5)
    outside = ~inside

    xr.testing.assert_allclose(
        scaled.spectral_response.where(inside, drop=True),
        2 * unscaled.spectral_response.where(inside, drop=True),
    )
    assert bool((scaled.spectral_response.where(outside, drop=True) == 0).all())


def test_selection_preserves_order_groups_duplicates_and_hides_contaminants_by_default():
    line_list = fake_line_list(3)
    line_list = xr.concat([line_list, line_list.isel(transition=[1])], dim="transition")
    main_lines = [line_list.full_name[1].item(), line_list.full_name[0].item()]
    kwargs = {
        "instrumental_fwhm": 0.02 * u.AA,
    }

    response = _build_spectral_response(line_list, main_lines=main_lines, **kwargs)
    reordered = _build_spectral_response(
        line_list.isel(transition=[3, 2, 0, 1]),
        main_lines=main_lines,
        **kwargs,
    )
    with_contaminants = _build_spectral_response(
        line_list,
        main_lines=main_lines,
        include_contaminants=True,
        **kwargs,
    )
    all_summed = _build_spectral_response(
        line_list,
        main_lines=[],
        include_contaminants=True,
        **kwargs,
    )
    single = _build_spectral_response(line_list.isel(transition=[1]), **kwargs)

    assert list(response.line.values) == main_lines
    xr.testing.assert_allclose(response.spectral_response, reordered.spectral_response)
    xr.testing.assert_allclose(
        response.spectral_response.sel(line=main_lines[0]),
        2 * single.spectral_response.sel(line=main_lines[0]),
    )
    assert list(with_contaminants.line.values) == [*main_lines, "contaminants"]
    assert np.isnan(with_contaminants.line_wavelength.sel(line="contaminants"))
    assert with_contaminants.line_wavelength.attrs["units"] == "Angstrom"
    xr.testing.assert_allclose(
        with_contaminants.spectral_response.sum("line"),
        all_summed.spectral_response.isel(line=0, drop=True),
    )


def test_no_main_lines_require_contaminant_opt_in():
    with pytest.raises(ValueError, match="include_contaminants=True"):
        create_spectral_response(fake_line_list(2), DEFAULT_WAVELENGTH_GRID, main_lines=None)


def test_instrumental_fwhm_converts_to_angstrom():
    kwargs = {
        "line_list": fake_line_list(1),
    }

    response_AA = _build_spectral_response(instrumental_fwhm=0.02 * u.AA, **kwargs)
    response_nm = _build_spectral_response(instrumental_fwhm=0.002 * u.nm, **kwargs)

    xr.testing.assert_allclose(response_AA, response_nm)


def test_instrumental_fwhm_must_be_scalar():
    with pytest.raises(ValueError, match="instrumental_fwhm must be scalar"):
        _build_spectral_response(fake_line_list(1), instrumental_fwhm=[0.01, 0.02] * u.AA)


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
        _build_spectral_response(
            fake_line_list(2),
            main_lines=main_lines,
        )


def _bin_width(response):
    return response.wavelength_bounds.isel(bound=1) - response.wavelength_bounds.isel(bound=0)


def _integral_ratio(response, line_list, index=0):
    integral = (response.spectral_response * _bin_width(response)).sum("wavelength_bin")
    return (integral / (line_list.gofnt.isel(transition=index) / RESPONSE_NORMALIZATION)).values


def test_average_conserves_the_area_of_an_unresolved_line():
    # The thermal width of the line is well below the 20 mA bins: the midpoint rule miscounts
    # it by up to about 2 exp(-2 pi^2 sigma^2 / h^2); the bin average is exact.
    line_list = fake_line_list(1)
    grid = np.arange(170.0, 172.02, 0.02) * u.AA
    kwargs = {"main_lines": [line_list.full_name[0].item()], "doppler_velocity": DOPPLER_VELOCITY}
    averaged = create_spectral_response(line_list, grid, **kwargs)
    centered = create_spectral_response(line_list, grid, bin_method="center", **kwargs)
    np.testing.assert_allclose(_integral_ratio(averaged, line_list), 1.0, rtol=1e-12)
    assert np.abs(_integral_ratio(centered, line_list) - 1.0).max() > 0.01
    assert averaged.spectral_response.attrs["bin_method"] == "average"
    assert centered.spectral_response.attrs["bin_method"] == "center"


def test_average_matches_center_when_the_line_is_resolved():
    # sigma / h ~ 25: the midpoint total is exact and the per-bin values differ only
    # by the curvature term (h / sigma)^2 / 24.
    line_list = fake_line_list(2)
    kwargs = {"main_lines": list(line_list.full_name.values), "instrumental_fwhm": 0.05 * u.AA * gaussian_sigma_to_fwhm}
    averaged = create_spectral_response(line_list, DEFAULT_WAVELENGTH_GRID, bin_method="average", **kwargs)
    centered = create_spectral_response(line_list, DEFAULT_WAVELENGTH_GRID, bin_method="center", **kwargs)
    assert averaged.spectral_response.dims == centered.spectral_response.dims
    peak = float(averaged.spectral_response.max())
    xr.testing.assert_allclose(
        averaged.spectral_response, centered.spectral_response, atol=peak * (0.002 / 0.05) ** 2 / 24
    )


def test_wavelength_edges_define_the_bins():
    line_list = fake_line_list(1)
    edges = create_log_wavelength_edges([100, 300] * u.AA, 0.01)
    response = create_spectral_response(
        line_list,
        main_lines=[line_list.full_name[0].item()],
        doppler_velocity=DOPPLER_VELOCITY,
        wavelength_edges=edges,
    )
    assert response.sizes["wavelength_bin"] == edges.size - 1
    np.testing.assert_allclose(response.wavelength_grid.values, 0.5 * (edges[1:] + edges[:-1]).value)
    np.testing.assert_allclose(_bin_width(response).values, np.diff(edges.value))
    assert response.wavelength_grid.attrs["bounds"] == "wavelength_bounds"
    assert response.wavelength_bounds.attrs["units"] == "Angstrom"
    # a line far narrower than its bin is conserved exactly
    np.testing.assert_allclose(_integral_ratio(response, line_list), 1.0, rtol=1e-12)


def test_contaminants_integrate_to_the_sum_of_their_areas():
    line_list = fake_line_list(5)
    response = create_spectral_response(
        line_list,
        DEFAULT_WAVELENGTH_GRID,
        main_lines=None,
        doppler_velocity=DOPPLER_VELOCITY,
        include_contaminants=True,
        bin_method="average",
    )
    assert list(response.line.values) == ["contaminants"]
    total = (line_list.gofnt / RESPONSE_NORMALIZATION).sum("transition")
    integral = (response.spectral_response * _bin_width(response)).sum("wavelength_bin").isel(line=0)
    np.testing.assert_allclose(integral.values, total.broadcast_like(integral).values, rtol=1e-12)


def test_grid_bounds_are_halfway_between_samples():
    line_list = fake_line_list(1)
    grid = np.arange(170.0, 172.0, 0.2) * u.AA
    response = create_spectral_response(line_list, grid, main_lines=[line_list.full_name[0].item()])
    np.testing.assert_allclose(_bin_width(response).values, 0.2)
    np.testing.assert_allclose(response.wavelength_bounds.mean("bound").values, grid.value)
    assert response.wavelength_bounds.dims == ("wavelength_bin", "bound")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bin_method": "spline"}, "bin_method"),
        ({"wavelength_edges": np.array([170.0, 171.0, 170.5]) * u.AA, "wavelength_grid": None}, "increasing"),
        ({"wavelength_edges": np.array([170.0, 171.0]) * u.AA}, "exactly one"),
        ({"wavelength_grid": None}, "exactly one"),
    ],
)
def test_rejects_bad_bins(kwargs, match):
    line_list = fake_line_list(1)
    arguments = {
        "wavelength_grid": DEFAULT_WAVELENGTH_GRID,
        "main_lines": [line_list.full_name[0].item()],
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        create_spectral_response(line_list, **arguments)


def _stepped_effective_area():
    """
    An area with a factor-of-three step at 171.8 A (beyond every Doppler-shifted line)
    and a 1 %/A slope.
    """
    wavelength = np.linspace(165.0, 177.0, 121)
    values = np.where(wavelength < 171.8, 1.0, 3.0) * (1 + 0.01 * (wavelength - 165.0))
    return fake_effective_area(values, wavelength)


def _total(response):
    width = response.wavelength_bounds.isel(bound=1) - response.wavelength_bounds.isel(bound=0)
    return (response.spectral_response * width).sum("wavelength_bin")


def test_methods_agree_on_the_fine_grid():
    line_list = fake_line_list(3)
    kwargs = {
        "main_lines": [line_list.full_name[0].item()],
        "doppler_velocity": DOPPLER_VELOCITY,
        "effective_area": _stepped_effective_area(),
    }
    centered = create_spectral_response(line_list, DEFAULT_WAVELENGTH_GRID, bin_method="center", **kwargs)
    averaged = create_spectral_response(line_list, DEFAULT_WAVELENGTH_GRID, **kwargs)
    assert centered.spectral_response.attrs["effective_area_at"] == "bin center"
    assert averaged.spectral_response.attrs["effective_area_at"] == "line center"
    # the lines sit on the smooth 1 %/A slope: the area at the center and the area across
    # the 2 mA grid differ only through the slope times the line width
    np.testing.assert_allclose(_total(averaged).values, _total(centered).values, rtol=1e-5)


def test_coarse_bins_keep_the_fine_grid_total():
    """
    Bins of 0.85 A, about fifty times the width of the lines, one of them straddling the
    area step.
    """
    line_list = fake_line_list(3)
    kwargs = {
        "main_lines": [line_list.full_name[0].item()],
        "doppler_velocity": DOPPLER_VELOCITY,
        "effective_area": _stepped_effective_area(),
        "bin_method": "average",
    }
    fine = create_spectral_response(line_list, DEFAULT_WAVELENGTH_GRID, **kwargs)
    coarse = create_spectral_response(
        line_list, wavelength_edges=create_log_wavelength_edges([170, 172] * u.AA, 0.005), **kwargs
    )
    np.testing.assert_allclose(_total(coarse).values, _total(fine).values, rtol=1e-6)


def test_area_is_zero_outside_the_curve():
    line_list = fake_line_list(1)
    area = fake_effective_area(np.ones(51), np.linspace(172.0, 177.0, 51))
    response = create_spectral_response(
        line_list,
        DEFAULT_WAVELENGTH_GRID,
        main_lines=[line_list.full_name[0].item()],
        effective_area=area,
        bin_method="average",
    )
    assert np.isfinite(response.spectral_response.values).all()
    assert float(np.abs(response.spectral_response).max()) == 0.0


def test_create_log_wavelength_edges():
    edges = create_log_wavelength_edges([25, 900] * u.AA, 0.01)
    np.testing.assert_allclose((edges[1:] / edges[:-1]).value, 1.01)
    assert edges[0] == 25 * u.AA
    assert edges[-2] < 900 * u.AA <= edges[-1]
    with pytest.raises(ValueError, match="fractional_width"):
        create_log_wavelength_edges([100, 50] * u.AA, 0.01)


def test_contaminants_take_the_area_at_their_own_wavelengths():
    """
    All lines summed into the contaminants, as an imager takes them, are the sum of the
    same lines kept apart: each is weighted by the area at its own wavelength.
    """
    line_list = fake_line_list(3, wavelength=[170.2, 171.0, 171.8])
    area = fake_effective_area([1.0, 3.0], [170.0, 172.0])
    edges = np.linspace(170.0, 172.0, 9) * u.AA
    velocities = [-100.0, 0.0, 100.0] * u.km / u.s
    options = {"wavelength_edges": edges, "effective_area": area, "doppler_velocity": velocities}

    summed = create_spectral_response(line_list, main_lines=None, include_contaminants=True, **options)
    apart = create_spectral_response(line_list, main_lines=list(line_list.full_name.values), **options)

    np.testing.assert_allclose(
        summed.spectral_response.sel(line="contaminants").values,
        apart.spectral_response.sum("line").transpose(*summed.spectral_response.sel(line="contaminants").dims).values,
        rtol=1e-12,
    )


def test_inputs_in_other_units_give_the_same_response():
    """
    Edges in nm are converted, not taken as Angstrom.
    """
    line_list = fake_line_list(2)
    edges = np.linspace(170.0, 172.0, 9) * u.AA
    options = {"main_lines": None, "include_contaminants": True, "doppler_velocity": [-100.0, 0.0, 100.0] * u.km / u.s}

    reference = create_spectral_response(line_list, wavelength_edges=edges, **options)
    converted = create_spectral_response(line_list, wavelength_edges=edges.to(u.nm), **options)

    xr.testing.assert_allclose(converted.spectral_response, reference.spectral_response, rtol=1e-12)
    np.testing.assert_allclose(converted.wavelength_bounds, reference.wavelength_bounds)
    np.testing.assert_allclose(
        create_log_wavelength_edges([17.0, 17.2] * u.nm, 0.01).value,
        create_log_wavelength_edges([170.0, 172.0] * u.AA, 0.01).value,
    )
