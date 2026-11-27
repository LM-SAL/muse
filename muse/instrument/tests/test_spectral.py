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
    # on three temperatures, so each width must take its own one
    line_list = fake_line_list(1, wavelength=[171.0])
    # sampled at the bin centers: a bin average would add the width of a bin to the line's
    response = create_spectral_response(
        line_list,
        DEFAULT_WAVELENGTH_GRID,
        main_lines=line_list.full_name.values.tolist(),
        bin_method="center",
        **kwargs,
    )
    profile = response.spectral_response.sel(logT=6.0).squeeze().values
    offset = response.wavelength_grid.values - 171.0
    assert abs(np.sum(profile * offset) / np.sum(profile)) < 1e-6  # centered on the rest wavelength
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
    # xi is the 1/e velocity of the EIS/IRIS literature: it adds (lambda xi / c)^2 / 2 to sigma^2, on top of the
    # instrumental width.
    instrumental = _sigma_of_an_iron_line_at_1_MK(instrumental_fwhm=0.05 * u.AA)
    broadened = _sigma_of_an_iron_line_at_1_MK(instrumental_fwhm=0.05 * u.AA, nonthermal_velocity=20 * u.km / u.s)
    np.testing.assert_allclose(broadened**2 - instrumental**2, (171.0 * 20 / 299792.458) ** 2 / 2, rtol=1e-3)


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

    from_dataset = _build_spectral_response(line_list, effective_area=dataset, bin_method="center")

    # identical, coordinates included: the Dataset's scalar channel does not come along
    xr.testing.assert_identical(
        from_dataset, _build_spectral_response(line_list, effective_area=curve, bin_method="center")
    )


@pytest.mark.parametrize(
    ("effective_area", "match"),
    [
        (4.3 * u.cm**2, r"effective_area must be an xarray\.DataArray"),
        (xr.DataArray(4.3), "effective_area must be convertible to cm2"),
        (xr.DataArray(3.0 * u.s), "effective_area must be convertible to cm2"),
        (xr.DataArray(-1.0 * u.cm**2), "finite, non-negative"),
        (fake_effective_area(area_units="s"), "must be convertible to cm2"),
        (fake_effective_area(wavelength_units="s"), "effective_area wavelength coordinate"),
        (fake_effective_area(values=[1.0, np.nan, 1.0]), "finite, non-negative"),
        (fake_effective_area(values=[1.0, -1.0, 1.0]), "finite, non-negative"),
        (fake_effective_area(wavelength=[169.0, np.nan, 173.0]), "effective_area wavelength coordinate"),
        (fake_effective_area(wavelength=[169.0, 171.0, 171.0]), "effective_area wavelength coordinate"),
        (fake_effective_area().expand_dims(order=[1, 2]), "one-dimensional"),
    ],
)
def test_effective_area_is_validated(effective_area, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _build_spectral_response(fake_line_list(1), effective_area=effective_area)


@pytest.mark.parametrize(
    ("line_list", "match"),
    [
        (fake_line_list(2).rename({"transition": "trans_index"}), r"trans_index, as muse 0\.3 wrote line lists"),
        (fake_line_list(1).squeeze("transition"), "must lie along a transition dimension"),
        (fake_line_list(0), "must not be empty"),
        (fake_line_list(1).drop_vars("atomic_number"), "missing required variables: atomic_number"),
        (fake_line_list(1).assign(wavelength=("other", [170.6])), "transition dimension: wavelength"),
        (fake_line_list(1).assign(gofnt=lambda ds: ds.gofnt * np.nan), r"line_list\.gofnt must contain finite"),
        (fake_line_list(1, wavelength=[0.0]), r"line_list\.wavelength must contain positive"),
        (fake_line_list(1).assign(gofnt=lambda ds: -ds.gofnt), r"line_list\.gofnt must contain non-negative"),
        (fake_line_list(1).assign(atomic_number=("transition", [0])), "atomic_number must contain positive integers"),
        (
            fake_line_list(1).assign(wavelength=lambda ds: ds.wavelength.assign_attrs(units="s")),
            r"line_list\.wavelength",
        ),
        (fake_line_list(1).assign(gofnt=lambda ds: ds.gofnt.assign_attrs(units="Angstrom")), r"line_list\.gofnt"),
        (fake_line_list(1).assign_coords(logT=lambda ds: ds.logT.assign_attrs(units="Angstrom")), r"line_list\.logT"),
    ],
)
def test_line_list_is_validated(line_list, match):
    with pytest.raises(ValueError, match=match):
        _build_spectral_response_impl(line_list, DEFAULT_WAVELENGTH_GRID, main_lines=[])


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("instrumental_fwhm", [0.01, 0.02] * u.AA),
        ("instrumental_fwhm", -0.01 * u.AA),
        ("instrumental_fwhm", np.nan * u.AA),
        ("doppler_velocity", np.array([0.0, np.nan]) * u.km / u.s),
        ("nonthermal_velocity", np.array([-1.0, 0.0]) * u.km / u.s),
    ],
)
def test_numeric_controls_are_validated(name, value):
    with pytest.raises(ValueError, match=name):
        _build_spectral_response(fake_line_list(1), **{name: value})


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


def _total(response):
    return (response.spectral_response * _bin_width(response)).sum("wavelength_bin")


def _integral_ratio(response, line_list, index=0):
    return (_total(response) / (line_list.gofnt.isel(transition=index) / RESPONSE_NORMALIZATION)).values


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
        ({"wavelength_grid": 170.0 * u.AA}, "wavelength_grid"),
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


def test_methods_agree_on_the_fine_grid():
    line_list = fake_line_list(3)
    kwargs = {
        "main_lines": [line_list.full_name[0].item()],
        "doppler_velocity": DOPPLER_VELOCITY,
        "effective_area": fake_effective_area(1 + 0.01 * np.arange(13.0), np.arange(165.0, 178.0)),
    }
    centered = create_spectral_response(line_list, DEFAULT_WAVELENGTH_GRID, bin_method="center", **kwargs)
    averaged = create_spectral_response(line_list, DEFAULT_WAVELENGTH_GRID, **kwargs)
    assert centered.spectral_response.attrs["effective_area_at"] == "bin center"
    assert averaged.spectral_response.attrs["effective_area_at"] == "line center"
    # the lines sit on the smooth 1 %/A slope: the area at the center and the area across
    # the 2 mA grid differ only through the slope times the line width
    np.testing.assert_allclose(_total(averaged).values, _total(centered).values, rtol=1e-5)


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
    Every input is converted, not taken as Angstrom, km/s, erg cm3 / (s sr) or cm2, and
    is left as it was.
    """
    line_list = fake_line_list(1)
    area = fake_effective_area(np.linspace(1.0, 2.0, 20), np.linspace(169.0, 173.0, 20))
    line_list_si = line_list.assign(
        wavelength=(line_list.wavelength / 10).assign_attrs(units="nm"),
        gofnt=(line_list.gofnt * 1e-13).assign_attrs(units="J m3 / (s sr)"),
    )
    area_si = fake_effective_area(
        area.values * 1e-4, area.wavelength.values / 10, area_units="m2", wavelength_units="nm"
    )
    before = (line_list_si.copy(deep=True), area_si.copy(deep=True))
    edges = create_log_wavelength_edges([170.0, 172.0] * u.AA, 0.001)
    np.testing.assert_allclose(create_log_wavelength_edges([17.0, 17.2] * u.nm, 0.001), edges)
    for bins, bins_si in (
        ({}, {"wavelength_grid": DEFAULT_WAVELENGTH_GRID.to(u.nm)}),
        (
            {"wavelength_grid": None, "wavelength_edges": edges},
            {"wavelength_grid": None, "wavelength_edges": edges.to(u.nm)},
        ),
    ):
        reference = _build_spectral_response(
            line_list,
            instrumental_fwhm=0.02 * u.AA,
            doppler_velocity=[-20.0, 20.0] * u.km / u.s,
            nonthermal_velocity=[0.0, 5.0] * u.km / u.s,
            effective_area=area,
            **bins,
        )
        converted = _build_spectral_response(
            line_list_si,
            instrumental_fwhm=0.002 * u.nm,
            doppler_velocity=[-2e4, 2e4] * u.m / u.s,
            nonthermal_velocity=[0.0, 5e3] * u.m / u.s,
            effective_area=area_si,
            **bins_si,
        )
        xr.testing.assert_allclose(converted, reference)
    xr.testing.assert_identical(line_list_si, before[0])
    xr.testing.assert_identical(area_si, before[1])
