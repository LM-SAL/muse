"""
Effective-area files: the wavelength-dependent area of one band, with its components.
"""

from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

import astropy.units as u

from muse.utils.utils import _require_increasing_axis, add_history, coord_as_unit, require_unit

__all__ = [
    "create_effective_area",
    "effective_area_wavelength_range",
    "read_effective_area",
    "select_effective_area_curve",
]

_DN_PER_PHOTON = u.cm**2 * u.DN / u.ph


@u.quantity_input(wavelength=u.AA, band=u.AA)
def create_effective_area(
    area: u.Quantity,
    wavelength: u.Quantity,
    *,
    band: u.Quantity,
    components: Mapping[str, u.Quantity] | None = None,
    attrs: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """
    Build an effective-area dataset from arrays.

    Parameters
    ----------
    area : `astropy.units.Quantity`
        The area at each wavelength, in cm2, or in cm2 DN / ph for an area that carries the
        camera gain. One value per wavelength, or a ``(wavelength, band)`` table.
    wavelength : `astropy.units.Quantity`
        Increasing wavelengths of the samples.
    band : `astropy.units.Quantity`
        Nominal wavelength of the band, or one per band column.
    components : `dict`, optional
        Further curves of the same shape and any unit, such as ``mirror``, ``filter``, ``ccd``
        or ``grating``, stored alongside under their names.
    attrs : `dict`, optional
        Attributes to record, e.g. the coating.

    Returns
    -------
    `xarray.Dataset`
        ``effective_area(wavelength, band)`` with its units, plus the components.
    """
    call_inputs = dict(locals())
    if not isinstance(area, u.Quantity):
        msg = "area must be an astropy.units.Quantity in cm2 or cm2 DN / ph"
        raise TypeError(msg)
    if not (area.unit.is_equivalent(u.cm**2) or area.unit.is_equivalent(_DN_PER_PHOTON)):
        msg = f"area must be in cm2 or cm2 DN / ph, got {area.unit}"
        raise ValueError(msg)
    wavelengths = np.asarray(wavelength.to_value(u.AA), dtype=float).reshape(-1)
    _require_increasing_axis(wavelengths, "wavelength", positive=True)
    bands = np.atleast_1d(band.to_value(u.AA))
    shape = (wavelengths.size, bands.size)

    def _as_table(values: u.Quantity) -> xr.DataArray:
        return xr.DataArray(
            np.asarray(values.value, dtype=float).reshape(shape),
            dims=("wavelength", "band"),
            attrs={"units": str(values.unit)},
        )

    dataset = xr.Dataset(
        {"effective_area": _as_table(area), **{name: _as_table(values) for name, values in (components or {}).items()}},
        coords={
            "wavelength": ("wavelength", wavelengths, {"units": str(u.AA)}),
            "band": ("band", bands, {"units": str(u.AA)}),
        },
        attrs=dict(attrs or {}),
    )
    add_history(dataset, call_inputs, create_effective_area)
    return dataset


def read_effective_area(effective_area_file: str | Path) -> xr.Dataset:
    """
    Read an effective-area file written by `create_effective_area`.

    Parameters
    ----------
    effective_area_file : `str` or `pathlib.Path`
        A zarr store or a netCDF file.

    Returns
    -------
    `xarray.Dataset`
        The dataset in memory, with ``effective_area(wavelength, band)``, its units convertible
        to cm2 or cm2 DN / ph, and ``wavelength`` in a wavelength unit.
    """
    effective_area_file = Path(effective_area_file)
    opened = (
        xr.open_zarr(effective_area_file, consolidated=False)
        if effective_area_file.is_dir()
        else xr.open_dataset(effective_area_file)
    )
    with opened:
        effective_area = opened.load()
    label = effective_area_file.name
    if "effective_area" not in effective_area or "band" not in effective_area.coords:
        msg = f"{label} must carry effective_area and a band coordinate"
        raise ValueError(msg)
    unit = require_unit(effective_area, "effective_area", f"{label}: effective_area")
    if not (unit.is_equivalent(u.cm**2) or unit.is_equivalent(_DN_PER_PHOTON)):
        msg = f"{label}: effective_area units {unit} must be convertible to cm2 or cm2 DN / ph"
        raise ValueError(msg)
    coord_as_unit(effective_area, "wavelength", u.AA, f"{label}: wavelength")
    # zarr hands back arrays with an explicit byte order, which netCDF then writes with a warning.
    native = {
        name: variable.astype(variable.dtype.newbyteorder("="))
        for name, variable in effective_area.variables.items()
        if variable.dtype.byteorder in "<>"
    }
    effective_area = effective_area.assign(
        {name: variable for name, variable in native.items() if name not in effective_area.coords}
    )
    effective_area = effective_area.assign_coords(
        {name: variable for name, variable in native.items() if name in effective_area.coords}
    )
    return effective_area.assign(effective_area=effective_area.effective_area.transpose("wavelength", "band"))


@u.quantity_input(band=u.AA)
def select_effective_area_curve(effective_area: xr.Dataset, *, band: u.Quantity | None = None) -> xr.DataArray:
    """
    The one-dimensional area curve of one band, as `create_spectral_response` takes it.

    Parameters
    ----------
    effective_area : `xarray.Dataset`
        From `create_effective_area` or `read_effective_area`.
    band : `astropy.units.Quantity`, optional
        The band to select; by default the only one.

    Returns
    -------
    `xarray.DataArray`
        ``effective_area(wavelength)`` with its units and the ``wavelength`` coordinate in Angstrom.
    """
    if band is None:
        if effective_area.sizes["band"] != 1:
            msg = f"the dataset holds {effective_area.sizes['band']} bands; pass band"
            raise ValueError(msg)
        curve = effective_area.effective_area.isel(band=0)
    else:
        curve = effective_area.effective_area.sel(band=band.to_value(u.AA))
    wavelength = coord_as_unit(effective_area, "wavelength", u.AA, "wavelength")
    return curve.drop_vars("band").assign_coords(wavelength=wavelength)


def effective_area_wavelength_range(effective_areas: Sequence[xr.Dataset]) -> u.Quantity:
    """
    The wavelength range every effective area spans together, as a line list must cover.

    Parameters
    ----------
    effective_areas : `list` of `xarray.Dataset`
        From `create_effective_area` or `read_effective_area`.

    Returns
    -------
    `astropy.units.Quantity`
        The lowest and the highest wavelength of all of them.
    """
    if not effective_areas:
        msg = "pass at least one effective area"
        raise ValueError(msg)
    wavelengths = [np.asarray(coord_as_unit(area, "wavelength", u.AA, "wavelength")) for area in effective_areas]
    return [min(w.min() for w in wavelengths), max(w.max() for w in wavelengths)] * u.AA
