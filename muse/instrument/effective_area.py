"""
Effective-area files: the wavelength-dependent area of one band, with its components.
"""

from pathlib import Path

import numpy as np
import xarray as xr

import astropy.units as u

from muse.utils.utils import _require_increasing_axis, add_history, coord_as_unit, require_unit

__all__ = ["create_effective_area", "effective_area_curve", "read_effective_area"]

_DN_PER_PHOTON = u.cm**2 * u.DN / u.ph


def create_effective_area(
    effective_area, wavelength: u.Quantity, band: u.Quantity, *, attrs: dict | None = None, **components
) -> xr.Dataset:
    """
    Build an effective-area dataset from arrays.

    Parameters
    ----------
    effective_area : `astropy.units.Quantity` or array-like
        The area at each wavelength, in cm2 (or cm2 DN / ph for an area that carries the
        camera gain); a plain array is in cm2. One value per wavelength, or a
        ``(wavelength, band)`` table.
    wavelength : `astropy.units.Quantity`
        Increasing wavelengths of the samples.
    band : `astropy.units.Quantity`
        Nominal wavelength of the band (or one per band column).
    attrs : `dict`, optional
        Attributes to record, e.g. the coating.
    **components
        Further curves of the same shape, such as ``mirror``, ``filter``, ``ccd``,
        ``grating`` or ``contamination``, stored alongside.

    Returns
    -------
    `xarray.Dataset`
        ``effective_area(wavelength, band)`` with units, plus the components.
    """
    call_inputs = dict(locals())
    unit = effective_area.unit if isinstance(effective_area, u.Quantity) else u.cm**2
    if not (unit.is_equivalent(u.cm**2) or unit.is_equivalent(_DN_PER_PHOTON)):
        msg = f"effective_area must be in cm2 or cm2 DN / ph, got {unit}"
        raise ValueError(msg)
    wavelength_values = np.asarray(wavelength.to_value(u.AA), dtype=float).reshape(-1)
    _require_increasing_axis(wavelength_values, "wavelength", positive=True)
    bands = np.atleast_1d(band.to_value(u.AA))
    shape = (wavelength_values.size, bands.size)

    def table(values):
        values = u.Quantity(values, unit).value if isinstance(values, u.Quantity) else np.asarray(values, dtype=float)
        return xr.DataArray(values.reshape(shape), dims=("wavelength", "band"), attrs={"units": str(unit)})

    dataset = xr.Dataset(
        {"effective_area": table(effective_area), **{name: table(values) for name, values in components.items()}},
        coords={
            "wavelength": ("wavelength", wavelength_values, {"units": str(u.AA)}),
            "band": ("band", bands, {"units": str(u.AA)}),
        },
        attrs=attrs or {},
    )
    add_history(dataset, call_inputs, create_effective_area)
    return dataset


def read_effective_area(path: str | Path) -> xr.Dataset:
    """
    Read an effective-area file written by `create_effective_area`.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        A zarr store or a netCDF file.

    Returns
    -------
    `xarray.Dataset`
        The dataset in memory, with ``effective_area(wavelength, band)``, its units convertible
        to cm2 or cm2 DN / ph, and ``wavelength`` in a wavelength unit.
    """
    path = Path(path)
    dataset = (xr.open_zarr(path, consolidated=False) if path.is_dir() else xr.open_dataset(path)).load()
    if "effective_area" not in dataset or "band" not in dataset.coords:
        msg = f"{path} must carry effective_area and a band coordinate"
        raise ValueError(msg)
    unit = require_unit(dataset, "effective_area", f"{path.name}: effective_area")
    if not (unit.is_equivalent(u.cm**2) or unit.is_equivalent(_DN_PER_PHOTON)):
        msg = f"{path.name}: effective_area units {unit} must be convertible to cm2 or cm2 DN / ph"
        raise ValueError(msg)
    coord_as_unit(dataset, "wavelength", u.AA, f"{path.name}: wavelength")
    if dataset.effective_area.dims != ("wavelength", "band"):
        dataset["effective_area"] = dataset.effective_area.transpose("wavelength", "band")
    return dataset


def effective_area_curve(dataset: xr.Dataset, band: u.Quantity | None = None) -> xr.DataArray:
    """
    The one-dimensional area curve of one band, as `create_spectral_response` takes it.

    Parameters
    ----------
    dataset : `xarray.Dataset`
        From `create_effective_area` or `read_effective_area`.
    band : `astropy.units.Quantity`, optional
        The band to select; by default the only one.

    Returns
    -------
    `xarray.DataArray`
        ``effective_area(wavelength)`` with its units and the ``wavelength`` coordinate in Angstrom.
    """
    if band is None:
        if dataset.sizes["band"] != 1:
            msg = f"the file holds {dataset.sizes['band']} bands; pass band"
            raise ValueError(msg)
        curve = dataset.effective_area.isel(band=0)
    else:
        curve = dataset.effective_area.sel(band=band.to_value(u.AA))
    wavelength = coord_as_unit(dataset, "wavelength", u.AA, "wavelength")
    return curve.drop_vars("band").assign_coords(wavelength=wavelength)
