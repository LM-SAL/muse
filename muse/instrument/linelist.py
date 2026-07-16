"""CHIANTI line lists with contribution functions (GOFNT)."""

import os
import re
import warnings
from types import ModuleType
from importlib import reload

import numpy as np
import xarray as xr

__all__ = ["create_chianti_line_list"]


def create_chianti_line_list(
    temperature: xr.DataArray,
    density: xr.DataArray | None = None,
    pressure: xr.DataArray | None = None,
    abundance: str | None = None,
    wavelength_range=None,
    minimum_abundance: float | None = None,
    element_list: list[str] | None = None,
    ion_list: list[str] | None = None,
) -> xr.Dataset:
    """
    Generate a line list with contribution functions using ChiantiPy.

    Parameters
    ----------
    temperature : `xarray.DataArray`
        Temperature array in K with a ``logT`` dimension.
    density : `xarray.DataArray`, optional
        Electron density array in cm^-3. Mutually exclusive with ``pressure``.
    pressure : `xarray.DataArray`, optional
        Electron pressure array in K cm^-3. Mutually exclusive with ``density``.
    abundance : `str`, optional
        CHIANTI abundance name, e.g. ``"sun_coronal_2021_chianti"``.
    wavelength_range : array-like
        Two-element wavelength range in Angstroms.
    minimum_abundance : `float`, optional
        Minimum elemental abundance to keep. Mutually exclusive with
        ``element_list`` and ``ion_list``.
    element_list : `list` of `str`, optional
        CHIANTI element symbols to include, such as ``"fe"`` and ``"o"``.
        Mutually exclusive with ``ion_list`` and ``minimum_abundance``.
    ion_list : `list` of `str`, optional
        CHIANTI ion names to include, such as ``"fe_9"``. Mutually exclusive
        with ``element_list`` and ``minimum_abundance``.

    Returns
    -------
    `xarray.Dataset`
        Line list with contribution functions and per-transition metadata.

    Notes
    -----
    The ``XUVTOP`` environment variable must point to a local CHIANTI database.
    ChiantiPy's ``gui`` default is forced off for headless batch jobs.
    """
    wavelength_range = _validate_line_list_inputs(temperature, density, pressure, wavelength_range)
    element_list, ion_list = _normalize_species_selection(element_list, ion_list)
    _validate_species_selection(minimum_abundance, element_list, ion_list)

    chiantipy_version, ch = _initialize_chianti()

    if density is not None:
        temperature_bc, density_bc = xr.broadcast(temperature, density)
        extra_coord_name = density.dims[0]
        extra_coord = np.log10(density.data)
    else:
        density_bc = pressure / temperature
        temperature_bc = temperature.broadcast_like(density_bc)
        extra_coord_name = pressure.dims[0]
        extra_coord = pressure.data
    temperature_flat = temperature_bc.data.reshape(-1)
    density_flat = density_bc.data.reshape(-1)

    chianti_kwargs = {
        "em": 1.0,
        "abundance": abundance,
        "allLines": True,
        "keepIons": True,
        "minAbund": minimum_abundance,
        "ionList": ion_list,
        "elementList": element_list,
    }
    bunch = ch.bunch(temperature_flat, density_flat, wavelength_range, **chianti_kwargs)

    line_list = _chianti_bunch_to_dataset(
        bunch,
        temperature=temperature,
        temperature_bc=temperature_bc,
        extra_coords={extra_coord_name: extra_coord},
        abundance=abundance,
        wavelength_range=wavelength_range,
        chiantipy_version=chiantipy_version,
    )
    if element_list is not None:
        line_list.attrs["element_list"] = ",".join(element_list)
    if ion_list is not None:
        line_list.attrs["ion_list"] = ",".join(ion_list)
    return line_list


def _initialize_chianti() -> tuple[str, ModuleType]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            import ChiantiPy  # noqa: PLC0415
        except ImportError:
            msg = "ChiantiPy is required for this function, install it with `pip install muse[chianti]`"
            raise ImportError(msg) from None

    xuvtop = os.environ.get("XUVTOP")
    if xuvtop is None:
        msg = (
            "The XUVTOP environment variable is not set; ChiantiPy cannot locate the CHIANTI database. "
            "Point it at a local copy of the database, e.g. `export XUVTOP=/path/to/chianti/dbase` "
            "(available from https://www.chiantidatabase.org)."
        )
        raise OSError(msg)

    with warnings.catch_warnings():
        # Without a chiantirc file, ChiantiPy evaluates os.path.isfile(False)
        # at import, which raises a RuntimeWarning on Python >= 3.14.
        warnings.simplefilter("ignore", RuntimeWarning)
        import ChiantiPy.tools.data as chdata  # noqa: PLC0415

        if not hasattr(chdata, "Defaults") or getattr(chdata, "Xuvtop", None) != xuvtop:
            chdata = reload(chdata)
        import ChiantiPy.core as ch  # noqa: PLC0415

    if not hasattr(chdata, "Defaults"):
        msg = f"ChiantiPy could not initialize the CHIANTI database at {xuvtop}"
        raise OSError(msg)

    chdata.Defaults["gui"] = False
    return ChiantiPy.__version__, ch


def _chianti_bunch_to_dataset(
    bunch,
    *,
    temperature: xr.DataArray,
    temperature_bc: xr.DataArray,
    extra_coords: dict[str, np.ndarray],
    abundance: str | None,
    wavelength_range: tuple[float, float],
    chiantipy_version: str,
) -> xr.Dataset:
    import ChiantiPy.tools.io as chio  # noqa: PLC0415

    per_transition = {
        "ion_name": bunch.Intensity["ionS"],
        "wavelength": bunch.Intensity["wvl"],
        "lower_level_label": bunch.Intensity["pretty1"],
        "upper_level_label": bunch.Intensity["pretty2"],
        "lower_level_index": bunch.Intensity["lvl1"],
        "upper_level_index": bunch.Intensity["lvl2"],
        "spectroscopic_name": np.array([bunch.IonInstances[ion].Spectroscopic for ion in bunch.Intensity["ionS"]]),
        "atomic_number": np.array([bunch.IonInstances[ion].Z for ion in bunch.Intensity["ionS"]]),
        "observed": bunch.Intensity["obs"] == "Y",
    }
    line_list = xr.Dataset({name: ("trans_index", values) for name, values in per_transition.items()})

    gofnt_values = bunch.Intensity["intensity"].reshape((*temperature_bc.data.shape, 1, -1))
    line_list["gofnt"] = xr.DataArray(
        gofnt_values,
        dims=(*temperature_bc.dims, "abundance", "trans_index"),
        coords={"logT": np.log10(temperature), **extra_coords, "abundance": np.array([abundance])},
    )
    line_list["logT_peak"] = np.log10(temperature[{"logT": line_list.gofnt.argmax(dim="logT")}])
    line_list["full_name"] = (
        line_list.spectroscopic_name.astype(object) + " " + line_list.wavelength.astype(str).astype(object)
    )

    line_list.attrs["Chiantipy"] = chiantipy_version
    line_list.attrs["Chianti"] = chio.versionRead()
    line_list.gofnt.attrs["units"] = "erg cm3 / (s sr)"

    in_range = (line_list.wavelength > wavelength_range[0]) & (line_list.wavelength < wavelength_range[1])
    return line_list.isel(trans_index=in_range)


def _validate_line_list_inputs(
    temperature: xr.DataArray,
    density: xr.DataArray | None,
    pressure: xr.DataArray | None,
    wavelength_range,
) -> tuple[float, float]:
    if density is None and pressure is None:
        msg = "Specify density or pressure"
        raise ValueError(msg)
    if density is not None and pressure is not None:
        msg = "density and pressure are mutually exclusive"
        raise ValueError(msg)

    _validate_positive_data_array(temperature, "temperature", dimension="logT")
    plasma_name, plasma_grid = ("density", density) if density is not None else ("pressure", pressure)
    _validate_positive_data_array(plasma_grid, plasma_name)

    return _validate_wavelength_range(wavelength_range)


def _validate_wavelength_range(wavelength_range) -> tuple[float, float]:
    try:
        values = np.asarray(wavelength_range, dtype=float)
    except (TypeError, ValueError):
        msg = "wavelength_range must contain exactly two values"
        raise ValueError(msg) from None
    if values.shape != (2,):
        msg = "wavelength_range must contain exactly two values"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "wavelength_range must contain only finite values"
        raise ValueError(msg)
    lower, upper = values
    if lower >= upper:
        msg = "wavelength_range must be in increasing order"
        raise ValueError(msg)
    return float(lower), float(upper)


def _validate_positive_data_array(values, name: str, *, dimension: str | None = None) -> None:
    if not isinstance(values, xr.DataArray):
        msg = f"{name} must be an xarray.DataArray"
        raise TypeError(msg)
    expected_dims = (dimension,) if dimension is not None else None
    if (expected_dims is not None and values.dims != expected_dims) or (expected_dims is None and values.ndim != 1):
        qualifier = f"one-dimensional {dimension}" if dimension is not None else "one-dimensional"
        msg = f"{name} must be a {qualifier} array"
        raise ValueError(msg)
    if values.size == 0:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    if not np.issubdtype(values.dtype, np.number):
        msg = f"{name} must contain numeric values"
        raise TypeError(msg)
    if not np.all(np.isfinite(values.data)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if np.any(values.data <= 0):
        msg = f"{name} must contain only positive values"
        raise ValueError(msg)


def _normalize_species_selection(element_list, ion_list):
    if element_list is not None and ion_list is not None:
        msg = "element_list and ion_list are mutually exclusive"
        raise ValueError(msg)

    if element_list is not None:
        return _normalize_species_names(element_list, "element_list", r"[a-z]{1,2}"), None
    if ion_list is not None:
        return None, _normalize_species_names(ion_list, "ion_list", r"[a-z]{1,2}_[1-9][0-9]*d?")
    return None, None


def _normalize_species_names(values, name, pattern):
    if isinstance(values, str) or not isinstance(values, list | tuple):
        msg = f"{name} must be a list of strings"
        raise TypeError(msg)
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        msg = f"{name} must contain unique, non-empty strings"
        raise ValueError(msg)
    normalized = [value.strip().lower() for value in values]
    if len(set(normalized)) != len(normalized):
        msg = f"{name} must contain unique, non-empty strings"
        raise ValueError(msg)
    normalized.sort()
    invalid = [value for value in normalized if re.fullmatch(pattern, value) is None]
    if invalid:
        msg = f"invalid {name}: {', '.join(invalid)}"
        raise ValueError(msg)
    return normalized


def _validate_species_selection(minimum_abundance, element_list, ion_list):
    if minimum_abundance is not None and (element_list is not None or ion_list is not None):
        msg = "minimum_abundance is mutually exclusive with element_list and ion_list"
        raise ValueError(msg)
