"""
Read and write the example data used by the documentation gallery.
"""

import os
from pathlib import Path
from collections.abc import Callable

import xarray as xr

from muse import __version__
from muse.log import logger
from muse.utils.utils import _open_dataset_file, _replace_atomically, _to_zarr

__all__ = ["fetch_example_data", "open_example_dataset", "save_example_dataset"]

# Development versions take main's data (dev), releases the latest release's (stable).
_EXAMPLE_DATA_URL = (
    f"https://github.com/LM-SAL/muse-data/releases/download/{'dev' if '.dev' in __version__ else 'stable'}/"
)

#: File name -> (download URL, SHA-256 hash, cache subdirectory).
_REGISTRY = {
    "muse_example_vdem.zarr": (
        f"{_EXAMPLE_DATA_URL}muse_example_vdem_20261121.zarr.tar.gz",
        "sha256:b9b607375d7390fa6b31b16910ffd9ce84540a91920cef8fb2328ad8f4483d4d",
        "muse_example_vdem",
    ),
    "aia_chianti_line_list_94_Fe_sun_coronal_2021_chianti.nc": (
        f"{_EXAMPLE_DATA_URL}aia_chianti_line_list_94_Fe_sun_coronal_2021_chianti.nc",
        "sha256:5e431851b821df6f2e8cf741954642f4904771ef6dcaee21165597d11173d8e6",
        "chianti_line_lists",
    ),
    "eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.nc": (
        f"{_EXAMPLE_DATA_URL}eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.nc",
        "sha256:73077c14781cf7f60bf0d1556defc22051ae03cbcbc37aaf572b26ef011769e9",
        "chianti_line_lists",
    ),
    "eis_chianti_line_list_174_175_FeX_sun_coronal_2021_chianti_density.nc": (
        f"{_EXAMPLE_DATA_URL}eis_chianti_line_list_174_175_FeX_sun_coronal_2021_chianti_density.nc",
        "sha256:1ca7ff0e8d184d3de5052d4c657fdd2ec92d6639be587fbdd1373656f9de31a3",
        "chianti_line_lists",
    ),
    "muse_chianti_line_list_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti.nc": (
        f"{_EXAMPLE_DATA_URL}muse_chianti_line_list_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti.nc",
        "sha256:7a4588bcb2a0f29c2f0fae3085ed59a24791256a475be41d61f481d4b0bafcaf",
        "chianti_line_lists",
    ),
    "muse_chianti_line_list_171_FeIX171.073_sun_coronal_2021_chianti.nc": (
        f"{_EXAMPLE_DATA_URL}muse_chianti_line_list_171_FeIX171.073_sun_coronal_2021_chianti.nc",
        "sha256:e7a459021f4ab1ec6c63264bd8b9a9ac8a015c4bb2029f707664aba09f164f5b",
        "chianti_line_lists",
    ),
    "muse_chianti_line_list_171_FeIX171.073_sun_coronal_2021_chianti_density.nc": (
        f"{_EXAMPLE_DATA_URL}muse_chianti_line_list_171_FeIX171.073_sun_coronal_2021_chianti_density.nc",
        "sha256:7ae1380f64d5f57d1fd9bd5061930caa5ddbde54f0052f91b39e48618124b67f",
        "chianti_line_lists",
    ),
    "muse_chianti_line_list_284_FeXV284.163_sun_coronal_2021_chianti.nc": (
        f"{_EXAMPLE_DATA_URL}muse_chianti_line_list_284_FeXV284.163_sun_coronal_2021_chianti.nc",
        "sha256:0b98a37d66b1553691bb6b3219e6ee084a2d1a25d835e6fce8addecb826ac3a2",
        "chianti_line_lists",
    ),
    "flare_nature_astro": (
        f"{_EXAMPLE_DATA_URL}flare_nature_astro.tar.gz",
        "sha256:f14aaf1608033ff6c5ac39f053f1c5f0a48ba7cb9c522350d146ff9561d3b349",
        "flare_nature_astro",
    ),
    "muse_sg_response_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti_effarea.zarr": (
        f"{_EXAMPLE_DATA_URL}muse_sg_response_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti_effarea_20261121.zarr.tar.gz",
        "sha256:49a8a88f5e0e9bbf36a524b29904fd6de47efcf151ea0ec34f53612046d46c82",
        "synthesis_tutorial",
    ),
    "muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea.zarr": (
        f"{_EXAMPLE_DATA_URL}muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea_20261121.zarr.tar.gz",
        "sha256:7325af4a88a2e4acbbf4d61afd1bc2a840d0740de9b3d7f99a15c8a945d4aa8b",
        "synthesis_tutorial",
    ),
    "muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea.zarr": (
        f"{_EXAMPLE_DATA_URL}muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea_20261121.zarr.tar.gz",
        "sha256:948b8d0deedab97ac621472f2bc4779e6957bdc649da048ec4efc1526c2286b4",
        "synthesis_tutorial",
    ),
    "EIS_EffArea_B.005": (
        "https://hesperia.gsfc.nasa.gov/ssw/hinode/eis/response/EIS_EffArea_B.005",
        "sha256:b56e5c2873b10bdc9bd31d0f342e5ecc0491ff752ece5807f1d8de2f94d15d64",
        "eis_calibration",
    ),
    "muse_synthetic_spectra.nc": (
        f"{_EXAMPLE_DATA_URL}muse_synthetic_spectra_20261121.nc",
        "sha256:02eba6538e604fcc991837b360fd7db7a8af9554dc81ee632e54e3851df1eec8",
        "synthesis_tutorial",
    ),
}


# Published ZIPs share a download location and cache directory.
_ZARR_HASHES = {
    "aia_chianti_line_list_94_Fe_sun_coronal_2021_chianti.zarr.zip": "5a04495ad1f04982460bfc2206967b4377a1728b2b1017a1fa93cfc4b8f2f5c4",
    "eis_chianti_line_list_174_175_FeX_sun_coronal_2021_chianti_density.zarr.zip": "5bc9e262107269af7668135441fa430d90a4f1eac2d5da05eaaafaa58c5f8518",
    "eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.zarr.zip": "98d6ba0116dfc6b321f235df0bc6155b0463c6895ddb4b706ff1949f233ddf73",
    "muse_chianti_line_list_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti.zarr.zip": "08238e09c3c845a706749b1486b44198ecda1a56190c2dd167cdf06e8c15fe58",
    "muse_chianti_line_list_171_FeIX171.073_sun_coronal_2021_chianti.zarr.zip": "4b2f89a886b1a08dba21f07b7d84485ab427573ad32fa41aab02634e6a9beff7",
    "muse_chianti_line_list_171_FeIX171.073_sun_coronal_2021_chianti_density.zarr.zip": "9ea6ea97aad8ad53d373d900f201df37686613d0769c7640e7b30d41be427ba4",
    "muse_chianti_line_list_284_FeXV284.163_sun_coronal_2021_chianti.zarr.zip": "09ef77b406d2517aea0af99815e688b29cb0c592a4c89109527b8f35e1537925",
    "muse_example_vdem.zarr.zip": "7f6a403f957ac8c13cd08303a24b5bf746495aa0c267d80341fb028a45a5a754",
    "muse_example_vdem_density.zarr.zip": "6e3f46e2ee737a2bd9d229f3b15ab0cd8d61e4e1a003033f2e9a4287a9e38508",
    "muse_sg_response_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti_effarea.zarr.zip": "368d4c954a5c6c84e24a4175bc5c852c535fa715a6af24424a95df879f029917",
    "muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea.zarr.zip": "50acd1f3d88377901c4178754925c672cda20f98c8498d62052c5ecf5288e5bd",
    "muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea.zarr.zip": "71e8a4c12183b726b3f90e0d75b05a6f38c971b08339fbd7063407c55f4f63b2",
    "muse_synthetic_spectra.zarr.zip": "8702961b5b7a12082f9fc0da54cb20216614150097f49900df95b3eb44b5a778",
}
_REGISTRY.update(
    {name: (f"{_EXAMPLE_DATA_URL}{name}", f"sha256:{checksum}", "gallery") for name, checksum in _ZARR_HASHES.items()}
)


def _tutorial_directory() -> Path:
    return Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))


def fetch_example_data(name: str) -> Path:
    """
    Download and cache an example data file of the documentation gallery.

    Gallery datasets written to
    ``MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR`` (by default
    ``examples/synthesis_tutorial/artifacts``) are used without downloading.

    Parameters
    ----------
    name : `str`
        File name, e.g. ``"muse_example_vdem.zarr.zip"``.

    Returns
    -------
    `pathlib.Path`
        Path to the local copy.
    """
    if Path(name).name == name and (
        name.endswith(".zarr.zip") or name in {"muse_example_vdem.zarr", "muse_synthetic_spectra.nc"}
    ):
        local_path = _tutorial_directory() / name
        if local_path.exists():
            logger.info(f"Using the local copy of {name}: {local_path.resolve()}")
            return local_path
    if name not in _REGISTRY:
        msg = f"{name!r} is not a known example data file, expected one of: {sorted(_REGISTRY)}"
        raise ValueError(msg)
    try:
        import pooch  # noqa: PLC0415
    except ImportError:
        msg = "pooch is required to download the example data, install it with `pip install pooch`"
        raise ImportError(msg) from None
    url, known_hash, subdirectory = _REGISTRY[name]
    cache = Path(pooch.os_cache("muse"))
    if url.endswith(".tar.gz"):
        # Directories, such as zarr stores, are published as tarballs; retrieve extracts once.
        pooch.retrieve(
            url,
            known_hash=known_hash,
            fname=f"{name}.tar.gz",
            path=cache,
            processor=pooch.Untar(extract_dir=subdirectory),
        )
        return cache / subdirectory / name
    return Path(pooch.retrieve(url, known_hash=known_hash, fname=name, path=cache / subdirectory))


def open_example_dataset(name: str, *, chunked: bool = False) -> xr.Dataset:
    """
    Open a locally generated or published gallery dataset.

    Parameters
    ----------
    name : str
        Dataset filename passed to `fetch_example_data`.
    chunked : bool, optional
        Read on demand with Dask using the stored chunks. By default, load the
        dataset into memory and close the file.

    Returns
    -------
    xarray.Dataset
        The example dataset. Call its `~xarray.Dataset.close` method when
        finished with a chunked dataset.
    """
    dataset = _open_dataset_file(fetch_example_data(name), chunked=chunked)
    if chunked:
        return dataset
    with dataset:
        return dataset.load()


def save_example_dataset(
    dataset: xr.Dataset, name: str, *, writer: Callable[[xr.Dataset, Path], None] = _to_zarr
) -> Path:
    """
    Write a zipped Zarr dataset in the gallery output directory.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to save.
    name : str
        Filename ending in ``.zarr.zip``. A successful write replaces any
        previous copy in ``MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR`` (by default
        ``examples/synthesis_tutorial/artifacts``).
    writer : callable, optional
        Function accepting a dataset and a new ZIP path. For detector responses,
        pass `muse.instrument.save_response` to use its compression and chunk sizes.

    Returns
    -------
    pathlib.Path
        Path to the completed archive.
    """
    if Path(name).name != name or not name.endswith(".zarr.zip"):
        msg = f"Expected a filename ending in .zarr.zip, got {name!r}"
        raise ValueError(msg)
    destination = _tutorial_directory() / name
    _replace_atomically(destination, lambda staging: writer(dataset, staging))
    return destination
