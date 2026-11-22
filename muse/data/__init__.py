"""
Download the example data of the documentation gallery.
"""

import os
from pathlib import Path

from muse import __version__

__all__ = ["fetch_example_data"]

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


def fetch_example_data(name: str):
    """
    Download and cache an example data file of the documentation gallery.

    The VDEM and synthetic spectra the synthesis tutorial wrote to
    ``MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR`` (by default
    ``examples/synthesis_tutorial/artifacts``) are used without downloading.

    Parameters
    ----------
    name : `str`
        File name, e.g. ``"muse_example_vdem.zarr"``.

    Returns
    -------
    `pathlib.Path`
        Path to the local copy.
    """
    if name not in _REGISTRY:
        msg = f"{name!r} is not a known example data file, expected one of: {sorted(_REGISTRY)}"
        raise ValueError(msg)
    if name in {"muse_example_vdem.zarr", "muse_synthetic_spectra.nc"}:
        local_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
        local_path = local_dir / name
        if local_path.exists():
            return local_path
    try:
        import pooch
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
