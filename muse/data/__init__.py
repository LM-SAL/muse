"""
Helpers for downloading the example data used by the documentation gallery.
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
        f"{_EXAMPLE_DATA_URL}muse_example_vdem.zarr.tar.gz",
        "sha256:3a7fbd748edfded16be710724fbf274a4f5d54e29db9a7e277c864792ec06223",
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
        f"{_EXAMPLE_DATA_URL}muse_sg_response_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti_effarea.zarr.tar.gz",
        "sha256:196ed29e403360e5f51308911ef25dc976dfb2806c960806a20673bbd198fde3",
        "synthesis_tutorial",
    ),
    "muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea.zarr": (
        f"{_EXAMPLE_DATA_URL}muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea.zarr.tar.gz",
        "sha256:de2b5ec2383cadab8311c358c7af59a19d27d8d5805ee45858e2062264fb3d4e",
        "synthesis_tutorial",
    ),
    "muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea.zarr": (
        f"{_EXAMPLE_DATA_URL}muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea.zarr.tar.gz",
        "sha256:fb3a9f5c6c23320a18fab69b20bd008bf714b5e1a2d42eb2a0bcc927d9c7032b",
        "synthesis_tutorial",
    ),
    "EIS_EffArea_B.005": (
        "https://hesperia.gsfc.nasa.gov/ssw/hinode/eis/response/EIS_EffArea_B.005",
        "sha256:b56e5c2873b10bdc9bd31d0f342e5ecc0491ff752ece5807f1d8de2f94d15d64",
        "eis_calibration",
    ),
    "muse_synthetic_spectra.nc": (
        f"{_EXAMPLE_DATA_URL}muse_synthetic_spectra.nc",
        "sha256:a34d4cd82288435f7d3df56d8069a34915eb9057998b57b4684d2796ff6e5c27",
        "synthesis_tutorial",
    ),
}


def fetch_example_data(name):
    """
    Download and cache one of the example data files used by the documentation gallery.

    A VDEM or synthetic spectrum already present in the synthesis-tutorial
    output directory (``MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR``, defaulting to
    ``examples/synthesis_tutorial/artifacts``) is returned without downloading.
    Every other file comes from the published copy.

    Parameters
    ----------
    name : `str`
        Name of the file to fetch, e.g. ``"muse_example_vdem.zarr"``.

    Returns
    -------
    `pathlib.Path`
        Path to the local copy of the file.
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
