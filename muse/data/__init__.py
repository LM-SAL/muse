"""
Helpers for downloading the example data used by the documentation gallery.
"""

import os
from pathlib import Path

__all__ = ["fetch_example_data"]

_EXAMPLE_DATA_URL = "https://github.com/LM-SAL/muse-data/releases/download/v1/"

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
        "sha256:098eef7a57549bbff1519bece9652fd04c4296aaf7939c21a2dac88a155a663a",
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
    if name.endswith(".zarr"):
        # Zarr stores are directories, published as tarballs; retrieve extracts once.
        pooch.retrieve(
            url,
            known_hash=known_hash,
            fname=f"{name}.tar.gz",
            path=cache,
            processor=pooch.Untar(extract_dir=subdirectory),
        )
        return cache / subdirectory / name
    return Path(pooch.retrieve(url, known_hash=known_hash, fname=name, path=cache / subdirectory))
