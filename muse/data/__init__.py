"""
Helpers for downloading the example data used by the documentation gallery.
"""

import os
from pathlib import Path

__all__ = ["fetch_example_data"]

_SAMPLE_DATA_URL = "https://github.com/LM-SAL/muse-sample-data/releases/download/v1/"

#: File name -> (download URL, SHA-256 hash, cache subdirectory).
_REGISTRY = {
    "muse_example_vdem.zarr": (
        f"{_SAMPLE_DATA_URL}muse_example_vdem.zarr.tar.gz",
        "sha256:e0fea3be03421d405a4e47653020b68f35ebfa1d0433a4c45479e270d4dbaf76",
        "muse_example_vdem",
    ),
    "aia_chianti_line_list_94_Fe_sun_coronal_2021_chianti.nc": (
        f"{_SAMPLE_DATA_URL}aia_chianti_line_list_94_Fe_sun_coronal_2021_chianti.nc",
        "sha256:f76fe28b1325e809ea4ec61ba916af447826220f436464f1b39d5000af12274e",
        "chianti_line_lists",
    ),
    "eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.nc": (
        f"{_SAMPLE_DATA_URL}eis_chianti_line_list_195_FeXII_sun_coronal_2021_chianti.nc",
        "sha256:47d046bcfd2c8f4a6ca6417e5a3a26d0ce49abb5a4d3389720dd579a736c35b9",
        "chianti_line_lists",
    ),
    "euvst_chianti_line_list_174_175_FeX_sun_coronal_2021_chianti_density.nc": (
        f"{_SAMPLE_DATA_URL}euvst_chianti_line_list_174_175_FeX_sun_coronal_2021_chianti_density.nc",
        "sha256:c5ee652cee96c1224c338a526fedea631d85fe2bb37499c2ab6254991d9e081c",
        "chianti_line_lists",
    ),
    "muse_sg_response_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti_effarea.nc": (
        f"{_SAMPLE_DATA_URL}muse_sg_response_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti_effarea.nc",
        "sha256:953f743d87d81b3e62da068e009544d09cc7419103a6cd8a72b88f65e8c3965f",
        "synthesis_tutorial",
    ),
    "muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea.nc": (
        f"{_SAMPLE_DATA_URL}muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea.nc",
        "sha256:4560e331902ab7d30107ef2fca7528f07d9ac77a5fd70051e7da5729fc806569",
        "synthesis_tutorial",
    ),
    "muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea.nc": (
        f"{_SAMPLE_DATA_URL}muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea.nc",
        "sha256:e9acf61f70d8bf96698eb360890dd6a56a370cdb274d3308523c5a99fc1b0612",
        "synthesis_tutorial",
    ),
    "EIS_EffArea_B.005": (
        "https://hesperia.gsfc.nasa.gov/ssw/hinode/eis/response/EIS_EffArea_B.005",
        "sha256:b56e5c2873b10bdc9bd31d0f342e5ecc0491ff752ece5807f1d8de2f94d15d64",
        "eis_calibration",
    ),
    "muse_synthetic_spectra.nc": (
        f"{_SAMPLE_DATA_URL}muse_synthetic_spectra.nc",
        "sha256:7df335a9064e757345d4c155a4a02df734acc3db9370170682a47cd252e35257",
        "synthesis_tutorial",
    ),
}


def fetch_example_data(name):
    """
    Download and cache one of the example data files used by the documentation gallery.

    An example VDEM already present in the synthesis-tutorial output directory
    (``MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR``, defaulting to
    ``examples/synthesis_tutorial/artifacts``) is returned without downloading,
    so a VDEM regenerated locally with the tutorial wins over the published
    one. Every other file always comes from the published copy: the gallery
    itself writes some of them, and reading those back during a parallel
    gallery build would race the writer.

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
    try:
        import pooch
    except ImportError:
        msg = "pooch is required to download the example data, install it with `pip install pooch`"
        raise ImportError(msg) from None
    url, known_hash, subdir = _REGISTRY[name]
    cache = Path(pooch.os_cache("muse"))
    if name == "muse_example_vdem.zarr":
        local_dir = Path(os.environ.get("MUSE_SYNTHESIS_TUTORIAL_OUTPUT_DIR", "examples/synthesis_tutorial/artifacts"))
        local_path = local_dir / name
        if local_path.exists():
            return local_path
        # The zarr store is published as a tarball; retrieve extracts it once.
        pooch.retrieve(
            url,
            known_hash=known_hash,
            fname=f"{name}.tar.gz",
            path=cache,
            processor=pooch.Untar(extract_dir=subdir),
        )
        return cache / subdir / name
    return Path(pooch.retrieve(url, known_hash=known_hash, fname=name, path=cache / subdir))
