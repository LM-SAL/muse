"""
Helpers for downloading the example data used by the documentation gallery.
"""

import os
from pathlib import Path

__all__ = ["fetch_example_data"]

_SAMPLE_DATA_URL = "https://github.com/LM-SAL/muse-data/releases/download/v1/"

#: File name -> (download URL, SHA-256 hash, cache subdirectory).
_REGISTRY = {
    "muse_example_vdem.zarr": (
        f"{_SAMPLE_DATA_URL}muse_example_vdem.zarr.tar.gz",
        "sha256:9736243c13be83b62986095f2cdfc9884c9eae941ff0498c2c2d37551b079c50",
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
    "eis_chianti_line_list_174_175_FeX_sun_coronal_2021_chianti_density.nc": (
        f"{_SAMPLE_DATA_URL}eis_chianti_line_list_174_175_FeX_sun_coronal_2021_chianti_density.nc",
        "sha256:c5ee652cee96c1224c338a526fedea631d85fe2bb37499c2ab6254991d9e081c",
        "chianti_line_lists",
    ),
    "muse_sg_response_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti_effarea.zarr": (
        f"{_SAMPLE_DATA_URL}muse_sg_response_108_FeXIX108.355_FeXXI108.117_sun_coronal_2021_chianti_effarea.zarr.tar.gz",
        "sha256:cafddf127568ec769caf9ffa6786757e8d78f75c032d24214d3892eef50b6009",
        "synthesis_tutorial",
    ),
    "muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea.zarr": (
        f"{_SAMPLE_DATA_URL}muse_sg_response_171_FeIX171.073_sun_coronal_2021_chianti_effarea.zarr.tar.gz",
        "sha256:5760826306bfffd1023b0e97a3a24b1d251a470edccdaf13bcb98e2d406f1477",
        "synthesis_tutorial",
    ),
    "muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea.zarr": (
        f"{_SAMPLE_DATA_URL}muse_sg_response_284_FeXV284.163_sun_coronal_2021_chianti_effarea.zarr.tar.gz",
        "sha256:cbc21e89fbcd90d599a2976dc9bcac4c5af0b18ae1cf6beb8022f2d4aadf80d9",
        "synthesis_tutorial",
    ),
    "EIS_EffArea_B.005": (
        "https://hesperia.gsfc.nasa.gov/ssw/hinode/eis/response/EIS_EffArea_B.005",
        "sha256:b56e5c2873b10bdc9bd31d0f342e5ecc0491ff752ece5807f1d8de2f94d15d64",
        "eis_calibration",
    ),
    "muse_synthetic_spectra.nc": (
        f"{_SAMPLE_DATA_URL}muse_synthetic_spectra.nc",
        "sha256:9b9ca430c298a719f7d0b0cb8ff66aa6c9b8d9c91299799d6d506a8d4cd79f58",
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
    url, known_hash, subdir = _REGISTRY[name]
    cache = Path(pooch.os_cache("muse"))
    if name.endswith(".zarr"):
        # Zarr stores are directories, published as tarballs; retrieve extracts once.
        pooch.retrieve(
            url,
            known_hash=known_hash,
            fname=f"{name}.tar.gz",
            path=cache,
            processor=pooch.Untar(extract_dir=subdir),
        )
        return cache / subdir / name
    return Path(pooch.retrieve(url, known_hash=known_hash, fname=name, path=cache / subdir))
