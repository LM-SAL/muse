"""
====
muse
====

A Python library focused on interfacing with data (real or synthetic) for
NASA's Medium-Class Explorers (MIDEX) Multi-slit Solar Explorer (MUSE).

* `Homepage <https://muse.lmsal.com/>`__
* Documentation: One day
"""

import contextlib
import os
import sys

import torch
import xarray
from loguru import logger

from .version import version as __version__

# This should set all this library wide.
# This is dangerous.
xarray.set_options(keep_attrs=True, use_new_combine_kwarg_defaults=True)


def change_logging_level(level: str) -> None:
    """
    Change the logging level of the logger.

    Parameters
    ----------
    level : str
        The level to change the logger to. Must be one of the following:
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    """
    if os.environ.get("MUSE_DEBUG"):
        return
    logger.remove()
    logger.add(sys.stdout, level=level)


change_logging_level("INFO")
FOUND_GPU = torch.cuda.is_available()
with contextlib.suppress(RuntimeError, AssertionError):
    logger.debug(f"GPU CUDA - pytorch: {torch.cuda.device(torch.cuda.current_device())}")
logger.debug(f"GPU FOUND STATUS: {FOUND_GPU}")

__all__ = [
    "FOUND_GPU",
    "__version__",
    "change_logging_level",
]
