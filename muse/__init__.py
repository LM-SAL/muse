"""
====
muse
====

A Python library focused on interfacing with data (real or synthetic) for
NASA's Medium-Class Explorers (MIDEX) Multi-slit Solar Explorer (MUSE).

* Homepage: One day
* Documentation: One day
"""

import xarray as xr
from .version import version as __version__

# This should set all this library wide.
xr.set_options(keep_attrs=True, use_new_combine_kwarg_defaults=True)

__all__ = ["__version__", "xr"]
