"""
Atomic data: line lists and continua from CHIANTI, through ChiantiPy or fiasco.
"""

from muse.instrument.atomic.continuum import create_chianti_continuum, create_fiasco_continuum
from muse.instrument.atomic.linelist import (
    boost_transitions,
    create_chianti_line_list,
    create_fiasco_line_list,
    rank_line_list,
    select_transitions,
)

__all__ = [
    "boost_transitions",
    "create_chianti_continuum",
    "create_chianti_line_list",
    "create_fiasco_continuum",
    "create_fiasco_line_list",
    "rank_line_list",
    "select_transitions",
]
