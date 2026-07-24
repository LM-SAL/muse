import pytest

from muse.instrument.utils import read_response
from muse.utils.documentation import format_docstring


def test_format_docstring_rejects_unknown_field():
    with pytest.raises(AttributeError, match="not_a_field"):
        format_docstring("DEFAULTS_MUSE", value="not_a_field")


def test_format_docstring_links_to_the_defaults_class():
    # The link target is the field on the class: DEFAULTS_MUSE is an instance, so it has
    # no documentation target of its own and Sphinx cannot resolve a reference to it.
    assert (
        ":attr:`DEFAULTS_MUSE.ccd_gain <muse.variables_schema.InstrumentDefaults.ccd_gain>`"
        " = ``10.0 electron / DN``" in read_response.__doc__
    )
