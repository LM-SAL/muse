import pytest

from muse.instrument.response import map_response_to_sg_detector
from muse.utils.documentation import format_docstring


def test_format_docstring_rejects_unknown_field():
    with pytest.raises(AttributeError, match="not_a_field"):
        format_docstring("DEFAULTS_MUSE", value="not_a_field")


def test_format_docstring_links_to_the_defaults_class():
    # The link target is the field on the class: DEFAULTS_MUSE is an instance, so it has
    # no documentation target of its own and Sphinx cannot resolve a reference to it.
    assert (
        ":attr:`DEFAULTS_MUSE.number_of_slits_SG <muse.variables_schema.InstrumentDefaults.number_of_slits_SG>`"
        " = ``35``" in map_response_to_sg_detector.__doc__
    )
