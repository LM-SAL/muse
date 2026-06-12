import pytest

from muse import variables
from muse.utils.documentation import format_docstring


def test_format_docstring_renders_field_path_and_value():
    assert "``DEFAULTS_MUSE.ccd_gain=10.0 electron / DN``" in variables.centroid_uncert_promised.__doc__


def test_format_docstring_rejects_unknown_field():
    with pytest.raises(AttributeError, match="not_a_field"):
        format_docstring("DEFAULTS_MUSE", value="not_a_field")
