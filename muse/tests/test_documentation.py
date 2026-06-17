import pytest

from muse.utils.documentation import format_docstring


def test_format_docstring_rejects_unknown_field():
    with pytest.raises(AttributeError, match="not_a_field"):
        format_docstring("DEFAULTS_MUSE", value="not_a_field")
