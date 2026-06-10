import pytest

from muse import variables
from muse.utils.documentation import format_docstring


def test_format_docstring_renders_field_path_and_value():
    assert "``DEFAULTS_MUSE.ccd_gain=10.0 electron / DN``" in variables.centroid_uncert_promised.__doc__


def test_format_docstring_rejects_non_attrs_defaults(monkeypatch):
    monkeypatch.setattr(variables, "NOT_DEFAULTS", object(), raising=False)

    with pytest.raises(TypeError, match="NOT_DEFAULTS is not an attrs-based defaults object"):
        format_docstring("NOT_DEFAULTS", value="field")
