from pathlib import Path

__all__ = ["format_docstring"]


class _SkipAwareFileNameSortKey:
    """
    Sphinx-gallery ``within_subsection_order`` key; private API for ``docs/conf.py``.

    Sorts gallery scripts by filename, ignoring a leading ``skip_`` prefix, so
    numbered scripts (``02_...``, ``skip_01_...``) interleave in numeric order,
    while un-numbered ``skip_`` scripts sort before everything else.

    Parameters
    ----------
    src_dir : `str`
        The gallery subsection source directory (unused, required by sphinx-gallery).
    """

    def __init__(self, src_dir):
        self.src_dir = src_dir

    def __call__(self, filename):
        name = Path(filename).name
        stripped = name.removeprefix("skip_")
        unnumbered_skip = name.startswith("skip_") and not stripped[:1].isdigit()
        return (not unnumbered_skip, stripped)


def format_docstring(defaults_name, /, **param_to_field):
    """
    A function decorator that substitutes ``{placeholders}`` in the docstring with the
    attribute path and import-time value of fields on a defaults object from
    `muse.variables`.

    Parameters
    ----------
    defaults_name : `str`
        Name of a defaults object in `muse.variables`, e.g. ``"DEFAULTS_MUSE"``.
    **param_to_field
        Maps each docstring placeholder to a field name (`str`) on the defaults
        object. A typo in either name raises `AttributeError` at import time.

    Notes
    -----
    The rendered value is the one at import time; the attribute path is included so
    the live value can always be looked up. The path links to the field on the
    defaults class, since the defaults object itself is an instance and so has no
    documentation target of its own.
    """
    from muse import variables  # NOQA: PLC0415 - Circular import

    defaults = getattr(variables, defaults_name)
    schema = type(defaults)
    schema_path = f"{schema.__module__}.{schema.__qualname__}"
    substitutions = {
        param: f":attr:`{defaults_name}.{field} <{schema_path}.{field}>` = ``{getattr(defaults, field)}``"
        for param, field in param_to_field.items()
    }

    def format_doc(f):
        if f.__doc__:
            f.__doc__ = f.__doc__.format(**substitutions)
        return f

    return format_doc
