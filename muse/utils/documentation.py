__all__ = ["format_docstring"]


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
    the live value can always be looked up.
    """
    from muse import variables  # NOQA: PLC0415 - Circular import

    defaults = getattr(variables, defaults_name)
    substitutions = {
        param: f"``{defaults_name}.{field}={getattr(defaults, field)}``"
        for param, field in param_to_field.items()
    }

    def format_doc(f):
        if f.__doc__:
            f.__doc__ = f.__doc__.format(**substitutions)
        return f

    return format_doc
