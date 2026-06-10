import attrs

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
        object. Field names are validated against the attrs field definitions, so a
        typo raises at import time. Non-string values are rendered literally.

    Notes
    -----
    The rendered value is the one at import time; the attribute path is included so
    the live value can always be looked up.
    """
    from muse import variables  # NOQA: PLC0415 - Circular import

    defaults = getattr(variables, defaults_name)
    try:
        fields = attrs.fields_dict(type(defaults))
    except attrs.exceptions.NotAnAttrsClassError as exc:
        msg = f"{defaults_name} is not an attrs-based defaults object: {type(defaults)!r}"
        raise TypeError(msg) from exc
    substitutions = {}
    for param, target in param_to_field.items():
        if isinstance(target, str):
            if target not in fields:
                msg = f"{target!r} is not a field of {defaults_name} ({type(defaults).__name__})"
                raise AttributeError(msg)
            substitutions[param] = f"``{defaults_name}.{target}={getattr(defaults, target)}``"
        else:
            substitutions[param] = f"{target}"

    def format_doc(f):
        if f.__doc__:
            f.__doc__ = f.__doc__.format(**substitutions)
        return f

    return format_doc
