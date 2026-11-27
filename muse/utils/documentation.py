from collections.abc import Callable

__all__ = ["format_docstring"]


def format_docstring[F: Callable](defaults_name: str, /, **param_to_field: str) -> Callable[[F], F]:
    """
    Decorator that fills ``{placeholders}`` in a docstring with fields of a defaults
    object: the attribute path and the value at import time.

    Parameters
    ----------
    defaults_name : `str`
        Name of a defaults object in `muse.variables`, e.g. ``"DEFAULTS_MUSE"``.
    **param_to_field
        Field name (`str`) for each placeholder. A misspelled name fails at import time.

    Notes
    -----
    The path lets readers look up the live value. It links to the field on the defaults
    class, since an instance has no documentation page of its own.
    """
    from muse import variables  # NOQA: PLC0415

    defaults = getattr(variables, defaults_name)
    schema = type(defaults)
    schema_path = f"{schema.__module__}.{schema.__qualname__}"
    substitutions = {
        param: f":attr:`{defaults_name}.{field} <{schema_path}.{field}>` = ``{getattr(defaults, field)}``"
        for param, field in param_to_field.items()
    }

    def format_doc(f: F) -> F:
        if f.__doc__:
            f.__doc__ = f.__doc__.format(**substitutions)
        return f

    return format_doc
