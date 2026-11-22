"""
Project-wide logger configuration.
"""

import sys

from loguru import logger

__all__ = ["change_logging_level", "log_gpu_status", "logger"]


def change_logging_level(level: str) -> None:
    """
    Show Loguru messages at ``level`` and above, for scripts and notebooks.

    Loguru has one global logger: this removes **every** sink, those of the host
    application included, and adds one ``sys.stderr`` sink at ``level``. Applications
    that configure Loguru themselves should not call it; importing ``muse`` never does.

    Parameters
    ----------
    level : str
        Minimum level: ``"TRACE"``, ``"DEBUG"``, ``"INFO"``, ``"SUCCESS"``,
        ``"WARNING"``, ``"ERROR"`` or ``"CRITICAL"``.
    """
    logger.remove()
    logger.add(sys.stderr, level=level)


def log_gpu_status() -> None:
    """
    Log at debug level whether torch finds a CUDA GPU.

    Importing ``muse`` neither calls it nor imports torch.
    """
    import importlib.util  # NOQA: PLC0415

    found_gpu = False
    if importlib.util.find_spec("torch") is not None:
        import torch

        found_gpu = torch.cuda.is_available()
        if found_gpu:
            logger.debug(f"GPU CUDA - torch: {torch.cuda.device(torch.cuda.current_device())}")
    logger.debug(f"GPU FOUND STATUS: {found_gpu}")
