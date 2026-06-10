"""
Project-wide logger configuration.
"""

import os
import sys

from loguru import logger

__all__ = ["change_logging_level", "log_gpu_status", "logger"]


def change_logging_level(level: str) -> None:
    """
    Change the logging level of the logger.

    Parameters
    ----------
    level : str
        The level to change the logger to. Must be one of the following:
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    """
    if os.environ.get("MUSE_DEBUG"):
        return
    logger.remove()
    logger.add(sys.stdout, level=level)


def log_gpu_status() -> None:
    """
    Log CUDA/GPU availability.

    Imports torch lazily and is not called at import time, so importing this module (or
    :mod:`muse`) stays torch-free. Call this explicitly when the GPU diagnostic is
    wanted.
    """
    import contextlib  # NOQA: PLC0415

    import torch  # NOQA: PLC0415

    found_gpu = torch.cuda.is_available()
    with contextlib.suppress(RuntimeError, AssertionError):
        logger.debug(f"GPU CUDA - pytorch: {torch.cuda.device(torch.cuda.current_device())}")
    logger.debug(f"GPU FOUND STATUS: {found_gpu}")


change_logging_level("INFO")
