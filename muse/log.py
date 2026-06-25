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

    Imports JAX lazily and is not called at import time, so importing this module (or
    :mod:`muse`) stays accelerator-backend-free. Call this explicitly when the GPU diagnostic is
    wanted.
    """
    import jax  # NOQA: PLC0415

    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    found_gpu = bool(gpu_devices)
    if gpu_devices:
        logger.debug(f"GPU CUDA - jax: {gpu_devices[0]}")
    logger.debug(f"GPU FOUND STATUS: {found_gpu}")
