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

    Optional accelerator backends are imported lazily and this is not called at import
    time, so importing this module (or :mod:`muse`) stays backend-free.
    """
    import importlib.util  # NOQA: PLC0415

    found_gpu = False
    if importlib.util.find_spec("jax") is not None:
        import jax  # NOQA: PLC0415

        try:
            gpu_devices = jax.devices("gpu")
        except RuntimeError:
            gpu_devices = []
        found_gpu = bool(gpu_devices)
        if gpu_devices:
            logger.debug(f"GPU CUDA - jax: {gpu_devices[0]}")

    if not found_gpu and importlib.util.find_spec("torch") is not None:
        import torch  # NOQA: PLC0415

        found_gpu = torch.cuda.is_available()
        if found_gpu:
            logger.debug(f"GPU CUDA - torch: {torch.cuda.device(torch.cuda.current_device())}")
    logger.debug(f"GPU FOUND STATUS: {found_gpu}")
