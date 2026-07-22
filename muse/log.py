"""
Project-wide logger configuration.
"""

import sys

from loguru import logger

__all__ = ["change_logging_level", "log_gpu_status", "logger"]


def change_logging_level(level: str) -> None:
    """
    Reconfigure the process-global Loguru logger to show ``level`` and above.

    This is an application-level convenience for scripts and notebooks. Loguru
    uses one global logger, so this removes **every** existing Loguru sink
    (including any the host application added) and installs a single
    ``sys.stderr`` sink at ``level``. Applications that manage their own Loguru
    configuration should configure Loguru directly instead of calling this.
    Importing ``muse`` never calls it.

    Parameters
    ----------
    level : str
        The minimum level to show. Must be one of the following:
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    """
    logger.remove()
    logger.add(sys.stderr, level=level)


def log_gpu_status() -> None:
    """
    Log CUDA/GPU availability.

    Optional accelerator backends are imported lazily and this is not called at import
    time, so importing this module (or :mod:`muse`) stays backend-free.
    """
    import importlib.util  # NOQA: PLC0415

    found_gpu = False
    if importlib.util.find_spec("torch") is not None:
        import torch  # NOQA: PLC0415

        found_gpu = torch.cuda.is_available()
        if found_gpu:
            logger.debug(f"GPU CUDA - torch: {torch.cuda.device(torch.cuda.current_device())}")
    logger.debug(f"GPU FOUND STATUS: {found_gpu}")
