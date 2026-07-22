"""
Private array-backend bridge for synthesis.

`vdem_synthesis` is the only production consumer of the Torch conversion helpers and
backend resolution, so they live here rather than in the public ``muse.utils`` surface.
"""

import importlib.util

import numpy as np


def _resolve_backend(cuda_device: int | None = None, backend: str = "numpy") -> str:
    """
    Validate and resolve the array backend, returning ``"numpy"`` or ``"torch"``.

    Torch is opt-in: ``backend`` defaults to ``"numpy"`` (also for `None`), so the array
    library never changes with what happens to be installed in the environment (the
    accelerator path is float32, the NumPy path keeps the input dtype). This is the
    single source of truth for which backends are valid.

    Parameters
    ----------
    cuda_device : `int` or `None`, optional
        CUDA device index for GPU use (requires ``backend="torch"``), or `None`
        for CPU.
    backend : `str`, optional
        ``"numpy"`` (default) or ``"torch"``.

    Raises
    ------
    ValueError
        For an unknown ``backend``, an accelerator backend that is not installed,
        NumPy asked for a CUDA device, or a negative CUDA device. A device index the
        backend cannot serve is reported later, when the array is placed.
    """
    if backend not in ("numpy", "torch"):
        msg = f"Unknown backend {backend!r}; choose 'numpy' or 'torch'"
        raise ValueError(msg)
    if backend == "numpy":
        if cuda_device is not None:
            msg = "The numpy backend does not support cuda_device; use backend='torch'"
            raise ValueError(msg)
        return "numpy"
    if cuda_device is not None and cuda_device < 0:
        msg = f"CUDA device {cuda_device} is not valid"
        raise ValueError(msg)
    if importlib.util.find_spec(backend) is None:
        msg = (
            f"backend={backend!r} requested but Torch is not installed. Install the build "
            "matching your hardware (https://pytorch.org/get-started/locally/, or "
            "`conda install pytorch`), then retry."
        )
        raise ValueError(msg)
    return backend


def torch_to_numpy(torch_tensor):
    """
    Convert a `torch.Tensor` to a `numpy.ndarray`.

    The tensor is detached from the autograd graph and moved to the CPU first
    so tensors on CUDA devices convert cleanly.

    Parameters
    ----------
    torch_tensor : `torch.Tensor`
        Torch tensor to convert.

    Returns
    -------
    `numpy.ndarray`
        The converted NumPy array.
    """
    tensor = torch_tensor.detach().cpu()
    try:
        return tensor.numpy()
    except TypeError:  # dtypes numpy lacks, e.g. bfloat16 / float8
        return np.array(tensor.tolist())


def numpy_to_torch(numpy_array: np.ndarray, cuda_device: int | None = None):
    """
    Convert a `numpy.ndarray` to a `torch.Tensor`.

    Floating-point precision is capped at float32: ``float64`` input is downcast
    to ``float32``, while narrower dtypes (``float16``, integers) are left as-is.

    Parameters
    ----------
    numpy_array : `numpy.ndarray`
        The array to convert.
    cuda_device : `int` or `None`, optional
        If provided, transfer the tensor to the specified CUDA device.

    Returns
    -------
    `torch.Tensor`
        The converted Torch tensor.
    """
    import torch  # NOQA: PLC0415

    tensor = torch.tensor(numpy_array)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    if cuda_device is not None:
        with torch.cuda.device(f"cuda:{cuda_device}"):
            return tensor.cuda()
    return tensor
