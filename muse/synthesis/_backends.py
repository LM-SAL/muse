"""
Private array-backend bridge for synthesis.

`vdem_synthesis` is the only production consumer of the Torch conversion helpers, the
device resolution and the memory-budgeted product, so they live here rather than in the
public ``muse.utils`` surface.
"""

import importlib.util
from typing import TYPE_CHECKING
from collections.abc import Sequence

import numpy as np

from muse.log import logger

if TYPE_CHECKING:
    import torch


def _resolve_backend(device: "str | torch.device | None" = None, backend: str = "numpy") -> str:
    """
    Validate and resolve the array backend, returning ``"numpy"`` or ``"torch"``.

    Torch is opt-in: ``backend`` defaults to ``"numpy"`` (also for `None`), so the array
    library never changes with what happens to be installed in the environment (the
    accelerator path is float32, the NumPy path keeps the input dtype). This is the
    single source of truth for which backends are valid.

    Parameters
    ----------
    device : `str`, `torch.device` or `None`, optional
        Torch device of the product (requires ``backend="torch"``), or `None` for the CPU.
    backend : `str`, optional
        ``"numpy"`` (default) or ``"torch"``.

    Raises
    ------
    ValueError
        For an unknown ``backend``, an accelerator backend that is not installed, or
        NumPy asked for a device. A device Torch cannot serve is reported later, when
        the arrays are placed.
    """
    if backend not in ("numpy", "torch"):
        msg = f"Unknown backend {backend!r}; choose 'numpy' or 'torch'"
        raise ValueError(msg)
    if backend == "numpy":
        if device is not None:
            msg = "The numpy backend does not support device; use backend='torch'"
            raise ValueError(msg)
        return "numpy"
    if importlib.util.find_spec(backend) is None:
        msg = (
            f"backend={backend!r} requested but Torch is not installed. Install the build "
            "matching your hardware (https://pytorch.org/get-started/locally/, or "
            "`conda install pytorch`), then retry."
        )
        raise ValueError(msg)
    return backend


def _resolve_device(device: "str | torch.device | None" = None) -> "torch.device":
    """
    The `torch.device` for ``device``.

    Parameters
    ----------
    device : `str`, `torch.device` or `None`, optional
        A device name such as ``"cpu"``, ``"cuda:0"`` or ``"mps"``; `None` for the CPU;
        or ``"auto"`` for the first of CUDA, MPS and the CPU that is available.

    Returns
    -------
    `torch.device`
        The resolved device.
    """
    import torch  # NOQA: PLC0415

    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cpu")
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _empty_device_cache(device: "torch.device") -> None:
    """
    Release the blocks the allocator of ``device`` caches, which count as allocated.

    Parameters
    ----------
    device : `torch.device`
        The device whose allocator cache to empty; the CPU has none.
    """
    import torch  # NOQA: PLC0415

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _available_memory_bytes(device: "torch.device") -> int:
    """
    Bytes that ``device`` may still allocate.

    Parameters
    ----------
    device : `torch.device`
        The device to measure.

    Returns
    -------
    `int`
        The free device memory on CUDA, the smaller of the host's available memory and
        what MPS may still allocate, or the host's available memory on the CPU.
    """
    import psutil  # NOQA: PLC0415
    import torch  # NOQA: PLC0415

    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return int(free)
    host_available = int(psutil.virtual_memory().available)
    if device.type == "mps":
        recommended = int(torch.mps.recommended_max_memory())
        allocated = int(torch.mps.driver_allocated_memory())
        return max(1, min(host_available, recommended - allocated))
    return host_available


def torch_to_numpy(torch_tensor: "torch.Tensor") -> np.ndarray:
    """
    Convert a `torch.Tensor` to a `numpy.ndarray`.

    The tensor is detached from the autograd graph and moved to the CPU first
    so tensors on accelerator devices convert cleanly.

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


def numpy_to_torch(numpy_array: np.ndarray, device: "str | torch.device | None" = None) -> "torch.Tensor":
    """
    Convert a `numpy.ndarray` to a `torch.Tensor`.

    Floating-point precision is capped at float32: ``float64`` input is downcast
    to ``float32``, while narrower dtypes (``float16``, integers) are left as-is.

    Parameters
    ----------
    numpy_array : `numpy.ndarray`
        The array to convert.
    device : `str`, `torch.device` or `None`, optional
        If provided, place the tensor on this device, as for `_resolve_device`.

    Returns
    -------
    `torch.Tensor`
        The converted Torch tensor.
    """
    import torch  # NOQA: PLC0415

    tensor = torch.tensor(numpy_array)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    if device is not None:
        return tensor.to(_resolve_device(device))
    return tensor


def torch_einsum(
    vdem: np.ndarray,
    detector_response: np.ndarray,
    vdem_indices: Sequence[int],
    response_indices: Sequence[int],
    output_indices: Sequence[int],
    device: "str | torch.device | None" = None,
) -> np.ndarray:
    """
    Contract ``vdem`` with ``detector_response`` on ``device`` in float32, in chunks
    that fit.

    The product goes in chunks along the first ``vdem`` axis kept in the output and not
    shared with the response, each within half of what the device may still allocate and
    half of the free host memory: a product too large for MPS is copied back to the host
    as zeros, with no exception and at most an out-of-memory report from Metal on stderr.
    On the device a chunk costs its vdem and output plus einsum's permuted copies of both,
    next to the response and its copy; on the host, the chunk copied back.

    Parameters
    ----------
    vdem, detector_response : `numpy.ndarray`
        The operands; float64 is downcast to float32.
    vdem_indices, response_indices, output_indices : `Sequence` of `int`
        Sublist-format subscripts, as for `numpy.einsum`.
    device : `str`, `torch.device` or `None`, optional
        Device of the product, as for `_resolve_device`; by default the CPU.

    Returns
    -------
    `numpy.ndarray`
        The float32 product, on the host.

    Raises
    ------
    MemoryError
        If one row of the product does not fit next to the response on an accelerator.
    RuntimeError
        If MPS returned an all-zero product, which it does when memory runs out.
    """
    import torch  # NOQA: PLC0415

    target = _resolve_device(device)
    vdem_indices, response_indices, output_indices = list(vdem_indices), list(response_indices), list(output_indices)
    logger.debug(f"Synthesis einsum on {target}")
    # One float32 copy each, made in NumPy: MPS has no float64, and a float64 tensor cast
    # afterwards would hold the VDEM three times over on the host.
    vdem_tensor = torch.from_numpy(np.array(vdem, dtype=np.float32))
    response_tensor = torch.from_numpy(np.array(detector_response, dtype=np.float32))
    sizes = dict(zip((*vdem_indices, *response_indices), (*vdem_tensor.shape, *response_tensor.shape), strict=True))
    out = np.empty([sizes[index] for index in output_indices], dtype=np.float32)
    # Not along an axis the response shares (slit): only the VDEM operand is sliced.
    axis = next(
        (i for i, index in enumerate(vdem_indices) if index in output_indices and index not in response_indices), None
    )
    n_rows = 1 if axis is None else vdem_tensor.shape[axis]
    device_per_row = 8 * (vdem_tensor.numel() + out.size) // n_rows
    host_per_row = 4 * out.size // n_rows
    # The allocator's cached blocks count as allocated: a solve just before would leave no budget.
    _empty_device_cache(target)
    free_device = _available_memory_bytes(target)
    free_host = _available_memory_bytes(torch.device("cpu"))
    rows = int(min((free_device // 2 - 8 * response_tensor.numel()) // device_per_row, free_host // 2 // host_per_row))
    if rows < 1 and target.type == "cpu":
        rows = 1  # the host pages to disk rather than failing
    if rows < 1:
        msg = (
            f"One row of the synthesis ({device_per_row / 1e9:.2f} GB on {target}) does not fit next to "
            f"the response ({8 * response_tensor.numel() / 1e9:.1f} GB)"
        )
        raise MemoryError(msg)
    response_tensor = response_tensor.to(target)
    for start in range(0, n_rows, rows):
        chunk = vdem_tensor if axis is None else vdem_tensor.narrow(axis, start, min(rows, n_rows - start))
        product = torch_to_numpy(
            torch.einsum(chunk.to(target), vdem_indices, response_tensor, response_indices, output_indices)
        )
        if target.type == "mps" and not product.any() and chunk.any() and response_tensor.any():
            # Oversubscribed unified memory: the product is right on the GPU, its copy to the host is zeros.
            msg = "The synthesis copied back from MPS is all zero, which happens when memory runs out; use device='cpu'"
            raise RuntimeError(msg)
        index = [slice(None)] * out.ndim
        if axis is not None:
            index[output_indices.index(vdem_indices[axis])] = slice(start, start + chunk.shape[axis])
        out[tuple(index)] = product
    return out
