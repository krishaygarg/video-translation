"""
nccl_transfer.py
================
Non-blocking, async CUDA-stream-based inter-GPU tensor transfer utilities.

Problem solved: a naive `.to("cuda:N")` call is a *blocking* PCIe memory
copy that stalls the sending GPU until the transfer finishes.

Solution: use a dedicated per-target CUDA Stream so the tensor copy is
enqueued asynchronously.  The sending GPU is released immediately; the
receiving GPU waits only when it actually needs the tensor.
"""

import logging
from typing import Dict, List, Optional

import torch

log = logging.getLogger(__name__)

# Module-level cache of per-device CUDA streams reused across transfers
_streams: Dict[int, torch.cuda.Stream] = {}


def _get_stream(device_id: int) -> torch.cuda.Stream:
    """Return (or lazily create) a dedicated transfer stream for device_id."""
    if device_id not in _streams:
        _streams[device_id] = torch.cuda.Stream(device=f"cuda:{device_id}")
    return _streams[device_id]


def async_send(
    tensor: torch.Tensor,
    target_device_id: int,
    non_blocking: bool = True,
) -> torch.Tensor:
    """
    Copy a tensor to a target GPU device asynchronously.

    The copy is enqueued on a dedicated per-target CUDA stream so the
    source GPU thread is not blocked waiting for PCIe transfer completion.

    Parameters
    ----------
    tensor : torch.Tensor
        Source tensor (may reside on CPU or any GPU).
    target_device_id : int
        Target CUDA device index.
    non_blocking : bool
        If True (default), the copy is non-blocking from the CPU's perspective.

    Returns
    -------
    torch.Tensor
        A tensor on the target device whose data will be valid once the
        transfer stream has finished.  Callers should call
        ``torch.cuda.current_stream(target).wait_stream(transfer_stream)``
        before consuming the tensor in a compute kernel.
    """
    target = torch.device(f"cuda:{target_device_id}")
    stream = _get_stream(target_device_id)

    with torch.cuda.stream(stream):
        target_tensor = tensor.to(device=target, non_blocking=non_blocking)

    log.debug(
        "[nccl_transfer] Enqueued async transfer %s → cuda:%d (shape=%s)",
        tensor.device, target_device_id, tuple(tensor.shape),
    )
    return target_tensor


def broadcast_to(
    tensor: torch.Tensor,
    target_device_ids: List[int],
) -> Dict[int, torch.Tensor]:
    """
    Broadcast a single tensor to multiple GPU devices asynchronously.

    Each copy is enqueued on an independent stream so all transfers proceed
    in parallel over the PCIe bus.

    Parameters
    ----------
    tensor : torch.Tensor
        Source tensor.
    target_device_ids : list[int]
        List of CUDA device indices to copy to.

    Returns
    -------
    dict[int, torch.Tensor]
        Mapping from device_id to the corresponding tensor replica.
    """
    return {dev: async_send(tensor, dev) for dev in target_device_ids}


def synchronize_stream(device_id: int) -> None:
    """
    Block the calling thread until all async transfers to device_id complete.

    Call this immediately before launching a compute kernel that consumes a
    tensor received via :func:`async_send` or :func:`broadcast_to`.
    """
    stream = _streams.get(device_id)
    if stream is not None:
        stream.synchronize()
        log.debug("[nccl_transfer] Transfer stream synced for cuda:%d", device_id)
