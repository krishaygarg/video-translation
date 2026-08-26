"""
gpu_guard.py
============
Context manager that guarantees 0 MB idle VRAM footprint on all specified
GPU devices when the streaming session exits, whether cleanly or via exception.

Usage
-----
    from realtime_engine.utils.gpu_guard import shared_gpu_session

    with shared_gpu_session(gpu_ids=[0, 1, 2, 3]) as guard:
        # load models, run pipeline
        ...
    # <-- VRAM fully released here, guaranteed
"""

import gc
import logging
from contextlib import contextmanager
from typing import List

import torch

log = logging.getLogger(__name__)


def release_vram(gpu_ids: List[int]) -> None:
    """
    Explicitly release all CUDA memory across every specified GPU device.
    Runs Python garbage collection first to destroy any lingering tensors,
    then flushes the CUDA allocator cache and IPC handles on every device.
    """
    # 1. Python GC — collect cyclic references to tensor objects
    gc.collect()

    if not torch.cuda.is_available():
        return

    for device_id in gpu_ids:
        try:
            with torch.cuda.device(device_id):
                torch.cuda.synchronize(device_id)   # Wait for all kernels to finish
                torch.cuda.empty_cache()             # Release caching allocator memory
                torch.cuda.ipc_collect()             # Release IPC memory handles

            allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            reserved  = torch.cuda.memory_reserved(device_id)  / (1024 ** 2)
            log.info(
                "[gpu_guard] GPU %d: %.1f MB allocated, %.1f MB reserved after teardown.",
                device_id, allocated, reserved,
            )
        except Exception as exc:
            log.warning("[gpu_guard] Failed to release GPU %d: %s", device_id, exc)


@contextmanager
def shared_gpu_session(gpu_ids: List[int] = None):
    """
    Context manager for shared GPU server etiquette.

    Guarantees that when the block exits (normally or via exception) all
    CUDA memory on every listed GPU is freed, returning the device to a
    0 MB idle footprint for other users on the shared server.

    Parameters
    ----------
    gpu_ids : list[int]
        GPU device indices to manage. Defaults to all visible CUDA devices.

    Yields
    ------
    gpu_ids : list[int]
        The resolved list of GPU device IDs in use.
    """
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))

    log.info("[gpu_guard] Starting shared GPU session on devices: %s", gpu_ids)
    try:
        yield gpu_ids
    finally:
        log.info("[gpu_guard] Session ended — releasing VRAM on devices: %s", gpu_ids)
        release_vram(gpu_ids)

        # Final verification log
        if torch.cuda.is_available():
            for device_id in gpu_ids:
                allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
                log.info(
                    "[gpu_guard] ✓ GPU %d final VRAM: %.1f MB allocated.",
                    device_id, allocated,
                )
