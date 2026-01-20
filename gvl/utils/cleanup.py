"""Lightweight resource cleanup helpers to run after each script invocation.

Intended to minimize cross-run slowdown in Hydra multi-runs by clearing GPU
caches and triggering garbage collection.
"""

from __future__ import annotations

import gc

from loguru import logger


def cleanup_resources() -> None:
    """Best-effort resource cleanup (CPU/GPU caches)."""
    try:
        import torch

        torch.cuda.empty_cache()
        logger.debug("Cleared torch CUDA cache")
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug(f"torch cleanup skipped: {exc}")
    gc.collect()
    logger.debug("Garbage collection complete")


__all__ = ["cleanup_resources"]
