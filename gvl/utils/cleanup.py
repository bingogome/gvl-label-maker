"""Lightweight resource cleanup helpers to run after each script invocation.

Intended to minimize cross-run slowdown in Hydra multi-runs by clearing GPU
caches and triggering garbage collection.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable
from typing import Any

from loguru import logger


def _iter_objects(objects: Iterable[Any] | Any | None) -> list[Any]:
    if objects is None:
        return []
    if isinstance(objects, (list, tuple, set)):
        return list(objects)
    return [objects]


def _clear_container(container: Any, *, label: str) -> None:
    if container is None:
        return
    clear_fn = getattr(container, "clear", None)
    if callable(clear_fn):
        try:
            clear_fn()
            logger.debug(f"Cleared {label}")
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug(f"Failed to clear {label}: {exc}")


def _cleanup_client(client: Any) -> None:
    if client is None:
        return
    for method_name in ("close", "shutdown", "cleanup", "release"):
        method = getattr(client, method_name, None)
        if callable(method):
            try:
                method()
                logger.debug(f"Called {method_name} on {type(client).__name__}")
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug(f"{type(client).__name__}.{method_name} failed: {exc}")
    for attr in ("model", "processor", "client"):
        if hasattr(client, attr):
            try:
                setattr(client, attr, None)
                logger.debug(f"Cleared {type(client).__name__}.{attr}")
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug(f"Failed to clear {type(client).__name__}.{attr}: {exc}")


def cleanup_resources(
    *,
    clients: Iterable[Any] | Any | None = None,
    models: Iterable[Any] | Any | None = None,
    records: Any | None = None,
) -> None:
    """Best-effort resource cleanup (clients/models/records + CPU/GPU caches)."""
    for client in _iter_objects(clients):
        _cleanup_client(client)
    for model in _iter_objects(models):
        _cleanup_client(model)
    _clear_container(records, label="records")

    try:
        import torch

        torch.cuda.empty_cache()
        logger.debug("Cleared torch CUDA cache")
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug(f"torch cleanup skipped: {exc}")
    gc.collect()
    logger.debug("Garbage collection complete")


__all__ = ["cleanup_resources"]
