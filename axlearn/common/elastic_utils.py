# Copyright © 2025 Apple Inc.

"""Elastic training utilities for the Pathways backend (no-ops on McJAX)."""

import inspect
import math
from typing import Any, Callable, Optional

import jax
import pathwaysutils
from absl import logging
from pathwaysutils.elastic import manager

ScaleUpSignalError = manager.ScaleUpSignalError

elastic_manager: Optional[manager.Manager] = None


def elastic_enabled() -> bool:
    return elastic_manager is not None


def ensure_elastic_manager_initialized(enabled: bool) -> Optional[manager.Manager]:
    global elastic_manager  # pylint: disable=global-statement
    if enabled and elastic_manager is None:
        if not pathwaysutils.is_pathways_backend_used():
            logging.warning("Elastic training requires the Pathways backend; running without it.")
        else:
            elastic_manager = _create_manager(poll_interval=5)
    return elastic_manager


def _create_manager(*, poll_interval: float) -> manager.Manager:
    """Creates a `Manager` that has waited for all slices.

    Released pathwaysutils (pinned to 0.1.11; latest 0.1.13) has no
    `Manager(wait_for_slices=..., poll_interval=...)`, so wait explicitly there.
    """
    if "wait_for_slices" in inspect.signature(manager.Manager).parameters:
        return manager.Manager(wait_for_slices=True, poll_interval=poll_interval)
    mgr = manager.Manager()
    mgr.active_slice_indices = elastic.wait_for_slices(
        slice_count=mgr.total_slice_count,
        slice_to_devices=mgr.slice_to_devices,
        poll_interval=poll_interval,
    )
    return mgr


def live_devices() -> list[jax.Device]:
    if elastic_manager is None:
        return jax.devices()
    # `Manager.active_devices` is not in released pathwaysutils.
    active = getattr(elastic_manager, "active_devices", None)
    if active is not None:
        return list(active)
    return [
        d
        for i in sorted(elastic_manager.active_slice_indices)
        for d in elastic_manager.slice_to_devices[i]
    ]


def total_slice_count() -> Optional[int]:
    return None if elastic_manager is None else elastic_manager.total_slice_count


def grad_accumulation_steps(
    base_steps: int, *, device_batch_sizes: Optional[tuple[int, int]] = None
) -> int:
    if elastic_manager is None:
        return base_steps
    full, live = device_batch_sizes or (
        elastic_manager.active_slice_count * max(1, base_steps),
        elastic_manager.total_slice_count * max(1, base_steps),
    )
    steps = math.ceil(live / max(1, full // max(1, base_steps)))
    while live % steps:
        steps += 1
    return base_steps if steps <= max(1, base_steps) else steps


def is_scale_up_event() -> bool:
    return elastic_manager is not None and bool(elastic_manager.available_inactive_slices)


def elastic_retry(
    *,
    min_slice_count: int = -1,
    timeout_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
    poll_interval: float = 5,
    pre_callback: Optional[Callable[[], Any]] = None,
    on_elastic_event_callback: Optional[Callable[[], Any]] = None,
) -> Callable[[Callable], Callable]:
    if elastic_manager is None:
        return lambda fn: fn
    return elastic_manager.elastic_retry(
        max_retries=max_retries,
        timeout=timeout_seconds,
        minimum_slice_count=None if min_slice_count == -1 else min_slice_count,
        poll_interval=poll_interval,
        pre_callback=pre_callback,
        on_elastic_event_callback=on_elastic_event_callback,
    )
