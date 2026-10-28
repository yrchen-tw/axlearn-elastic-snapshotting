# Copyright © 2025 Apple Inc.

"""Elastic training utilities for the Pathways backend (no-ops on McJAX)."""

import contextlib
import inspect
import math
from typing import Any, Callable, Optional

import jax
import pathwaysutils
from absl import logging
from pathwaysutils.elastic import elastic, manager

ScaleUpSignalError = manager.ScaleUpSignalError

elastic_manager: Optional[manager.Manager] = None


def _is_host_array(array: jax.Array) -> bool:
    sharding = getattr(array, "sharding", None)
    if getattr(sharding, "memory_kind", None) in ("pinned_host", "unpinned_host"):
        return True
    return any(getattr(d, "platform", None) == "cpu" for d in getattr(sharding, "device_set", ()))


def _elastic_event_cleanup() -> None:
    """Replaces `manager._elastic_event_cleanup`, keeping host-resident arrays alive.

    The released cleanup (pathwaysutils 0.1.11) deletes *every* live array after a slice-down,
    including the `Snapshotter`'s pinned-host snapshot, so the snapshot restore that follows fails
    with "Input array #0 has been donated or deleted". Device arrays are still deleted and JAX
    caches are still cleared, as upstream does.
    """
    try:
        jax.profiler.stop_trace()
    except (RuntimeError, ValueError):
        pass
    jax.clear_caches()
    for array in jax.live_arrays():
        if _is_host_array(array):
            continue
        with contextlib.suppress(Exception):
            array.delete()


def install_elastic_event_cleanup():
    """Makes `manager.Manager.elastic_retry` keep host-resident snapshots. Idempotent."""
    manager._elastic_event_cleanup = _elastic_event_cleanup  # pylint: disable=protected-access


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
    install_elastic_event_cleanup()
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


def available_slice_count() -> Optional[int]:
    """Returns the number of healthy slices, or None when not elastic or if the check fails."""
    if elastic_manager is None:
        return None
    try:
        return len(
            elastic.get_active_slice_indices(slice_to_devices=elastic_manager.slice_to_devices)
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Failed to check the available slices: %s", e)
        return None


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
