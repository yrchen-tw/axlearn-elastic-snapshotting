# Copyright © 2025 Apple Inc.

"""Elastic training utilities for the Pathways backend (no-ops on McJAX)."""

import contextlib
from typing import Optional

import jax
from absl import logging
from pathwaysutils.elastic import elastic, manager

# The elastic manager of the `Snapshotter`, set when it is created on Pathways.
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
