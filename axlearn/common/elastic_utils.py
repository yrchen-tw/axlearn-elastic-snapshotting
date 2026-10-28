# Copyright © 2025 Apple Inc.

"""Elastic training utilities for the Pathways backend (no-ops on McJAX)."""

from typing import Optional

from pathwaysutils.elastic import manager

# The elastic manager of the `Snapshotter`, set when it is created on Pathways.
elastic_manager: Optional[manager.Manager] = None


def elastic_enabled() -> bool:
    return elastic_manager is not None


def total_slice_count() -> Optional[int]:
    return None if elastic_manager is None else elastic_manager.total_slice_count
