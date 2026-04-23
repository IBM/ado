# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Merge helpers for :class:`orchestrator.core.metadata.ConfigurationMetadata`."""

from __future__ import annotations

import typing


def merge_configuration_metadata_dicts(
    base: dict[str, typing.Any], patch: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    """
    Deep-merge *patch* into a copy of *base* for metadata-shaped dicts.

    ``name`` and ``description`` are overwritten (including with ``null``) when
    present in the patch. ``labels`` are merged: missing base labels are treated
    as an empty dict. Otherwise, dict + dict at the same key is merged; other
    value types are replaced by the patch.
    """
    out: dict[str, typing.Any] = dict(base)
    for key, value in patch.items():
        if key == "labels":
            if value is None:
                out["labels"] = None
            elif isinstance(value, dict):
                left = out.get("labels")
                if left is None or not isinstance(left, dict):
                    left = {}
                out["labels"] = {**left, **value}
            else:
                out["labels"] = value
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out
