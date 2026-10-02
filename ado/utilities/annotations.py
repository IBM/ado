# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Annotated, get_args, get_origin


def _unwrap_annotated(hint: object) -> object:
    """Return the underlying type if *hint* is ``Annotated[T, ...]``, else *hint*."""
    if get_origin(hint) is Annotated:
        args = get_args(hint)
        return args[0] if args else hint
    return hint
