# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import math
import typing

import pydantic
import yaml


def format_elided_list(items: list, max_items: int | None) -> str:
    """Format a list, optionally truncating in the middle with an omission count.

    When truncation is needed the first ``ceil(max_items / 2)`` and last
    ``floor(max_items / 2)`` items are shown with the omitted count reported in
    the middle, e.g. ``['A', 'B', ..., 'Y', 'Z'] (+10 more)``.

    Args:
        items: The list of items to format.
        max_items: Maximum number of items to show (head + tail combined).
            Must be at least 2 when provided. Pass ``None`` to disable
            truncation.

    Returns:
        Formatted string representation of the list.

    Raises:
        ValueError: If max_items is less than 2.
    """
    if max_items is not None and max_items < 2:
        raise ValueError("max_items must be at least 2")

    if max_items is None or len(items) <= max_items:
        return "[" + ", ".join(f"{v!r}" for v in items) + "]"

    omitted = len(items) - max_items
    items_at_start = math.ceil(max_items / 2)
    items_at_end = max_items - items_at_start
    parts = (
        [f"{v!r}" for v in items[:items_at_start]]
        + ["..."]
        + [f"{v!r}" for v in items[-items_at_end:]]
    )
    return "[" + ", ".join(parts) + f"] (+{omitted} more)"


def printable_pydantic_model(
    model: pydantic.BaseModel | list[pydantic.BaseModel],
) -> pydantic.BaseModel:
    # We use a RootModel to create on-the-fly a model for a list of the resources of the
    # required type, to mimic the output of kubectl/oc, a list of the resources
    if isinstance(model, list):
        if len(model) > 0:
            PrintablePydanticModel = pydantic.RootModel[list[type(model[0])]]
        else:
            PrintablePydanticModel = pydantic.RootModel[list[pydantic.BaseModel]]
        model = PrintablePydanticModel(model)
    return model


def pydantic_model_as_yaml(
    model: pydantic.BaseModel | list[pydantic.BaseModel],
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    indent: int = 2,
    context: typing.Any | None = None,  # noqa: ANN401
) -> str:

    model = printable_pydantic_model(model)
    return yaml.safe_dump(
        yaml.safe_load(
            model.model_dump_json(
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                indent=indent,
                context=context,
            )
        )
    )
