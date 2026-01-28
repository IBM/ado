# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import re
import typing
from typing import Annotated, TypeVar

import pydantic
from pydantic import BeforeValidator
from pydantic_core import PydanticUseDefault


def default_if_none(value: typing.Any) -> typing.Any:  # noqa: ANN401
    if value is None:
        raise PydanticUseDefault
    return value


T = TypeVar("T")
Defaultable = Annotated[T, BeforeValidator(default_if_none)]


def model_dict_representation_with_field_exclusions_for_custom_model_serializer(
    model: pydantic.BaseModel, info: pydantic.SerializationInfo
) -> dict[str, typing.Any]:

    dict_representation = dict(model)

    # We need to enforce the behaviour for field exclusions
    if info.exclude:
        field_names_to_exclude = (
            set(info.exclude.keys()) if isinstance(info.exclude, dict) else info
        )
        for field_name in field_names_to_exclude:
            dict_representation.pop(field_name, None)

    for field_name, field_info in model.__class__.model_fields.items():

        if field_name not in dict_representation:
            continue

        # Enforce exclude_unset
        if (  # noqa: SIM114
            info.exclude_unset and field_name not in model.model_fields_set
        ):
            del dict_representation[field_name]

        # Enforce exclude_none
        elif (  # noqa: SIM114
            info.exclude_none and dict_representation[field_name] is None
        ):
            del dict_representation[field_name]

        # Enforce exclude_defaults
        elif (
            info.exclude_defaults
            and dict_representation[field_name] == field_info.default
        ):
            del dict_representation[field_name]

    return dict_representation


rfc_1123_pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$"
rfc_1123_regex = re.compile(rfc_1123_pattern)


def validate_rfc_1123(value: str | None) -> str | None:

    if value is None:
        return value

    if len(value) == 0 or len(value) >= 64:
        raise ValueError("The string must be between 1 and 63 characters")

    if not rfc_1123_regex.match(value):
        raise ValueError(
            f"The string does not match RFC1123. Regex: {rfc_1123_pattern}"
        )

    return value
