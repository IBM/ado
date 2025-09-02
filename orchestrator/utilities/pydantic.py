# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing

import pydantic


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
