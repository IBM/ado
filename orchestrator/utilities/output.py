# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing

import pydantic
import yaml


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
