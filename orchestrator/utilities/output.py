# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing

import pydantic
import yaml


def printable_pydantic_model(
    model: typing.Union[pydantic.BaseModel, list[pydantic.BaseModel]],
) -> pydantic.BaseModel:
    # We use a RootModel to create on-the-fly a model for a list of the resources of the
    # required type, to mimic the output of kubectl/oc, a list of the resources
    if isinstance(model, typing.List):
        if len(model) > 0:
            PrintablePydanticModel = pydantic.RootModel[typing.List[type(model[0])]]
        else:
            PrintablePydanticModel = pydantic.RootModel[typing.List[pydantic.BaseModel]]
        model = PrintablePydanticModel(model)
    return model


def pydantic_model_as_yaml(
    model: typing.Union[pydantic.BaseModel, typing.List[pydantic.BaseModel]],
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    indent: int = 2,
    context: typing.Optional[typing.Any] = None,
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
