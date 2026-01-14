# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import contextlib
import json

import pydantic
import typer

from orchestrator.cli.utils.output.prints import ERROR, console_print, cyan


def override_values_in_pydantic_model(
    model: pydantic.BaseModel, override_values: list[dict[str, str]]
) -> pydantic.BaseModel:
    import jsonpath_ng.ext
    from jsonpath_ng.exceptions import JsonPathLexerError, JsonPathParserError

    model_dict = model.model_dump()
    for override_dict in override_values:
        key = next(iter(override_dict.keys()))
        value = next(iter(override_dict.values()))

        # Ensure we have a valid JSONPath
        try:
            path = jsonpath_ng.ext.parse(key)
        except (JsonPathParserError, JsonPathLexerError) as e:
            console_print(
                f"{ERROR}The provided path {cyan(key)} was not a valid JSONPath string:\n{e}",
                stderr=True,
            )
            raise typer.Exit(1) from e

        # We can either have a JSON document or a string
        with contextlib.suppress(ValueError):
            value = json.loads(value)

        # AP 12/06/2025:
        # path.update will not raise any errors or anything when the field
        # is not in the model. We need to check manually.
        matches = path.find(model_dict)
        if not matches:
            console_print(
                f"{ERROR}The provided path {cyan(key)} was not found in the provided configuration. "
                "Check your input.",
                stderr=True,
            )
            raise typer.Exit(1)

        # Update the dictionary - this might contain invalid values or fields
        model_dict = path.update(model_dict, value)

    # Return the model only if it's still valid
    try:
        return model.model_validate(model_dict)
    except pydantic.ValidationError as error:
        console_print(
            f"{ERROR}The model obtained by overriding the fields provided is invalid.\n{error}",
            stderr=True,
        )
        raise typer.Exit(1) from error
