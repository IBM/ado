# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from pathlib import Path

import pydantic
import yaml

from ado.cli.utils.output.prints import SUCCESS, console_print, magenta


def serialise_pydantic_model(
    model: pydantic.BaseModel,
    output_path: Path | None,
    suppress_success_message: bool = False,
    context: dict | None = None,
    exclude_none: bool = False,
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
) -> None:
    from ado.utilities.output import pydantic_model_as_yaml

    yaml_content = pydantic_model_as_yaml(
        model,
        exclude_none=exclude_none,
        exclude_unset=exclude_unset,
        exclude_defaults=exclude_defaults,
        context=context,
    )

    if output_path is None:
        # Write to stdout
        console_print(yaml_content)
    else:
        # Write to file
        output_path.write_text(yaml_content)
        if not suppress_success_message:
            console_print(
                f"{SUCCESS}File saved as {magenta(str(output_path))}", stderr=True
            )


def serialise_pydantic_model_json_schema(
    model: pydantic.BaseModel,
    output_path: Path | None,
    suppress_success_message: bool = False,
) -> None:
    schema_content = yaml.safe_dump(model.model_json_schema())

    if output_path is None:
        # Write to stdout
        console_print(schema_content)
    else:
        # Write to file
        output_path.write_text(schema_content)
        if not suppress_success_message:
            console_print(f"Schema saved as {magenta(str(output_path))}", stderr=True)
