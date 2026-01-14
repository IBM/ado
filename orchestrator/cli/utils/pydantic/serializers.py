# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from pathlib import Path

import pydantic
import yaml

from orchestrator.cli.utils.output.prints import SUCCESS, console_print, magenta


def serialise_pydantic_model(
    model: pydantic.BaseModel,
    output_path: Path,
    suppress_success_message: bool = False,
) -> None:
    from orchestrator.utilities.output import pydantic_model_as_yaml

    output_path.write_text(pydantic_model_as_yaml(model))
    if not suppress_success_message:
        console_print(
            f"{SUCCESS}File saved as {magenta(str(output_path))}", stderr=True
        )


def serialise_pydantic_model_json_schema(
    model: pydantic.BaseModel,
    output_path: Path,
    suppress_success_message: bool = False,
) -> None:
    output_path.write_text(yaml.safe_dump(model.model_json_schema()))
    if not suppress_success_message:
        console_print(f"Schema saved as {magenta(str(output_path))}", stderr=True)
