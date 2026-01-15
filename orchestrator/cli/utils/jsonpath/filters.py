# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typer

from orchestrator.cli.utils.output.prints import ERROR, console_print, cyan


def remove_fields_from_dictionary(
    input_dictionary: dict, fields_to_remove: list[str]
) -> dict:
    import jsonpath_ng.ext
    from jsonpath_ng.exceptions import JsonPathLexerError, JsonPathParserError

    for field_to_remove in fields_to_remove:

        # Ensure we have a valid JSONPath
        try:
            path = jsonpath_ng.ext.parse(field_to_remove)
        except (JsonPathParserError, JsonPathLexerError) as e:
            console_print(
                f"{ERROR}The provided path {cyan(field_to_remove)} was not a valid JSONPath string:\n{e}",
                stderr=True,
            )
            raise typer.Exit(1) from e

        # AP 28/07/2025:
        # path.filter will not raise any errors or anything when the field
        # is not in the model. We need to check manually.
        matches = path.find(input_dictionary)
        if not matches:
            console_print(
                f"{ERROR}The provided path {cyan(field_to_remove)} was not found in the provided configuration. "
                "Check your input.",
                stderr=True,
            )
            raise typer.Exit(1)

        # Remove fields from the dictionary
        input_dictionary = path.filter(lambda d: True, input_dictionary)

    return input_dictionary
