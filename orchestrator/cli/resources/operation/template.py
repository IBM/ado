# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib

import typer

from orchestrator.cli.models.parameters import AdoTemplateCommandParameters
from orchestrator.cli.utils.output.prints import (
    ERROR,
    HINT,
    WARN,
    console_print,
    cyan,
)
from orchestrator.cli.utils.pydantic.serializers import (
    serialise_pydantic_model,
    serialise_pydantic_model_json_schema,
)
from orchestrator.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
    OperatorFunctionConf,
)


def template_operation(parameters: AdoTemplateCommandParameters) -> None:
    import orchestrator.modules.operators.collections

    operators = orchestrator.modules.operators.collections.operationCollectionMap
    supported_operator_types = operators.keys()

    # Exit early on wrong configurations
    if parameters.operator_type and not parameters.operator_name:
        console_print(
            f"{ERROR}If you specify the operator type, you must "
            f"also specify the operator name with the --operator-name flag",
            stderr=True,
        )
        raise typer.Exit(1)

    if (
        parameters.operator_type
        and parameters.operator_type not in supported_operator_types
    ):
        console_print(
            f"{ERROR}The only operator types supported are "
            f"{[t.value for t in list(supported_operator_types)]}",
            stderr=True,
        )
        raise typer.Exit(1)

    # Exit early on generic template
    if not parameters.operator_name:
        model_instance = DiscoveryOperationResourceConfiguration(
            spaces=["your-spaces"],
            operation=DiscoveryOperationConfiguration(),
        )

        serialise_pydantic_model(
            model=model_instance,
            output_path=parameters.output_path,
        )

        if parameters.include_schema:
            schema_output_path = pathlib.Path(
                parameters.output_path.stem + "_schema.yaml"
            )
            serialise_pydantic_model_json_schema(model_instance, schema_output_path)
        return

    # The user has requested a specific operator for the template
    # and has provided us with everything we need
    if parameters.operator_name and parameters.operator_type:

        if not operator_type_has_operator(
            parameters.operator_name, parameters.operator_type
        ):
            console_print(
                f"{ERROR}Operator {parameters.operator_name} does not exist for type {parameters.operator_type.value}\n"
                f"{HINT}Try running {cyan('ado get operators')} to see what operators are available.\n\t"
                f"Or run {cyan(f'ado template operation --operator-name {parameters.operator_name}')} "
                "to autodetect the operator type.",
                stderr=True,
            )
            raise typer.Exit(1)

    # The user has requested a specific operator for the template
    # but didn't provide the operator type, just the name
    # we are sure of this because of the earlier validation
    else:

        parameters.operator_type = find_operator_type_by_name(parameters.operator_name)
        if not parameters.operator_type:
            console_print(
                f"{ERROR}Unable to find operator {parameters.operator_name}.\n"
                f"{HINT}Try running {cyan('ado get operators')} to see what operators are available.\n\t"
                f"Or run {cyan('ado template operation')} to get a generic operation template.",
                stderr=True,
            )
            raise typer.Exit(1)

    # We are now sure that the operator_type is supported and
    # has a value
    default_operation_parameters = operators[
        parameters.operator_type
    ].default_configuration_model_for_operation(parameters.operator_name)

    # Certain operators may not have a default configuration model
    # Use an OperatorFunctionConf and set the values we have
    if not default_operation_parameters:

        console_print(
            f"{WARN}Operator {parameters.operator_name} does not have a specific template. "
            "A generic template will be output.",
            stderr=True,
        )
        default_operation_parameters = {}

    default_operation_configuration = DiscoveryOperationConfiguration(
        module=OperatorFunctionConf(
            operatorName=parameters.operator_name,
            operationType=parameters.operator_type,
        ),
        parameters=default_operation_parameters,
    )

    model_instance = DiscoveryOperationResourceConfiguration(
        spaces=["your-spaces"],
        operation=default_operation_configuration,
    )

    # It's more helpful if the file name contains the operator name
    output_path = (
        pathlib.Path(parameters.output_path.stem + f"_{parameters.operator_name}.yaml")
        if not parameters.output_path
        else parameters.output_path
    )

    orchestrator.cli.utils.pydantic.serializers.serialise_pydantic_model(
        model=model_instance,
        output_path=output_path,
    )

    if parameters.include_schema:
        schema_output_path = pathlib.Path(output_path.stem + "_schema.yaml")
        schema_model_instance = (
            default_operation_configuration.parameters
            if parameters.parameters_only_schema
            else model_instance
        )
        orchestrator.cli.utils.pydantic.serializers.serialise_pydantic_model_json_schema(
            schema_model_instance, schema_output_path
        )


def find_operator_type_by_name(
    operator_name: str,
) -> DiscoveryOperationEnum | None:
    import orchestrator.modules.operators.collections

    supported_operator_types = (
        orchestrator.modules.operators.collections.operationCollectionMap.keys()
    )

    for operator_type in supported_operator_types:
        if operator_type_has_operator(operator_name, operator_type):
            return operator_type

    return None


def operator_type_has_operator(
    operator_name: str,
    operator_type: DiscoveryOperationEnum,
) -> bool:
    import orchestrator.modules.operators.collections

    return (
        operator_name
        in orchestrator.modules.operators.collections.operationCollectionMap[
            operator_type
        ].function_operations
    )
