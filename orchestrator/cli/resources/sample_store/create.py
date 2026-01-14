# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pydantic
import typer
import yaml
from rich.status import Status

from orchestrator.cli.models.parameters import AdoCreateCommandParameters
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.prints import (
    ADO_CREATE_DRY_RUN_CONFIG_VALID,
    ADO_SPINNER_SAVING_TO_DB,
    ERROR,
    HINT,
    INFO,
    SUCCESS,
    console_print,
    magenta,
)
from orchestrator.cli.utils.pydantic.updaters import override_values_in_pydantic_model
from orchestrator.core import CoreResourceKinds
from orchestrator.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreModuleConf,
    SampleStoreSpecification,
)


def create_sample_store(parameters: AdoCreateCommandParameters) -> str:

    if parameters.new_sample_store:
        console_print(f"{INFO}A new SQLSampleStore was requested.")
        sample_store_configuration = SampleStoreConfiguration(
            specification=SampleStoreSpecification(
                module=SampleStoreModuleConf(
                    moduleClass="SQLSampleStore",
                    moduleName="orchestrator.core.samplestore.sql",
                ),
            )
        )

    else:
        try:
            sample_store_configuration = SampleStoreConfiguration.model_validate(
                yaml.safe_load(parameters.resource_configuration_file.read_text())
            )
        except pydantic.ValidationError as error:
            console_print(
                f"{ERROR}The sample store configuration provided was not valid:\n{error}",
                stderr=True,
            )
            raise typer.Exit(1) from error

    if parameters.override_values:
        sample_store_configuration = override_values_in_pydantic_model(
            model=sample_store_configuration, override_values=parameters.override_values
        )

    if sample_store_configuration.specification.module.moduleClass == "SQLSampleStore":
        if sample_store_configuration.specification.storageLocation:
            console_print(
                f"{ERROR}The storageLocation section must be empty when creating sample stores. "
                "It will be set automatically based on the context information.\n"
                f"{HINT}Remove the storageLocation field and retry.",
                stderr=True,
            )
            raise typer.Exit(1)

        sample_store_configuration.specification.storageLocation = (
            parameters.ado_configuration.project_context.metadataStore
        )

    if parameters.dry_run:
        console_print(ADO_CREATE_DRY_RUN_CONFIG_VALID, stderr=True)
        return None

    from orchestrator.core.samplestore.utils import create_sample_store_resource

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)
    with Status(ADO_SPINNER_SAVING_TO_DB):
        _, sample_store = create_sample_store_resource(
            sample_store_configuration,
            sql,
        )

    # Save the identifier of the resource we created
    # for reuse
    parameters.ado_configuration.latest_resource_ids[CoreResourceKinds.SAMPLESTORE] = (
        sample_store.identifier
    )

    console_print(
        f"{SUCCESS}Created sample store with identifier {magenta(sample_store.identifier)}",
        stderr=True,
    )

    return sample_store.identifier
