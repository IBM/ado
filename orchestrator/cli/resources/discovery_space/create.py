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
    cyan,
    magenta,
)
from orchestrator.cli.utils.pydantic.updaters import override_values_in_pydantic_model
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.metastore.base import ResourceDoesNotExistError


def create_discovery_space(parameters: AdoCreateCommandParameters):

    try:
        space_configuration = DiscoverySpaceConfiguration.model_validate(
            yaml.safe_load(parameters.resource_configuration_file.read_text())
        )
    except pydantic.ValidationError as error:
        console_print(
            f"{ERROR}The space configuration provided was not valid:\n{error}",
            stderr=True,
        )
        raise typer.Exit(1) from error

    if parameters.override_values:
        space_configuration = override_values_in_pydantic_model(
            model=space_configuration, override_values=parameters.override_values
        )

    if parameters.new_sample_store:

        # Replay experiments cannot use --new-sample-store
        # We want to check whether the replay actuator is being used
        # in the space.
        actuator_ids = {
            e.actuatorIdentifier
            for e in space_configuration.convert_experiments_to_reference_list().experiments
        }

        if "replay" in actuator_ids:
            console_print(
                f"{ERROR}You cannot use {cyan('--new-sample-store')} with a space that uses the replay actuator.\n"
                f"{HINT}Provide a sampleStoreIdentifier in the space configuration.",
                stderr=True,
            )
            raise typer.Exit(1)

        info_message = (
            f"{INFO}A new sample store was requested.\n"
            f"\tSample store {cyan(space_configuration.sampleStoreIdentifier)} referenced in the space definition "
            "will be ignored."
        )

        console_print(
            info_message,
            stderr=True,
        )

        from orchestrator.core.samplestore.config import (
            SampleStoreConfiguration,
            SampleStoreModuleConf,
            SampleStoreSpecification,
        )
        from orchestrator.core.samplestore.utils import create_sample_store_resource

        sample_store_configuration = SampleStoreConfiguration(
            specification=SampleStoreSpecification(
                module=SampleStoreModuleConf(
                    moduleClass="SQLSampleStore",
                    moduleName="orchestrator.core.samplestore.sql",
                ),
                storageLocation=parameters.ado_configuration.project_context.metadataStore,
            )
        )
        sql_store = get_sql_store(
            project_context=parameters.ado_configuration.project_context
        )
        with Status("Creating your sample store"):
            sample_store_resource, _ = create_sample_store_resource(
                conf=sample_store_configuration, resourceStore=sql_store
            )

        space_configuration.sampleStoreIdentifier = sample_store_resource.identifier

    if parameters.dry_run:
        console_print(ADO_CREATE_DRY_RUN_CONFIG_VALID, stderr=True)
        return

    with Status("Initializing DiscoverySpace") as status:
        try:
            space = DiscoverySpace.from_configuration(
                space_configuration,
                project_context=parameters.ado_configuration.project_context,
                identifier=None,
            )
        except ResourceDoesNotExistError:
            raise
        except Exception as error:
            status.stop()
            console_print(
                f"{ERROR}An exception occurred when creating the discovery space: {error}",
                stderr=True,
            )
            raise

        status.update(ADO_SPINNER_SAVING_TO_DB)
        space.saveSpace()

    console_print(
        f"{SUCCESS}Created space with identifier: {magenta(space.uri)}", stderr=True
    )
