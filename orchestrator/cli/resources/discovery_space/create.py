# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pydantic
import typer
import yaml
from rich.status import Status

from orchestrator.cli.models.parameters import AdoCreateCommandParameters
from orchestrator.cli.resources.sample_store.create import create_sample_store
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
    latest_identifier_for_resource_not_found,
    magenta,
    value_in_configuration_replaced_with_latest_identifier_for_resource,
)
from orchestrator.cli.utils.pydantic.updaters import override_values_in_pydantic_model
from orchestrator.core import CoreResourceKinds
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.metastore.base import ResourceDoesNotExistError


def create_discovery_space(parameters: AdoCreateCommandParameters) -> str | None:

    # Fail early if there is an invalid combination of parameters
    mutually_exclusive_options = [
        parameters.new_sample_store,
        parameters.use_latest is not None
        and len(parameters.use_latest) >= 0
        and CoreResourceKinds.SAMPLESTORE in parameters.use_latest,
        parameters.with_resources is not None
        and CoreResourceKinds.SAMPLESTORE in parameters.with_resources,
    ]

    if sum(mutually_exclusive_options) >= 2:
        console_print(
            f"{ERROR}You can only set one of --new-sample-store, "
            f"--use-latest {CoreResourceKinds.SAMPLESTORE.value}, "
            f"--with {CoreResourceKinds.SAMPLESTORE.value}=path/id",
            stderr=True,
        )
        raise typer.Exit(1)

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

    if (
        parameters.with_resources
        and CoreResourceKinds.SAMPLESTORE in parameters.with_resources
    ):

        if isinstance(parameters.with_resources[CoreResourceKinds.SAMPLESTORE], str):
            space_configuration.sampleStoreIdentifier = parameters.with_resources[
                CoreResourceKinds.SAMPLESTORE
            ]
        else:
            from orchestrator.cli.models.types import AdoCreateSupportedResourceTypes

            space_configuration.sampleStoreIdentifier = create_sample_store(
                AdoCreateCommandParameters(
                    ado_configuration=parameters.ado_configuration,
                    dry_run=False,
                    new_sample_store=False,
                    override_values=[],
                    resource_configuration_file=parameters.with_resources[
                        CoreResourceKinds.SAMPLESTORE
                    ],
                    use_default_sample_store=False,
                    use_latest=[],
                    with_resources={},
                    resource_type=AdoCreateSupportedResourceTypes.SAMPLE_STORE,
                )
            )

    elif (
        parameters.use_latest and CoreResourceKinds.SAMPLESTORE in parameters.use_latest
    ):

        latest_recorded_sample_store = (
            parameters.ado_configuration.latest_resource_ids.get(
                CoreResourceKinds.SAMPLESTORE
            )
        )
        if not latest_recorded_sample_store:
            console_print(
                latest_identifier_for_resource_not_found(CoreResourceKinds.SAMPLESTORE),
                stderr=True,
            )
            raise typer.Exit(1)

        console_print(
            value_in_configuration_replaced_with_latest_identifier_for_resource(
                reused_resource_kind=CoreResourceKinds.SAMPLESTORE,
                target_resource_kind=CoreResourceKinds.DISCOVERYSPACE,
                replacement_identifier=latest_recorded_sample_store,
            ),
            stderr=True,
        )
        space_configuration.sampleStoreIdentifier = latest_recorded_sample_store

    elif parameters.new_sample_store:

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

        console_print(info_message, stderr=True)

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
        parameters.ado_configuration.latest_resource_ids[
            CoreResourceKinds.SAMPLESTORE
        ] = sample_store_resource.identifier

    elif parameters.use_default_sample_store:
        space_configuration.sampleStoreIdentifier = "default"

    if parameters.dry_run:
        console_print(ADO_CREATE_DRY_RUN_CONFIG_VALID, stderr=True)
        return None

    if space_configuration.sampleStoreIdentifier == "default":
        info_message = (
            f"{INFO}The {cyan('default')} sample store was requested to be used.\n"
            f"\tThe sample store referenced in the space definition will be ignored."
        )
        console_print(info_message, stderr=True)

        sql_store = get_sql_store(
            project_context=parameters.ado_configuration.project_context
        )

        if not sql_store.containsResourceWithIdentifier(
            identifier="default", kind=CoreResourceKinds.SAMPLESTORE
        ):
            from orchestrator.core import SampleStoreResource
            from orchestrator.core.samplestore.config import (
                SampleStoreConfiguration,
                SampleStoreModuleConf,
                SampleStoreSpecification,
            )

            with Status(ADO_SPINNER_SAVING_TO_DB):
                sql_store.addResource(
                    resource=SampleStoreResource(
                        identifier="default",
                        config=SampleStoreConfiguration(
                            specification=SampleStoreSpecification(
                                module=SampleStoreModuleConf(
                                    moduleClass="SQLSampleStore",
                                    moduleName="orchestrator.core.samplestore.sql",
                                ),
                                storageLocation=parameters.ado_configuration.project_context.metadataStore,
                            )
                        ),
                    )
                )

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

    # Save the identifier of the resource we created
    # for reuse
    parameters.ado_configuration.latest_resource_ids[
        CoreResourceKinds.DISCOVERYSPACE
    ] = space.uri

    console_print(
        f"{SUCCESS}Created space with identifier: {magenta(space.uri)}", stderr=True
    )

    return space.uri
