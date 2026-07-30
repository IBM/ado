# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from ado.cli.models.parameters import AdoUpgradeCommandParameters
from ado.core import CoreResourceKinds


def upgrade_sample_store(parameters: AdoUpgradeCommandParameters) -> None:
    from rich.status import Status

    from ado.cli.utils.generic.wrappers import get_sql_store
    from ado.cli.utils.output.prints import (
        ADO_SPINNER_QUERYING_DB,
        ERROR,
        SUCCESS,
        console_print,
    )
    from ado.cli.utils.resources.handlers import (
        handle_ado_upgrade,
    )
    from ado.core.samplestore.base import (
        FailedToDecodeStoredEntityError,
        FailedToDecodeStoredMeasurementResultForEntityError,
        SampleStore,
    )

    # Step 1: upgrade SampleStoreResource metastore records (unchanged behaviour).
    handle_ado_upgrade(
        parameters=parameters, resource_type=CoreResourceKinds.SAMPLESTORE
    )

    if not parameters.upgrade_entities_and_results:
        console_print(SUCCESS)
        return

    # Step 2: open each sample store and upgrade entities and results.
    # Only runs when --upgrade-entities-and-results is passed, as it can be slow.
    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )

    with Status(ADO_SPINNER_QUERYING_DB) as status:
        resources = sql_store.getResourcesOfKind(
            kind=CoreResourceKinds.SAMPLESTORE.value
        )

        total = len(resources)
        for idx, resource in enumerate(resources.values()):
            store_id = resource.identifier
            progress = f"[{idx + 1}/{total}]"

            try:
                status.update(
                    f"Upgrading entities in store [b cyan]{store_id}[/b cyan] {progress}"
                )
                store = SampleStore.from_resource(resource)
                n_entities = store.upgrade_entities()

                status.update(
                    f"Upgrading measurement results in store [b cyan]{store_id}[/b cyan] {progress}"
                )
                n_results = store.upgrade_measurement_results()

            except FailedToDecodeStoredEntityError as e:
                status.stop()
                console_print(
                    f"{ERROR}Store [b cyan]{store_id}[/b cyan]: "
                    f"failed to decode entity [b]{e.entity_identifier}[/b].\n"
                    f"  Cause: {e.cause}"
                )
                return

            except FailedToDecodeStoredMeasurementResultForEntityError as e:
                status.stop()
                console_print(
                    f"{ERROR}Store [b cyan]{store_id}[/b cyan]: "
                    f"failed to decode measurement result for entity [b]{e.entity_identifier}[/b].\n"
                    f"  Cause: {e.cause}"
                )
                return

            except SystemError as e:
                status.stop()
                console_print(
                    f"{ERROR}Store [b cyan]{store_id}[/b cyan]: "
                    f"unexpected row-count mismatch during upgrade.\n"
                    f"  Cause: {e}"
                )
                return

            status.stop()
            console_print(
                f"  {progress} Store [b cyan]{store_id}[/b cyan]: "
                f"upgraded {n_entities} entities, {n_results} measurement results"
            )
            status.start()

    console_print(SUCCESS)
