# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
from collections.abc import Callable

from orchestrator.core import OperationResource
from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.metastore.sqlstore import SQLStore


def test_update_operation(
    random_operation_resource_from_db: Callable[
        [str | None, str | None], OperationResource
    ],
    sql_store: SQLStore,
    update_resource: Callable[[ADOResource, SQLStore], None],
    get_single_resource_by_identifier: Callable[
        [str, CoreResourceKinds], ADOResource | None
    ],
) -> None:
    operation = random_operation_resource_from_db()

    metadata = {
        "new_samples_generated": 10,
        "entities_submitted": 20,
        "experiments_requested": 40,
    }
    operation.metadata = metadata
    update_resource(operation)

    resource_from_db = get_single_resource_by_identifier(
        identifier=operation.identifier, kind=CoreResourceKinds.OPERATION
    )
    assert resource_from_db.metadata["new_samples_generated"] == 10
    assert resource_from_db.metadata["entities_submitted"] == 20
    assert resource_from_db.metadata["experiments_requested"] == 40
