# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from orchestrator.core.resources import CoreResourceKinds


def test_update_operation(
    random_operation_resource_from_db,
    sql_store,
    update_resource,
    get_single_resource_by_identifier,
):
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
