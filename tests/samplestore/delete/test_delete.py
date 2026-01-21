# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import re
from collections.abc import Callable

import pytest

from orchestrator.core import ADOResource, CoreResourceKinds
from orchestrator.metastore.sqlstore import SQLStore


def test_resource_deletion(
    resource_generator_from_db: tuple[CoreResourceKinds, str],
    delete_resource: Callable[[str], ADOResource | None],
    sql_store: SQLStore,
    request: pytest.FixtureRequest,
) -> None:
    _resource_kind, generator = resource_generator_from_db
    resource = request.getfixturevalue(generator)()
    delete_resource(resource.identifier)
    assert not sql_store.containsResourceWithIdentifier(identifier=resource.identifier)
    assert (
        sql_store.getRelatedResourceIdentifiers(identifier=resource.identifier).shape[0]
        == 0
    )


def test_nonexistent_resource_deletion(
    delete_resource: Callable[[str], ADOResource | None], sql_store: SQLStore
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot delete resource with id IDoNotExist - it is not present"
        ),
    ):
        sql_store.deleteResource(identifier="IDoNotExist")
