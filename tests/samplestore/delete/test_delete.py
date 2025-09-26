# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import re

import pytest


def test_resource_deletion(
    resource_generator_from_db, delete_resource, sql_store, request
):
    _resource_kind, generator = resource_generator_from_db
    resource = request.getfixturevalue(generator)()
    delete_resource(resource.identifier)
    assert not sql_store.containsResourceWithIdentifier(identifier=resource.identifier)
    assert (
        sql_store.getRelatedResourceIdentifiers(identifier=resource.identifier).shape[0]
        == 0
    )


def test_nonexistent_resource_deletion(delete_resource, sql_store):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot delete resource with id IDoNotExist - it is not present"
        ),
    ):
        sql_store.deleteResource(identifier="IDoNotExist")
