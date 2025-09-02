# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pandas as pd
import pytest

from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.sqlstore import (
    SQLStore,
)
from orchestrator.schema.entity import Entity


##################################################################
#
#                         CREATE
#
##################################################################
@pytest.fixture
def create_resources(sql_store):
    def _create_resource(resources: list[ADOResource], db: SQLStore = sql_store):
        for resource in resources:
            db.addResource(resource)

    return _create_resource


@pytest.fixture
def create_resource_with_related_identifiers(sql_store):
    def _create_resource_with_related_identifiers(
        resource: ADOResource,
        related_identifiers: list[str],
        db: SQLStore = sql_store,
    ):
        db.addResourceWithRelationships(
            resource=resource, relatedIdentifiers=related_identifiers
        )

    return _create_resource_with_related_identifiers


@pytest.fixture
def add_entities_to_sample_store():
    def _add_entities_to_sample_store(
        sql_sample_store: SQLSampleStore, entities: list[Entity]
    ):
        sql_sample_store.addEntities(entities)

    return _add_entities_to_sample_store


@pytest.fixture
def upsert_entities_to_sample_store():
    def _upsert_entities_to_sample_store(
        sql_sample_store: SQLSampleStore, entities: list[Entity]
    ):
        sql_sample_store.upsertEntities(entities)

    return _upsert_entities_to_sample_store


##################################################################
#
#                         READ
#
##################################################################
@pytest.fixture
def get_single_resource_by_identifier(sql_store):
    def _get_single_resource_by_identifier(
        identifier: str, kind: CoreResourceKinds
    ) -> ADOResource | None:

        return sql_store.getResource(identifier=identifier, kind=kind)

    return _get_single_resource_by_identifier


@pytest.fixture
def get_multiple_resources_by_identifier(sql_store):
    def _get_multiple_resources_by_identifier(
        identifiers: list[str],
    ) -> dict[str, ADOResource]:

        return sql_store.getResources(identifiers=identifiers)

    return _get_multiple_resources_by_identifier


@pytest.fixture
def get_resource_identifiers_by_resource_kind(sql_store):
    def _get_resource_identifiers_by_resource_kind(kind: str) -> pd.DataFrame:

        return sql_store.getResourceIdentifiersOfKind(kind=kind)

    return _get_resource_identifiers_by_resource_kind


@pytest.fixture
def get_related_resource_identifiers_by_identifier(sql_store):
    def _get_related_resource_identifiers_by_identifier(
        identifier: str,
    ) -> pd.DataFrame:

        return sql_store.getRelatedResourceIdentifiers(identifier=identifier)

    return _get_related_resource_identifiers_by_identifier


##################################################################
#
#                         UPDATE
#
##################################################################
@pytest.fixture
def update_resource(sql_store):
    def _update_resource(resource: ADOResource, db: SQLStore = sql_store):
        db.updateResource(resource)

    return _update_resource


##################################################################
#
#                         DELETE
#
##################################################################
@pytest.fixture
def delete_resource(sql_store):
    def _delete_resource(
        identifier: str,
    ) -> ADOResource | None:
        return sql_store.deleteResource(identifier=identifier)

    return _delete_resource
