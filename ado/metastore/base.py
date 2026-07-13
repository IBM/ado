# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import abc
from typing import TYPE_CHECKING, Literal

import pydantic

if TYPE_CHECKING:
    import pandas as pd

    from ado.core.datacontainer.stats import DataContainerStatistics
    from ado.core.discoveryspace.stats import DiscoverySpaceStatistics

from ado.core.resources import ADOResource, CoreResourceKinds
from ado.core.samplestore.resource import SampleStoreResource
from ado.utilities.location import (
    SQLiteStoreConfiguration,
    SQLStoreConfiguration,
)


class ResourceDoesNotExistError(ValueError):
    def __init__(self, resource_id: str, kind: CoreResourceKinds | None = None) -> None:
        self.resource_id = resource_id
        self.kind = kind
        # Value Error will print the args passed to init when the exception is printed
        kind_specifier = f"of kind {kind} " if kind else ""
        super().__init__(
            f"There is no resource {kind_specifier}with the requested id, {resource_id}, in the project"
        )


class NoRelatedResourcesError(ValueError):
    def __init__(self, resource_id: str, kind: CoreResourceKinds) -> None:
        self.resource_id = resource_id
        self.kind = kind
        super().__init__(
            f"The resource with id, {resource_id}, does not have any related resources of kind {kind}"
        )


class ResourceHasChildrenError(ValueError):
    def __init__(
        self,
        resource_id: str,
        kind: CoreResourceKinds,
        children_resources: "pd.DataFrame",
    ) -> None:
        self.resource_id = resource_id
        self.kind = kind
        self.children_resources = children_resources
        super().__init__(
            f"Cannot delete {kind.value} {resource_id} because it has children resources"
        )


class ContextDoesNotExistError(ValueError):
    def __init__(self, resource_id: str, available_contexts: list[str]) -> None:
        self.resource_id = resource_id
        self.available_contexts = available_contexts
        super().__init__(f"Context {resource_id} does not exist")


class DatabaseOperationError(Exception): ...


class NotSupportedOnSQLiteError(DatabaseOperationError): ...


class DeleteFromDatabaseError(DatabaseOperationError):
    def __init__(
        self,
        resource_id: str,
        resource_kind: CoreResourceKinds,
        rollback_occurred: bool,
        message: str | None = None,
    ) -> None:
        self.resource_id = resource_id
        self.resource_kind = resource_kind
        self.message = message
        self.rollback_occurred = rollback_occurred

        rollback_message = (
            "The deletion was rolled back."
            if rollback_occurred
            else "The deletion was not rolled back."
        )
        additional_message = message or ""

        super().__init__(
            f"Failed to delete {resource_kind.value} {resource_id}. {additional_message}. {rollback_message}"
        )


class NonEmptySampleStorePreventingDeletionError(DatabaseOperationError):
    sample_store_id: str
    results_in_source: int

    def __init__(self, sample_store_id: str, results_in_source: int) -> None:
        self.sample_store_id = sample_store_id
        self.results_in_source = results_in_source

        super().__init__(
            f"Cannot delete sample store {sample_store_id} because "
            f"there are {results_in_source} measurement results present in the sample store."
        )


class RunningOperationsPreventingDeletionError(DatabaseOperationError):
    def __init__(self, operation_id: str, running_operations: list[str]) -> None:
        self.operation_id = operation_id
        self.running_operations = running_operations
        super().__init__(
            f"Cannot delete operation {operation_id} because the following operations "
            f"have started and have not completed: {running_operations}"
        )


class ResourceStore(abc.ABC):
    """Base class for ResourceStores"""

    @abc.abstractmethod
    def getResource(
        self,
        identifier: str,
        kind: CoreResourceKinds,
        raise_error_if_no_resource: bool = False,
        ignore_plugin_validation: bool = True,
    ) -> ADOResource | None:
        """Returns the resource object with the given identifier

        NOTE:

         Parameters:
            identifier: A string. Identifier of a resource object
            ignore_plugin_validation: When True (default), skip plugin registry
                validation on nested operation and actuator configuration fields.

        Returns:
            A resource instance corresponding to the identifier
            or None if there is no resource stored with the given identifier

        Exceptions:
            Raises a SystemError if the backend is not active
            Raises ValueError if there is a problem retrieving the resource
        """

    @abc.abstractmethod
    def getResources(self, identifiers: list[str]) -> dict[str, ADOResource]:
        """Returns a list of resource objects with the given identifiers

        Parameters:
            identifiers: A list. A set of identifier of resource objects

        Returns:
            A dictionary whose keys are identifiers and values are the resource objects.
            If there are no resources with any of the identifiers an empty dict is returned
            If there is no resource with the given identifier it will not be in the dict

        Exceptions:
            Raises a SystemError if the backend is not active
            Raises ValueError if there is a problem retrieving an existing resource
        """

    @abc.abstractmethod
    def getResourceIdentifiersOfKind(
        self,
        kind: str,
        version: str | None = None,
        field_selectors: list[dict[str, str]] | None = None,
        details: bool = False,
    ) -> "pd.DataFrame":
        """Returns a Pandas dataframe containing identifiers of the given resource type

        Parameter:
            kind: A string. A resource object type as defined by CoreResourceKinds
            version: A version of the kind. If None all versions of the resource kind are returned
            labels: A dictionary of key/value labels to filter the resources by.

        Return:
            A DataFrame with one column "IDENTIFIER"
            If there are no resources of the requested kind the dataframe will be empty
            If the backend is not active returns empty DataFrame

        Exception:
            Raises ValueError if resourceType is not one of the supported types
        """

    @abc.abstractmethod
    def getResourcesOfKind(
        self,
        kind: str,
        version: str | None = None,
        field_selectors: list[dict[str, str]] | None = None,
        ignore_validation_errors: bool = True,
    ) -> dict[str, ADOResource]:
        """Returns all resource objects of a given kind

        Parameter:
            kind: A string. A resource object type as defined by CoreResourceKinds
            version: A version of the kind. If None all versions of the resource kind are returned
            field_selectors: A list of dictionaries of key/value selectors to filter the resources by.
            ignore_validation_errors: If True (default), resources with validation errors are skipped.
                If False, ValueError is raised when a resource fails validation.

        Returns:
            A dictionary whose keys are identifiers and values are the resource objects of the requested kind

        Exceptions:
            Raise a ValueError if the kind is not ADOResource subclass
            Raises ValueError if ignore_validation_errors is False and a resource fails validation
            Raises a SystemError if the backend is not active"""

    @abc.abstractmethod
    def getRelatedSubjectResourceIdentifiers(
        self, identifier: str, kind: str | None = None, version: str | None = None
    ) -> "pd.DataFrame":
        """Returns identifiers of resources that have a relationship with
        "identifier" where "identifier" is the object"""

    @abc.abstractmethod
    def getRelatedObjectResourceIdentifiers(
        self, identifier: str, kind: str | None = None, version: str | None = None
    ) -> "pd.DataFrame":
        """Returns identifiers of resources that have a relationship with "identifier" where "identifier" is the subject"""

    @abc.abstractmethod
    def containsResourceWithIdentifier(
        self, identifier: str, kind: CoreResourceKinds | None = None
    ) -> bool:
        """Returns True if the receiver contains a resource object with a given identifier
        (optionally of a specific kind)

        False otherwise
        """

    @abc.abstractmethod
    def addResource(self, resource: ADOResource) -> None:

        pass

    @abc.abstractmethod
    def addRelationship(
        self,
        subjectIdentifier: str,
        objectIdentifier: str,
    ) -> None:

        pass

    @abc.abstractmethod
    def addRelationshipForResources(
        self, subjectResource: pydantic.BaseModel, objectResource: pydantic.BaseModel
    ) -> None:

        pass

    @abc.abstractmethod
    def addResourceWithRelationships(
        self,
        resource: ADOResource,
        relatedIdentifiers: list,
    ) -> None:
        """For the relationship, the resource id is stored as object and the other ids as subjects

        This is because the others ids must already exist"""

    @abc.abstractmethod
    def updateResource(self, resource: ADOResource) -> None:
        """Replaces any data stored against "resource.identifier" with resource

        Raises:
            ValueError if resource is not already stored.

        """

    @abc.abstractmethod
    def deleteResource(self, identifier: str) -> None:

        pass

    @abc.abstractmethod
    def deleteObjectRelationships(self, identifier: str) -> None:
        """Deletes all recorded relationships for identifier where it is the object

        Only works if it is not the subject of another relationship"""

    @abc.abstractmethod
    def delete_sample_store(
        self, identifier: str, force_deletion: bool = False
    ) -> None:
        pass

    @abc.abstractmethod
    def delete_operation(
        self, identifier: str, ignore_running_operations: bool = False
    ) -> None:
        pass

    @abc.abstractmethod
    def delete_discovery_space(self, identifier: str) -> None:
        pass

    @abc.abstractmethod
    def delete_data_container(self, identifier: str) -> None:
        pass

    @abc.abstractmethod
    def delete_actuator_configuration(self, identifier: str) -> None:
        pass

    @abc.abstractmethod
    def delete_document(self, identifier: str) -> None:
        pass

    @abc.abstractmethod
    def get_resources_by_relationship(
        self,
        kind: CoreResourceKinds,
        identifier: str | set[str] | None,
        hierarchy_direction: Literal["up", "down", "both"],
        max_hops: int | None = None,
        identifiers_only: bool = False,
        include_start_resources: bool = False,
    ) -> (
        dict[CoreResourceKinds, set[str]]
        | dict[str, dict[CoreResourceKinds, set[str]]]
        | dict[CoreResourceKinds, dict[str, ADOResource]]
        | dict[str, dict[CoreResourceKinds, dict[str, ADOResource]]]
    ):
        """Walk the resource hierarchy and return related resources.

        Args:
            kind: The :class:`~ado.core.resources.CoreResourceKinds` of
                the starting resources.
            identifier: Controls which resources are used as traversal origins.
                ``str`` for a single start resource, ``set[str]`` for multiple,
                or ``None`` to seed from all resources of ``kind``.
            hierarchy_direction: ``'up'`` (child → parent), ``'down'``
                (parent → child), or ``'both'``.
            max_hops: Maximum number of relationship hops to follow. ``None``
                traverses to the full depth of the hierarchy.
            identifiers_only: When ``True`` return only discovered identifiers;
                when ``False`` (default) return hydrated
                :class:`~ado.core.resources.ADOResource` objects.
            include_start_resources: When ``True`` include the start resource(s)
                in the result. Requires ``identifiers_only=False`` and
                ``identifier`` to be a ``str`` or ``set[str]``.

        Returns:
            A nested dict whose shape depends on whether a single or multiple
            identifiers were requested and whether ``identifiers_only`` is set.

        Raises:
            ValueError: If arguments are invalid or incompatible.
        """

    @abc.abstractmethod
    def get_space_metastore_stats(
        self,
        space_ids: str | set[str],
    ) -> "DiscoverySpaceStatistics | dict[str, DiscoverySpaceStatistics]":
        """Return lightweight metastore-level statistics for one or many spaces.

        All counts are computed with pure SQL against the ``resources`` and
        ``resource_relationships`` tables — no sample store access is needed.
        The returned :class:`~ado.core.discoveryspace.stats.DiscoverySpaceStatistics`
        objects only have ``number_of_experiments``, ``number_of_operations``,
        and ``number_of_explore_operations`` populated; all other fields are
        ``None`` or ``0`` as appropriate.

        Args:
            space_ids: A single space identifier (``str``) or a set of space
                identifiers (``set[str]``).

        Returns:
            When ``space_ids`` is a ``str``: a
            :class:`~ado.core.discoveryspace.stats.DiscoverySpaceStatistics`
            for that space.
            When ``space_ids`` is a ``set[str]``: a ``dict`` keyed by space ID
            mapping each to its
            :class:`~ado.core.discoveryspace.stats.DiscoverySpaceStatistics`.
            Space IDs that have no operations are included with zero counts.
        """

    @abc.abstractmethod
    def get_datacontainer_stats(
        self,
        datacontainer_ids: set[str],
    ) -> "dict[str, DataContainerStatistics]":
        """Return lightweight statistics for a set of DataContainer IDs.

        Args:
            datacontainer_ids: A set of DataContainer identifiers to query.

        Returns:
            A ``dict`` keyed by DataContainer ID mapping each to its
            :class:`~ado.core.datacontainer.stats.DataContainerStatistics`.
            IDs that are not present in the database are returned with all-zero
            stats.  An empty input set returns an empty dict immediately
            (no query issued).
        """


def sample_store_dump(
    sample_store_resource: SampleStoreResource,
) -> str:

    # We want to apply the following policies to sample store resources
    # 1. Do not store SQLSampleStore storage access information in the resource
    #
    # We can implement this policy by adding the following constraint
    # - If a sample store resource uses a SQLSampleStore it is stored in the same DB as the resource
    #
    # This allows us to remove the SQL storage information from sample store resource when dumping it
    # and re-add it when it is loaded (as the SQL accesses information is == the resource stores access information)

    if (
        sample_store_resource.config.specification.module.moduleClass
        == "SQLSampleStore"
    ):
        exclude = {"config": {"specification": {"storageLocation": True}}}
    else:
        exclude = None

    return sample_store_resource.model_dump_json(exclude_none=True, exclude=exclude)


def sample_store_load(
    sample_store_resource_dict: dict,
    storage_location: SQLiteStoreConfiguration | SQLStoreConfiguration,
) -> SampleStoreResource:
    """Adds storage location information to SQL sample stores"""
    # Check for required keys in the nested structure
    key_chain = ["config", "specification", "module", "moduleClass"]
    current_dict = sample_store_resource_dict

    for i, key in enumerate(key_chain):
        if not isinstance(current_dict, dict):
            missing_path = ".".join(key_chain[:i])
            raise ValueError(
                f"Invalid sample store resource structure: expected dictionary at '{missing_path}', "
                f"but got {type(current_dict).__name__}"
            )

        if key not in current_dict:
            missing_path = ".".join(key_chain[: i + 1])
            raise ValueError(
                f"Invalid sample store resource structure: missing required key '{missing_path}'"
            )

        current_dict = current_dict[key]

    if (
        sample_store_resource_dict["config"]["specification"]["module"]["moduleClass"]
        == "SQLSampleStore"
    ):
        sample_store_resource_dict["config"]["specification"]["storageLocation"] = (
            storage_location.model_dump()
        )

    from ado.utilities.pydantic import (
        do_not_populate_ado_provenance_context,
    )

    return SampleStoreResource.model_validate(
        sample_store_resource_dict,
        context=do_not_populate_ado_provenance_context,
    )


kind_custom_model_dump = {CoreResourceKinds.SAMPLESTORE.value: sample_store_dump}
kind_custom_model_load = {CoreResourceKinds.SAMPLESTORE.value: sample_store_load}
