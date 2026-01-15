# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import abc
from typing import TYPE_CHECKING

import pydantic

if TYPE_CHECKING:
    import pandas as pd

from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.core.samplestore.resource import SampleStoreResource
from orchestrator.utilities.location import (
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
        additional_message = message if message else ""

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
    ) -> ADOResource | None:
        """Returns the resource object with the given identifier

        NOTE:

         Parameters:
            identifier: A string. Identifier of a resource object

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
    ) -> dict[str, ADOResource]:
        """Returns all resource objects of a given kind

        Parameter:
            kind: A string. A resource object type as defined by CoreResourceKinds
            version: A version of the kind. If None all versions of the resource kind are returned
            field_selectors: A list of dictionaries of key/value selectors to filter the resources by.

        Returns:
            A dictionary whose keys are identifiers and values are the resource objects of the requested kind

        Exceptions:
            Raise a ValueError if the kind is not ADOResource subclass
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
    def getRelatedResourceIdentifiers(
        self, identifier: str, kind: str | None = None, version: str | None = None
    ) -> "pd.DataFrame":
        """Returns a DataFrame of resource identifiers related to a given resource identifier

        The returned identifiers can optionally be limited to those of a given kind

        Parameters:
            identifier: A string. Identifies a resource object
            kind: A string. A resource object type as defined by CoreResourceKinds
            version: A version of the kind. If None all versions of the resource kind are returned

        Returns:
            A pandas DataFrame

            The DataFrame has one column "IDENTIFIER" that contains the identifiers of the related resources.

            If there are no related resources for the identifier this method returns an empty DataFrame
        """

    @abc.abstractmethod
    def getRelatedResources(
        self, identifier: str, kind: CoreResourceKinds | None = None
    ) -> dict[str, ADOResource]:
        """
        Returns all resource object associated with identifier.
        Optionally returns only resources of the provided kind.
        """

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
    if (
        sample_store_resource_dict["config"]["specification"]["module"]["moduleClass"]
        == "SQLSampleStore"
    ):
        sample_store_resource_dict["config"]["specification"][
            "storageLocation"
        ] = storage_location.model_dump()

    return SampleStoreResource.model_validate(sample_store_resource_dict)


kind_custom_model_dump = {CoreResourceKinds.SAMPLESTORE.value: sample_store_dump}
kind_custom_model_load = {CoreResourceKinds.SAMPLESTORE.value: sample_store_load}
