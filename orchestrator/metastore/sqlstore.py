# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import logging
import os

import pandas as pd
import pydantic
import sqlalchemy

import orchestrator.core
import orchestrator.metastore
import orchestrator.metastore.sql.statements
import orchestrator.utilities
from orchestrator.core.resources import ADOResourceEventEnum, CoreResourceKinds
from orchestrator.metastore.base import (
    DeleteFromDatabaseError,
    NonEmptySampleStorePreventingDeletionError,
    NotSupportedOnSQLiteError,
    ResourceDoesNotExistError,
    ResourceStore,
    RunningOperationsPreventingDeletionError,
    kind_custom_model_dump,
    kind_custom_model_load,
)
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sql.utils import (
    create_sql_resource_store,
    engine_for_sql_store,
)

#
# Temporary - Refactoring plan
# 1. (10) Update SQLStore to new API and refactor consumers and test - DONE
# 2. (5) Update SQLEntityStore to new API and refactor consumers - DONE (25/43)
# 3. (7) Implement SQLResourceStore (need 4 to test) - DONE (32/43)
# 4. (2) Add/Use resources for DiscoverySpace and EntityStore and test - DONE
# 5. (2) Refactor MetricServer to use new TS Metrics API - TODO
# 6. (2) Enable dynamic switch for resource store - DONE
# 7. (5) Implement NewSQLEntityStore and test - TODO
# 8. (2) Set NewSQLEntityStore as default while maintaining backwards compatibility - TODO


class SQLStore(ResourceStore):
    """Base class for SQLStores"""

    def __new__(cls, project_context: ProjectContext):

        engine = engine_for_sql_store(configuration=project_context.metadataStore)
        inspector = sqlalchemy.inspect(engine)

        # We set ensureExists manually by checking just one table.
        return SQLResourceStore(
            project_context=project_context,
            ensureExists=not inspector.has_table("resources"),
        )

    def __init__(self, project_context: ProjectContext):

        pass


class SQLResourceStore(ResourceStore):
    """

    A SQLResourceStore can be used to store resources and their relationships
    A SQLResourceStore can be active or inactive.
    If inactive it does not send data to the store - this is useful for debugging.

    In inactive mode
    - methods to add data to the db will instead print the information added.
    - methods to get data from the db will raise exceptions

    """

    def __init__(self, project_context: ProjectContext, ensureExists=True):
        """
        Creates a SQLResourceStore instance based on the ProjectContext

        Parameters:
            project_context: The ProjectContext containing credentials to connect to the SQL db
            ensureExists: If True the existence of the required tables is checked, and
                they are created if missing. If False the check is not performed (assumes existence).
                This can be used to skip the check if the caller knows the tables exist.

        Note:
        -  If a project_context object is passed the value of its active field determines is the SQLStore is active.
           By default, this field is True

        """

        self.project_context = project_context
        self.configuration = project_context.metadataStore

        FORMAT = orchestrator.utilities.logging.FORMAT
        LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
        logging.basicConfig(level=LOGLEVEL, format=FORMAT)

        self.log = logging.getLogger("SQLStore")
        self.log.debug(
            f"Initialised SQLStore. Host: {self.configuration.host} "
            f"Database: {self.configuration.database if self.configuration.scheme != 'sqlite' else self.configuration.path}"
        )

        if ensureExists:
            # self.log.warning("Ensuring store existence")
            self.log.debug("Initialising SQL db if it does not exist")
            create_sql_resource_store(self.engine)
            self.log.debug("Done")

        super().__init__()

    @property
    def engine(self):

        return engine_for_sql_store(configuration=self.configuration)

    def getResourceRaw(self, identifier) -> dict | None:

        query = sqlalchemy.text(
            "SELECT * FROM resources WHERE identifier=:identifier"
        ).bindparams(identifier=identifier)

        # self.log.warning("GETTING RESOURCE")
        with self.engine.connect() as connectable:
            table = pd.read_sql(query, con=connectable)
        # self.log.warning("GOT RESOURCE")

        raw = None
        if table.shape[0] > 0:
            raw = json.loads(table.data[0])

        return raw

    def getResource(
        self,
        identifier: str,
        kind: CoreResourceKinds,
        raise_error_if_no_resource: bool = False,
    ) -> orchestrator.core.resources.ADOResource | None:

        query = sqlalchemy.text(
            """
            SELECT * FROM resources
            WHERE identifier=:identifier
            AND kind=:kind
            """
        ).bindparams(identifier=identifier, kind=kind.value)

        # self.log.warning("GETTING RESOURCE")
        with self.engine.connect() as connectable:
            table = pd.read_sql(query, con=connectable)
        # self.log.warning("GOT RESOURCE")

        resource = None
        if table.shape[0] > 0:
            d = json.loads(table.data[0])
            custom_model_loader = kind_custom_model_load.get(table.kind[0])
            if custom_model_loader:
                resource = custom_model_loader(d, self.configuration)
            else:
                resource = orchestrator.core.kindmap[table.kind[0]](**d)

            # The stored resource should always have a version - if somehow it doesn't we want this to fail
            if orchestrator.core.resources.VersionIsGreaterThan(
                resource.version, d.get("version", "v0")
            ):
                self.updateResource(resource)

        if not resource and raise_error_if_no_resource:
            raise ResourceDoesNotExistError(resource_id=identifier, kind=kind)

        return resource

    def getResources(
        self, identifiers: list[str]
    ) -> dict[str, orchestrator.core.resources.ADOResource]:

        retval = {}
        if len(identifiers) != 0:
            if isinstance(identifiers, pd.Series):
                identifiers = identifiers.tolist()

            query = sqlalchemy.text(
                "SELECT * FROM resources WHERE identifier in :identifiers"
            ).bindparams(
                sqlalchemy.bindparam(
                    key="identifiers", value=identifiers, expanding=True
                )
            )

            with self.engine.connect() as connectable:
                table = pd.read_sql(query, con=connectable)

            if table.shape[0] > 0:
                for identifier, data, kind in zip(
                    table.identifier, table.data, table.kind
                ):
                    d = json.loads(data)
                    custom_model_loader = kind_custom_model_load.get(kind)
                    if custom_model_loader:
                        resource = custom_model_loader(d, self.configuration)
                        retval[identifier] = resource
                    else:
                        try:
                            resource = orchestrator.core.kindmap[kind](**d)
                        except pydantic.ValidationError as error:
                            self.log.warning(
                                f"Unable to create pydantic model for resource with id, {identifier} with data, {data}. {error}"
                            )
                        else:
                            retval[identifier] = resource

        return retval

    def getResourceIdentifiersOfKind(
        self,
        kind: str,
        version: str | None = None,
        field_selectors: list[dict[str, str]] | None = None,
        details: bool = False,
    ) -> pd.DataFrame:

        if kind not in [v.value for v in orchestrator.core.resources.CoreResourceKinds]:
            raise ValueError(f"Unknown kind specified: {kind}")

        # SELECT
        select_statement = "SELECT identifier"
        select_name = (
            orchestrator.metastore.sql.statements.resource_select_metadata_field(
                field_name="name", needs_select=False, dialect=self.engine.dialect.name
            )
        )
        select_age = (
            orchestrator.metastore.sql.statements.resource_select_created_field(
                as_age=True, needs_select=False, dialect=self.engine.dialect.name
            )
        )

        if details:
            select_description = (
                orchestrator.metastore.sql.statements.resource_select_metadata_field(
                    field_name="description",
                    needs_select=False,
                    dialect=self.engine.dialect.name,
                )
            )
            select_labels = (
                orchestrator.metastore.sql.statements.resource_select_metadata_field(
                    field_name="labels",
                    needs_select=False,
                    dialect=self.engine.dialect.name,
                )
            )

            select_statement = f"{select_statement} {select_name} {select_description} {select_labels} {select_age} "
        else:
            select_statement = f"{select_statement} {select_name} {select_age} "

        # Add the status to the resources that have it
        if kind == orchestrator.core.resources.CoreResourceKinds.OPERATION.value:
            select_status = (
                orchestrator.metastore.sql.statements.resource_select_data_field(
                    field_name="status",
                    needs_select=False,
                    dialect=self.engine.dialect.name,
                )
            )
            select_statement = f"{select_statement} {select_status} "

        # FROM
        from_statement = "FROM resources "

        field_selectors = field_selectors if field_selectors else {}

        # WHERE
        where_statement = f"WHERE kind = '{kind}'"
        field_queries = ""
        if not field_selectors:
            field_selectors = {}

        for field_selector in field_selectors:
            for path, candidate in field_selector.items():
                field_queries += orchestrator.metastore.sql.statements.resource_filter_by_arbitrary_selection(
                    path=path,
                    candidate=candidate,
                    needs_where=False,
                    dialect=self.engine.dialect.name,
                )

        version_filter = f"AND version = '{version}'" if version else ""
        where_statement = f"""{where_statement} {field_queries} {version_filter}"""

        # ORDER BY
        order_by_statement = (
            orchestrator.metastore.sql.statements.resource_order_by_age_desc(
                self.engine.dialect.name
            )
        )

        query = f"{select_statement} {from_statement} {where_statement} {order_by_statement};"
        with self.engine.connect() as connectable:
            table = pd.read_sql(query, con=connectable)

        columns = (
            ["IDENTIFIER", "NAME", "DESCRIPTION", "LABELS", "AGE"]
            if details
            else ["IDENTIFIER", "NAME", "AGE"]
        )

        output_df = pd.DataFrame(
            data={
                "IDENTIFIER": table["identifier"],
                "NAME": table["name"],
                "AGE": table["age"],
            }
        )

        import datetime
        import math

        # The DB returns us timedelta objects in seconds, we want Pandas to
        # parse them correctly
        output_df["AGE"] = output_df["AGE"].apply(
            lambda x: (datetime.timedelta(seconds=x) if not math.isnan(x) else x)
        )

        if details:
            output_df["DESCRIPTION"] = table["description"]
            output_df["LABELS"] = table["labels"]

        if kind == orchestrator.core.resources.CoreResourceKinds.OPERATION.value:
            columns.insert(-1, "STATUS")
            output_df["STATUS"] = table["status"]

        return output_df[columns]

    def resourceTable(self):

        query = """SELECT * FROM resources"""

        with self.engine.connect() as connectable:
            return pd.read_sql(query, con=connectable)

    def getResourcesOfKind(
        self,
        kind: str,
        version: str | None = None,
        field_selectors: list[dict[str, str]] | None = None,
    ) -> dict[str, orchestrator.core.resources.ADOResource]:
        """Returns all resources of a given kind

        A kind is a version+type"""

        identifiers = self.getResourceIdentifiersOfKind(
            kind=kind, version=version, field_selectors=field_selectors
        )
        return self.getResources(identifiers=identifiers["IDENTIFIER"])

    def getRelatedSubjectResourceIdentifiers(
        self, identifier, kind: str | None = None, version: str | None = None
    ) -> pd.DataFrame:
        """Returns identifiers of resources that have a relationship with "identifier"
        where "identifier" is the object"""

        query_text = """SELECT subject_identifier, resources.kind
                              FROM resource_relationships
                              INNER JOIN resources
                                 ON resource_relationships.subject_identifier = resources.identifier
                              WHERE resource_relationships.object_identifier=:identifier"""
        query_parameters = {"identifier": identifier}

        if kind is not None:
            query_text += """ AND resources.kind=:kind"""
            query_parameters["kind"] = kind

        if version is not None:
            query_text += """ AND resources.version=:version"""
            query_parameters["version"] = version

        query = sqlalchemy.text(query_text).bindparams(**query_parameters)
        with self.engine.connect() as connectable:
            table = pd.read_sql(query, con=connectable)

        related_identifiers = table["subject_identifier"].values
        related_kinds = table["kind"].values

        return pd.DataFrame({"IDENTIFIER": related_identifiers, "TYPE": related_kinds})

    def getRelatedObjectResourceIdentifiers(
        self, identifier, kind: str | None = None, version: str | None = None
    ) -> pd.DataFrame:
        """Returns identifiers of resources that have a relationship with "identifier"
        where "identifier" is the subject"""

        # First select where identifier is the subject
        query_text = """SELECT object_identifier, resources.kind
                    FROM resource_relationships
                    INNER JOIN resources
                       ON resource_relationships.object_identifier = resources.identifier
                    WHERE resource_relationships.subject_identifier=:identifier"""
        query_parameters = {"identifier": identifier}

        if kind is not None:
            query_text += " AND resources.kind=:kind"
            query_parameters["kind"] = kind

        if version is not None:
            query_text += " AND resources.version=:version"
            query_parameters["version"] = version

        query = sqlalchemy.text(query_text).bindparams(**query_parameters)
        with self.engine.connect() as connectable:
            table = pd.read_sql(query, con=connectable)

        related_identifiers = table["object_identifier"].values
        related_kinds = table["kind"].values

        return pd.DataFrame({"IDENTIFIER": related_identifiers, "TYPE": related_kinds})

    def getRelatedResourceIdentifiers(
        self, identifier, kind: str | None = None, version: str | None = None
    ) -> pd.DataFrame:

        relatedAsObject = self.getRelatedObjectResourceIdentifiers(
            identifier=identifier, kind=kind, version=version
        )
        relatedAsSubject = self.getRelatedSubjectResourceIdentifiers(
            identifier=identifier, kind=kind, version=version
        )

        return pd.DataFrame(
            {
                "IDENTIFIER": relatedAsObject["IDENTIFIER"].values.tolist()
                + relatedAsSubject["IDENTIFIER"].values.tolist(),
                "TYPE": relatedAsObject["TYPE"].values.tolist()
                + relatedAsSubject["TYPE"].values.tolist(),
            }
        )

    def getRelatedResources(
        self, identifier: str, kind: CoreResourceKinds | None = None
    ) -> dict[str, orchestrator.core.resources.ADOResource]:
        """
        Returns all resource object associated with identifier.
        Optionally returns only resources of the provided kind.
        """

        identifiers = self.getRelatedResourceIdentifiers(
            identifier=identifier, kind=kind.value
        )
        return self.getResources(identifiers=identifiers["IDENTIFIER"])

    def containsResourceWithIdentifier(
        self, identifier: str, kind: CoreResourceKinds | None = None
    ) -> bool:

        query_text = "SELECT COUNT(1) FROM resources WHERE identifier=:identifier"
        query_parameters = {"identifier": identifier}
        if kind:
            query_text += " AND kind=:kind"
            query_parameters["kind"] = kind.value

        query = sqlalchemy.text(query_text).bindparams(**query_parameters)
        with self.engine.connect() as connectable:
            exe = connectable.execute(query)
            row_count = exe.scalar()

        return row_count != 0

    def addResource(self, resource: orchestrator.core.resources.ADOResource):

        if not isinstance(resource, orchestrator.core.resources.ADOResource):
            raise ValueError(
                f"Cannot add resource, {resource}, that is not a subclass of ADOResource"
            )

        # Connect to SQL and add entry
        if self.containsResourceWithIdentifier(resource.identifier):
            raise ValueError(
                f"Resource with id {resource.identifier} already present. "
                f"Use updateResource if you want to overwrite it"
            )
        resource.status.append(
            orchestrator.core.resources.ADOResourceStatus(
                event=ADOResourceEventEnum.ADDED
            )
        )
        custom_model_dump = kind_custom_model_dump.get(resource.kind)
        if custom_model_dump:
            representation = custom_model_dump(resource)
        else:
            representation = resource.model_dump_json()

        with self.engine.begin() as connectable:
            query = sqlalchemy.text(
                r"INSERT INTO resources"
                r"(identifier, kind, version, data)"
                r"VALUES(:identifier, :kind, :version, :data)"
            ).bindparams(
                identifier=resource.identifier,
                kind=resource.kind.value,
                version=resource.version,
                data=representation,
            )
            connectable.execute(query)

    def addRelationship(
        self,
        subjectIdentifier: str,
        objectIdentifier: str,
    ):

        # Connect to SQL and add entry
        with self.engine.begin() as connectable:
            query = sqlalchemy.text(
                r"INSERT INTO resource_relationships"
                r"(subject_identifier, object_identifier)"
                r"VALUES(:subject_identifier, :object_identifier)"
            ).bindparams(
                subject_identifier=subjectIdentifier,
                object_identifier=objectIdentifier,
            )
            connectable.execute(query)

    def addRelationshipForResources(
        self, subjectResource: pydantic.BaseModel, objectResource: pydantic.BaseModel
    ):

        self.addRelationship(
            subjectIdentifier=subjectResource.identifier,
            objectIdentifier=objectResource.identifier,
        )

    def addResourceWithRelationships(
        self,
        resource: orchestrator.core.resources.ADOResource,
        relatedIdentifiers: list,
    ):
        """For the relationship, the resource id is stored as object and the other ids as subjects

        This is because the others ids must already exist"""

        # Test that the relatedIdentifiers exist before adding
        r = [
            self.containsResourceWithIdentifier(identifier=ident)
            for ident in relatedIdentifiers
        ]
        if False in r:
            raise ValueError(f"Unknown resource identifier passed {relatedIdentifiers}")

        self.addResource(resource=resource)
        for identifier in relatedIdentifiers:
            self.addRelationship(
                subjectIdentifier=identifier, objectIdentifier=resource.identifier
            )

    def updateResource(self, resource: orchestrator.core.resources.ADOResource):
        """Replaces any data stored against "resource.identifier" with resource

        Raises:
            ValueError if resource is not already stored.

        """

        resource.status.append(
            orchestrator.core.resources.ADOResourceStatus(
                event=ADOResourceEventEnum.UPDATED
            )
        )
        custom_model_dump = kind_custom_model_dump.get(resource.kind)
        if custom_model_dump:
            representation = custom_model_dump(resource)
        else:
            representation = resource.model_dump_json()

        with self.engine.begin() as connectable:
            query = orchestrator.metastore.sql.statements.resource_upsert(
                resource=resource,
                json_representation=representation,
                dialect=self.engine.dialect.name,
            )

            connectable.execute(query)

    def deleteResource(self, identifier):

        if not self.containsResourceWithIdentifier(identifier):
            raise ValueError(
                f"Cannot delete resource with id {identifier} - it is not present"
            )

        # Cannot delete if there are relationships where the identifier is the subject
        relatedAsObject = self.getRelatedObjectResourceIdentifiers(
            identifier=identifier
        )
        if len(relatedAsObject) > 0:
            raise ValueError(
                f"Cannot delete resource {identifier} as there are existing relationships where it is the subject. "
                f"You must delete all the related object resources first:\n{relatedAsObject['IDENTIFIER']}"
            )
        # Delete all relationships where the identifier is the object
        self.deleteObjectRelationships(identifier=identifier)
        with self.engine.begin() as connectable:
            query = sqlalchemy.text(
                r"DELETE FROM resources WHERE identifier=:identifier"
            ).bindparams(identifier=identifier)
            connectable.execute(query)

    def deleteObjectRelationships(self, identifier):
        """Deletes all recorded relationships for identifier where it is the object

        Only works if it is not the subject of another relationship"""

        # Cannot delete if there are object relationships (the identifier is the subject) as this breaks provenance
        relatedAsObject = self.getRelatedObjectResourceIdentifiers(
            identifier=identifier
        )
        if len(relatedAsObject) > 0:
            raise ValueError(
                f"Cannot delete relationships where {identifier} is the object as there are existing relationships where it is the subject. "
                f"You must delete all the related object resources first:\n{relatedAsObject['IDENTIFIER']}"
            )
        with self.engine.begin() as connectable:
            query = sqlalchemy.text(
                r"DELETE FROM resource_relationships WHERE object_identifier=:identifier"
            ).bindparams(identifier=identifier)
            connectable.execute(query)

    def delete_sample_store(self, identifier: str, force_deletion: bool = False):
        import sqlalchemy.orm

        with sqlalchemy.orm.Session(self.engine) as session:

            if not force_deletion:
                with session.begin():

                    results_in_source = session.execute(
                        sqlalchemy.text(
                            f"SELECT COUNT(*) FROM sqlsource_{identifier}_measurement_results"  # noqa: S608 - identifier is trusted
                        )
                    ).scalar_one()

                    if results_in_source != 0:
                        raise NonEmptySampleStorePreventingDeletionError(
                            sample_store_id=identifier,
                            results_in_source=results_in_source,
                        )

            # AP 05/08/2025:
            # DROP TABLE statements trigger an implicit commit on MySQL
            # ref:https://dev.mysql.com/doc/refman/8.4/en/implicit-commit.html
            # This means we must delete everything from the tables first,
            # to reduce the chances of the DB being left in an unclean state
            try:
                with session.begin():

                    session.execute(
                        sqlalchemy.text(
                            "DELETE FROM resource_relationships WHERE object_identifier=:identifier"
                        ).bindparams(identifier=identifier)
                    )

                    session.execute(
                        sqlalchemy.text(
                            "DELETE FROM resources WHERE identifier=:identifier AND kind=:kind"
                        ).bindparams(
                            identifier=identifier,
                            kind=CoreResourceKinds.SAMPLESTORE.value,
                        )
                    )

                    session.execute(
                        sqlalchemy.text(
                            f"DELETE FROM sqlsource_{identifier}_measurement_requests_results"  # noqa: S608 - identifier is trusted
                        )
                    )

                    session.execute(
                        sqlalchemy.text(
                            f"DELETE FROM sqlsource_{identifier}_measurement_requests"  # noqa: S608 - identifier is trusted
                        )
                    )

                    session.execute(
                        sqlalchemy.text(
                            f"DELETE FROM sqlsource_{identifier}_measurement_results"  # noqa: S608 - identifier is trusted
                        )
                    )

                    session.execute(
                        sqlalchemy.text(
                            f"DELETE FROM sqlsource_{identifier}"  # noqa: S608 - identifier is trusted
                        )
                    )

            except Exception as e:
                session.rollback()
                raise DeleteFromDatabaseError(
                    resource_id=identifier,
                    resource_kind=CoreResourceKinds.SAMPLESTORE,
                    rollback_occurred=True,
                ) from e

            # We still attempt a rollback in case things go wrong as it's
            # supported by SQLite
            try:
                with session.begin():

                    session.execute(
                        sqlalchemy.text(f"DROP TABLE sqlsource_{identifier}")
                    )

                    session.execute(
                        sqlalchemy.text(
                            f"DROP TABLE sqlsource_{identifier}_measurement_requests"
                        )
                    )

                    session.execute(
                        sqlalchemy.text(
                            f"DROP TABLE sqlsource_{identifier}_measurement_results"
                        )
                    )

                    session.execute(
                        sqlalchemy.text(
                            f"DROP TABLE sqlsource_{identifier}_measurement_requests_results"
                        )
                    )
            except Exception as e:
                session.rollback()
                raise DeleteFromDatabaseError(
                    resource_id=identifier,
                    resource_kind=CoreResourceKinds.SAMPLESTORE,
                    message="Some sample store tables were not deleted",
                    rollback_occurred=False,
                ) from e

    def delete_operation(
        self, identifier: str, ignore_running_operations: bool = False
    ):
        import sqlalchemy.orm

        if self.engine.dialect.name == "sqlite" and not ignore_running_operations:
            raise NotSupportedOnSQLiteError(
                "SQLite does not support checking if there are other operations running "
                "and using the same sample store."
            )

        with sqlalchemy.orm.Session(self.engine) as session:
            try:
                with session.begin():

                    # We need the ID of the sample store the operation
                    # belongs to. This is to find all the spaces that
                    # belong to the sample store to see if operations
                    # are currently running on them.
                    sample_store_id = session.execute(
                        sqlalchemy.text(
                            "SELECT data->>'$.config.sampleStoreIdentifier' "
                            "FROM resources "
                            "WHERE identifier = ("
                            "   SELECT subject_identifier"
                            "   FROM resource_relationships"
                            "   WHERE object_identifier=:operation_identifier)"
                        ).bindparams(operation_identifier=identifier)
                    ).first()[0]

                    # The user might choose to ignore running operations
                    # <--------- START CHECKS FOR RUNNING OPERATIONS --------->
                    if not ignore_running_operations:

                        spaces_in_sample_store = session.execute(
                            sqlalchemy.text(
                                "SELECT object_identifier "
                                "FROM resource_relationships "
                                "WHERE subject_identifier=:sample_store_id "
                                "AND object_identifier LIKE 'space-%'"
                            ).bindparams(sample_store_id=sample_store_id)
                        )
                        spaces_in_sample_store = [
                            result[0] for result in spaces_in_sample_store
                        ]

                        running_operations = session.execute(
                            sqlalchemy.text(
                                """
                                SELECT identifier
                                FROM resources
                                WHERE kind = 'operation'
                                    AND JSON_OVERLAPS(data->'$.config.spaces', :spaces_in_sample_store)
                                    AND JSON_CONTAINS(data->'$.status', '{"event":"started"}')
                                    AND NOT JSON_CONTAINS(data->'$.status', '{"event":"finished"}')
                                """
                            ).bindparams(
                                spaces_in_sample_store=json.dumps(
                                    spaces_in_sample_store
                                )
                            )
                        )
                        running_operations = [
                            result[0] for result in running_operations
                        ]

                        if running_operations:
                            raise RunningOperationsPreventingDeletionError(
                                operation_id=identifier,
                                running_operations=running_operations,
                            )

                    # <--------- END CHECKS FOR RUNNING OPERATIONS --------->

                    # We first delete the mappings from the results belonging
                    # to this operation to the requests.
                    # We need to do this before removing the results as we
                    # would otherwise break foreign key constraints
                    session.execute(
                        sqlalchemy.text(
                            f"""
                            WITH
                                operation_result_uids AS (
                                    SELECT result_uid
                                    FROM sqlsource_{sample_store_id}_measurement_requests_results
                                    WHERE request_uid IN (
                                        SELECT uid
                                        FROM sqlsource_{sample_store_id}_measurement_requests
                                        WHERE operation_id = :operation_id
                                    )
                                ),
                                shared_result_uids AS (
                                    SELECT reqres.result_uid
                                    FROM sqlsource_{sample_store_id}_measurement_requests_results reqres
                                    JOIN sqlsource_{sample_store_id}_measurement_requests req
                                         ON reqres.request_uid = req.uid
                                    WHERE reqres.result_uid IN (SELECT result_uid FROM operation_result_uids)
                                        AND req.operation_id != :operation_id
                                )
                            DELETE FROM
                                sqlsource_{sample_store_id}_measurement_requests_results
                            WHERE
                                result_uid IN (SELECT result_uid FROM operation_result_uids)
                                AND result_uid NOT IN (SELECT result_uid FROM shared_result_uids)
                            """  # noqa: S608 - sample store id is not a user input
                        ).bindparams(operation_id=identifier)
                    )

                    # The results that have no link to requests anymore
                    # can now be safely deleted
                    session.execute(
                        sqlalchemy.text(
                            f"""
                            DELETE
                            FROM sqlsource_{sample_store_id}_measurement_results
                            WHERE uid NOT IN (
                                SELECT DISTINCT(result_uid)
                                FROM sqlsource_{sample_store_id}_measurement_requests_results
                            )
                            """  # noqa: S608 - sample store id is not a user input
                        )
                    )

                    # The requests that have no link to results anymore
                    # can now be safely deleted.
                    session.execute(
                        sqlalchemy.text(
                            f"""
                            DELETE
                            FROM sqlsource_{sample_store_id}_measurement_requests
                            WHERE uid NOT IN (
                                SELECT DISTINCT(request_uid)
                                FROM sqlsource_{sample_store_id}_measurement_requests_results
                            )
                            """  # noqa: S608 - sample store id is not a user input
                        )
                    )

                    # We must delete the resource from the relationships table
                    # as we otherwise would break its foreign key constraint
                    session.execute(
                        sqlalchemy.text(
                            "DELETE FROM resource_relationships WHERE object_identifier=:identifier"
                        ).bindparams(identifier=identifier)
                    )

                    # As the last step, we can now delete the operation resource
                    session.execute(
                        sqlalchemy.text(
                            r"DELETE FROM resources "
                            r"WHERE identifier=:identifier AND kind=:kind"
                        ).bindparams(
                            identifier=identifier,
                            kind=CoreResourceKinds.OPERATION.value,
                        )
                    )

            except Exception as e:
                session.rollback()
                raise DeleteFromDatabaseError(
                    resource_id=identifier,
                    resource_kind=CoreResourceKinds.OPERATION,
                    rollback_occurred=True,
                ) from e

    def delete_discovery_space(self, identifier: str):
        import sqlalchemy.orm

        with sqlalchemy.orm.Session(self.engine) as session:
            try:
                with session.begin():

                    session.execute(
                        sqlalchemy.text(
                            r"DELETE FROM resource_relationships WHERE object_identifier=:identifier"
                        ).bindparams(identifier=identifier)
                    )

                    session.execute(
                        sqlalchemy.text(
                            r"DELETE FROM resources "
                            r"WHERE identifier=:identifier AND kind=:kind"
                        ).bindparams(
                            identifier=identifier,
                            kind=CoreResourceKinds.DISCOVERYSPACE.value,
                        )
                    )

            except Exception as e:
                session.rollback()
                raise DeleteFromDatabaseError(
                    resource_id=identifier,
                    resource_kind=CoreResourceKinds.DISCOVERYSPACE,
                    rollback_occurred=True,
                ) from e

    def delete_data_container(self, identifier: str):
        import sqlalchemy.orm

        with sqlalchemy.orm.Session(self.engine) as session:
            try:
                with session.begin():

                    session.execute(
                        sqlalchemy.text(
                            r"DELETE FROM resource_relationships WHERE object_identifier=:identifier"
                        ).bindparams(identifier=identifier)
                    )

                    session.execute(
                        sqlalchemy.text(
                            r"DELETE FROM resources "
                            r"WHERE identifier=:identifier AND kind=:kind"
                        ).bindparams(
                            identifier=identifier,
                            kind=CoreResourceKinds.DATACONTAINER.value,
                        )
                    )

            except Exception as e:
                session.rollback()
                raise DeleteFromDatabaseError(
                    resource_id=identifier,
                    resource_kind=CoreResourceKinds.DATACONTAINER,
                    rollback_occurred=True,
                ) from e

    def delete_actuator_configuration(self, identifier: str):
        import sqlalchemy.orm

        with sqlalchemy.orm.Session(self.engine) as session:
            try:
                with session.begin():

                    session.execute(
                        sqlalchemy.text(
                            r"DELETE FROM resource_relationships WHERE object_identifier=:identifier"
                        ).bindparams(identifier=identifier)
                    )

                    session.execute(
                        sqlalchemy.text(
                            r"DELETE FROM resources "
                            r"WHERE identifier=:identifier AND kind=:kind"
                        ).bindparams(
                            identifier=identifier,
                            kind=CoreResourceKinds.ACTUATORCONFIGURATION.value,
                        )
                    )

            except Exception as e:
                session.rollback()
                raise DeleteFromDatabaseError(
                    resource_id=identifier,
                    resource_kind=CoreResourceKinds.ACTUATORCONFIGURATION,
                    rollback_occurred=True,
                ) from e

    def recordTimeSeriesMetrics(self, df: pd.DataFrame, observedPropertyName: str):

        self.log.warning(
            "SQLResourceStore does not support recording time-series metrics yet. Will write to file."
        )
        name = f"{observedPropertyName}-ts.csv"
        df.to_csv(name, mode="a", header=not os.path.exists(name))
