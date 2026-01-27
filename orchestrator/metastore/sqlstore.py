# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import logging
import os
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import pandas as pd


class SQLStore(ResourceStore):
    """Base class for SQLStores"""

    def __new__(cls, project_context: ProjectContext) -> "SQLResourceStore":

        engine = engine_for_sql_store(configuration=project_context.metadataStore)
        inspector = sqlalchemy.inspect(engine)

        # We set ensureExists manually by checking just one table.
        return SQLResourceStore(
            project_context=project_context,
            ensureExists=not inspector.has_table("resources"),
        )

    def __init__(self, project_context: ProjectContext) -> None:

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

    def __init__(
        self, project_context: ProjectContext, ensureExists: bool = True
    ) -> None:
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
            self.log.debug("Initialising SQL db if it does not exist")
            create_sql_resource_store(self.engine)
            self.log.debug("Done")

        super().__init__()

    @property
    def engine(self) -> sqlalchemy.Engine:

        return engine_for_sql_store(configuration=self.configuration)

    def getResourceRaw(self, identifier: str) -> dict | None:
        """Retrieve the raw JSON data for a resource.

        The method queries the ``resources`` table for a row with the
        specified ``identifier``.  The `data` column holds a JSON string
        representing the resource, which is deserialized and returned as
        a Python ``dict``.  If the identifier is not present in the
        database, the method returns ``None`` instead of raising an
        exception.

        Args:
            identifier: The unique identifier of the resource to fetch.

        Returns:
            dict | None: The deserialized JSON object stored in the
                database for the given identifier, or ``None`` when no
                matching record is found.

        Note:
            This method does **not** perform any validation against the
            resource schema - callers should use :meth:`getResource` if they
            need a fully-typed object.
        """
        import pandas as pd

        query = sqlalchemy.text(
            "SELECT * FROM resources WHERE identifier=:identifier"
        ).bindparams(identifier=identifier)

        with self.engine.connect() as connectable:
            table = pd.read_sql(query, con=connectable)

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
        """Retrieve a resource from the SQL store.

        This method selects the resource with the given *identifier* and
        *kind* from the ``resources`` table.  The JSON payload stored in
        the database is deserialized and converted into the appropriate
        :class:`~orchestrator.core.resources.ADOResource` subclass.

        If the stored version is older than the resource instance being
        retrieved (`resource.version`) the object is automatically updated
        in the database.

        Args:
            identifier: The unique identifier of the resource to fetch.
            kind: The :class:`~orchestrator.core.resources.CoreResourceKinds`
                enum value that specifies the expected resource kind.
            raise_error_if_no_resource: If ``True``, a
                :class:`~orchestrator.metastore.base.ResourceDoesNotExistError`
                is raised when the resource cannot be found.  When ``False``
                (default) the method simply returns ``None``.

        Returns:
            An instance of the appropriate
            :class:`~orchestrator.core.resources.ADOResource` subclass if the
            resource was found; otherwise ``None`` when
            ``raise_error_if_no_resource`` is ``False``.

        Raises:
            ResourceDoesNotExistError:
                If the resource is not located in the database and the
                *raise_error_if_no_resource* flag is ``True``.

        Notes:
            * The database uses SQLAlchemy under the hood, and the query
              result is loaded into a :class:`pandas.DataFrame` before the
              JSON column is parsed.
            * Custom load functions registered in
              ``kind_custom_model_load`` are used when available; otherwise
              the default Pydantic model from ``orchestrator.core.kindmap``
              is instantiated.
        """

        import pandas as pd

        query = sqlalchemy.text("""
            SELECT * FROM resources
            WHERE identifier=:identifier
            AND kind=:kind
            """).bindparams(identifier=identifier, kind=kind.value)

        with self.engine.connect() as connectable:
            table = pd.read_sql(query, con=connectable)

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
        """Retrieve multiple resources by identifier.

        This method queries the `resources` table for all rows whose
        ``identifier`` column matches an element of *identifiers*.  The
        JSON payload stored in the `data` column is deserialized and
        converted into the appropriate :class:`orchestrator.core.resources.ADOResource`
        subclass.  The resulting objects are returned in a dictionary that maps each
        identifier to its `ADOResource` instance.  Identifiers that
        are not present in the database are simply omitted from the
        returned mapping.

        ``identifiers`` may be passed as either a plain list or a
        :class:`pandas.Series`; if a series is supplied it is converted
        to a list first.

        Args:
            identifiers: The list of resource identifiers to retrieve.
                Duplicate identifiers are ignored.

        Returns:
            dict[str, orchestrator.core.resources.ADOResource]:
                A mapping where each key is an identifier found in the
                database and the value is the corresponding deserialized
                resource instance. If a particular identifier does not
                exist, it will not appear in the returned dictionary.
        """

        import pandas as pd

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
                    table.identifier, table.data, table.kind, strict=True
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
    ) -> "pd.DataFrame":
        """
        Retrieve identifiers of resources of a given kind.

        This method queries the ``resources`` table to return identifiers and
        selected metadata for all resources that match the specified ``kind``.
        Optionally, a version and a list of JSON field selectors may be
        provided to further refine the results.  By default the returned
        dataframe contains only the identifier, name and age of each
        resource.  When ``details=True`` the returned dataframe also
        includes the description, labels, and, for operation resources,
        the current status.

        Args:
            kind (str):
                The kind of resource to filter on.  Must be a value from
                :class:`orchestrator.core.resources.CoreResourceKinds`.
            version (str | None, optional):
                When provided only resources with this exact version are
                returned.  Set to ``None`` to ignore the version filter.
            field_selectors (list[dict[str, str]] | None, optional):
                A list of dictionaries used to filter on JSON fields. Each
                dictionary maps a MySQL JSON path (e.g. ``"$.config.owner"``)
                to the value the field must contain. The matcher uses
                ``JSON_CONTAINS`` under the hood and is subject to its
                restrictions listed at:
                https://dev.mysql.com/doc/refman/8.4/en/json-search-functions.html#function_json-contains.
            details (bool, optional):
                If ``True`` the dataframe will contain extra columns
                (``DESCRIPTION``, ``LABELS`` and, for operations, ``STATUS``).
                Defaults to ``False`` for a lightweight payload.

        Field Selectors:
            - The keys of the dictionaries are MySQL JSON paths as defined in:
            https://dev.mysql.com/doc/refman/8.4/en/json.html#json-path-syntax,
            with some additional limitations as per the documentation from JSON_CONTAINS:
            https://dev.mysql.com/doc/refman/8.4/en/json-search-functions.html#function_json-contains.
            Notably, single (*) and double-asterisk (**) wildcards are not supported.
            - The values can be any valid JSON documents (including plain strings, etc.)

            In practical terms, this means that, when searching for objects within arrays we
            should use document matching instead of wildcard-based value matching.

            DO NOT: {"config.experiments[*].experiments.identifier": "my-experiment"}
            DO: {"config.experiments": {"experiments":{"identifier":"my-experiment"}}}

        Returns:
            pandas.DataFrame:
                A dataframe containing the selected columns.  When
                ``details`` is ``False`` the columns are ``IDENTIFIER``,
                ``NAME`` and ``AGE``.  When ``details`` is ``True`` the
                columns become ``IDENTIFIER``, ``NAME``, ``DESCRIPTION``,
                ``LABELS`` and ``AGE``; for operation resources an
                additional ``STATUS`` column is appended.  If
                ``field_selectors`` or ``version`` exclude all rows the
                dataframe is empty.

        Raises:
            ValueError:
                If the supplied ``kind`` is not a known
                ``CoreResourceKinds`` value.
        """

        import pandas as pd

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

    def resourceTable(self) -> "pd.DataFrame":
        import pandas as pd

        query = """SELECT * FROM resources"""

        with self.engine.connect() as connectable:
            return pd.read_sql(query, con=connectable)

    def getResourcesOfKind(
        self,
        kind: str,
        version: str | None = None,
        field_selectors: list[dict[str, str]] | None = None,
    ) -> dict[str, orchestrator.core.resources.ADOResource]:
        """
        Retrieve all resources of a given kind.

        The method first obtains the identifiers of matching resources by
        calling :meth:`getResourceIdentifiersOfKind`. The identifiers are
        then used to fetch the full resource objects via
        :meth:`getResources`.

        Args:
            kind (str): The kind of resources to fetch. Must be one of
                :class:`orchestrator.core.resources.CoreResourceKinds`.
            version (str, optional): If supplied, only resources with this
                exact version are returned.
            field_selectors (list[dict[str, str]], optional): A list of
                JSON-field selectors used to narrow the result set.  Each
                selector maps a MySQL JSON path (e.g. ``"$.config.owner"``)
                to the value the field must contain.

        Returns:
            dict[str, orchestrator.core.resources.ADOResource]: A mapping
            where the key is the resource identifier and the value is the
            fully-deserialized :class:`orchestrator.core.resources.ADOResource`
            instance.  An empty dictionary is returned when no matching
            resources are found.

        Raises:
            ValueError: If ``kind`` is not a recognised
                :class:`orchestrator.core.resources.CoreResourceKinds`
                value.

        See Also:
            - getResourceIdentifiersOfKind's documentation
            - https://dev.mysql.com/doc/refman/8.4/en/json-search-functions.html#function_json-contains
        """

        identifiers = self.getResourceIdentifiersOfKind(
            kind=kind, version=version, field_selectors=field_selectors
        )
        return self.getResources(identifiers=identifiers["IDENTIFIER"])

    def getRelatedSubjectResourceIdentifiers(
        self, identifier: str, kind: str | None = None, version: str | None = None
    ) -> "pd.DataFrame":
        """Retrieve identifiers of resources that have a relationship to the
        supplied ``identifier`` where that identifier acts as the *object*.

        The method queries the ``resource_relationships`` table and returns
        a ``pandas.DataFrame`` containing identifiers of all resources that
        are the *subject* of a relationship whose *object* is the supplied
        ``identifier``.  Optional filtering by the other resource's
        ``kind`` or ``version`` is supported.

        Args:
            identifier (str):
                The resource identifier that will be queried as the object
                side of the relationship.
            kind (str | None, optional):
                If provided, only resources whose ``kind`` matches this
                value will be returned.  Pass ``None`` to ignore the kind
                filter.
            version (str | None, optional):
                If provided, only resources whose ``version`` matches this
                value will be returned.  Pass ``None`` to ignore the
                version filter.

        Returns:
            pandas.DataFrame:
                A two-column dataframe with the columns ``IDENTIFIER`` and
                ``TYPE``.  ``IDENTIFIER`` is the identifier of a resource
                that is the subject of a relationship, and ``TYPE`` is its
                ``kind``.  If no related resources are found an empty
                dataframe is returned.

        Raises:
            sqlalchemy.exc.SQLAlchemyError:
                Propagated if the underlying database query fails.

        See Also:
            getRelatedObjectResourceIdentifiers
                The inverse relationship: fetches subjects where the given
                identifier is the *subject*.
            getRelatedResourceIdentifiers
                Convenience wrapper that returns a dataframe with both subject
                and object relationships merged.
        """

        import pandas as pd

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
        self, identifier: str, kind: str | None = None, version: str | None = None
    ) -> "pd.DataFrame":
        """Retrieve identifiers of resources that have a relationship to the
        supplied ``identifier`` where that identifier acts as the *subject*.

        The method queries the ``resource_relationships`` table and returns
        a ``pandas.DataFrame`` containing identifiers of all resources that
        are the *object* of a relationship whose *subject* is the supplied
        ``identifier``.  Optional filtering by the other resource's
        ``kind`` or ``version`` is supported.

        Args:
            identifier (str):
                The resource identifier that will be queried as the subject
                side of the relationship.
            kind (str | None, optional):
                If provided, only resources whose ``kind`` matches this
                value will be returned.  Pass ``None`` to ignore the kind
                filter.
            version (str | None, optional):
                If provided, only resources whose ``version`` matches this
                value will be returned.  Pass ``None`` to ignore the
                version filter.

        Returns:
            pandas.DataFrame:
                A two-column dataframe with the columns ``IDENTIFIER`` and
                ``TYPE``.  ``IDENTIFIER`` is the identifier of a resource
                that is the object of a relationship, and ``TYPE`` is its
                ``kind``.  If no related resources are found an empty
                dataframe is returned.

        Raises:
            sqlalchemy.exc.SQLAlchemyError:
                Propagated if the underlying database query fails.

        See Also:
            getRelatedSubjectResourceIdentifiers
                The inverse relationship: fetches subjects where the given
                identifier is the *object*.
            getRelatedResourceIdentifiers
                Convenience wrapper that returns a dataframe with both subject
                and object relationships merged.
        """

        import pandas as pd

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
        self, identifier: str, kind: str | None = None, version: str | None = None
    ) -> "pd.DataFrame":
        """
        Retrieve identifiers of resources that are related to ``identifier`` either as a
        subject or an object.

        This method concatenates the results of
        :meth:`getRelatedObjectResourceIdentifiers` and
        :meth:`getRelatedSubjectResourceIdentifiers`.  The returned
        :class:`pandas.DataFrame` has two columns:

        * ``IDENTIFIER`` - the resource identifier
        * ``TYPE``      - the resource kind

        Args:
            identifier : str
                The resource identifier for which related resources are being
                queried.
            kind : str, optional
                Filter by the resource *kind*.  If ``None`` (default) no kind
                filtering is applied.
            version : str, optional
                Filter by the resource *version*.  If ``None`` (default) no
                version filtering is applied.

        Returns:
            pandas.DataFrame
                A DataFrame containing the identifiers of all related resources.
                If no relationships exist an empty DataFrame is returned.
        """

        import pandas as pd

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
        Retrieve all resources that are related to a given identifier.

        Args:
            identifier (str):
                The identifier of the primary resource.  The method will fetch
                every other resource that shares a relationship with this
                identifier - either as the **subject** or **object** of a
                relationship entry in ``resource_relationships``.
            kind (orchestrator.core.resources.CoreResourceKinds, optional):
                If supplied, only resources whose ``kind`` matches this value
                are returned.  Pass ``None`` (the default) to retrieve
                resources of any kind.

        Returns:
            dict[str, orchestrator.core.resources.ADOResource]:
                A mapping from resource identifier to a fully deserialized
                ``ADOResource`` instance.  The dictionary keys are the
                identifiers of all resources that are related to
                ``identifier``; the values are the corresponding
                resource objects.  When ``kind`` is set, the dictionary
                contains only resources of that kind.
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

    def addResource(self, resource: orchestrator.core.resources.ADOResource) -> None:

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
    ) -> None:

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
    ) -> None:

        self.addRelationship(
            subjectIdentifier=subjectResource.identifier,
            objectIdentifier=objectResource.identifier,
        )

    def addResourceWithRelationships(
        self,
        resource: orchestrator.core.resources.ADOResource,
        relatedIdentifiers: list,
    ) -> None:
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

    def updateResource(self, resource: orchestrator.core.resources.ADOResource) -> None:
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

    def deleteResource(self, identifier: str) -> None:

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

    def deleteObjectRelationships(self, identifier: str) -> None:
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

    def delete_sample_store(
        self, identifier: str, force_deletion: bool = False
    ) -> None:
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
    ) -> None:
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
                            sqlalchemy.text("""
                                SELECT identifier
                                FROM resources
                                WHERE kind = 'operation'
                                    AND JSON_OVERLAPS(data->'$.config.spaces', :spaces_in_sample_store)
                                    AND JSON_CONTAINS(data->'$.status', '{"event":"started"}')
                                    AND NOT JSON_CONTAINS(data->'$.status', '{"event":"finished"}')
                                """).bindparams(
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
                    session.execute(sqlalchemy.text(f"""
                            DELETE
                            FROM sqlsource_{sample_store_id}_measurement_results
                            WHERE uid NOT IN (
                                SELECT DISTINCT(result_uid)
                                FROM sqlsource_{sample_store_id}_measurement_requests_results
                            )
                            """))  # noqa: S608 - sample store id is not a user input

                    # The requests that have no link to results anymore
                    # can now be safely deleted.
                    session.execute(sqlalchemy.text(f"""
                            DELETE
                            FROM sqlsource_{sample_store_id}_measurement_requests
                            WHERE uid NOT IN (
                                SELECT DISTINCT(request_uid)
                                FROM sqlsource_{sample_store_id}_measurement_requests_results
                            )
                            """))  # noqa: S608 - sample store id is not a user input

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

    def delete_discovery_space(self, identifier: str) -> None:
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

    def delete_data_container(self, identifier: str) -> None:
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

    def delete_actuator_configuration(self, identifier: str) -> None:
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
