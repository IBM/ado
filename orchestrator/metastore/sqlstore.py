# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
import logging
import os
from typing import TYPE_CHECKING, Literal

import pydantic
import sqlalchemy

import orchestrator.core
import orchestrator.metastore
import orchestrator.metastore.sql.statements
import orchestrator.utilities
from orchestrator.core.datacontainer.stats import DataContainerStatistics
from orchestrator.core.discoveryspace.stats import DiscoverySpaceStatistics
from orchestrator.core.operation.config import DiscoveryOperationEnum
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
    check_table_exists,
    create_sql_resource_store,
    engine_for_sql_store,
)
from orchestrator.utilities.pydantic import (
    do_not_populate_ado_provenance_context,
    ignore_plugin_validation_context,
    merge_validation_context,
)

if TYPE_CHECKING:
    import pandas as pd

# Cache to track databases where we've verified tables exist
# Key: database connection string, Value: True if tables exist
_tables_exist_cache: dict[str, bool] = {}


class SQLStore(ResourceStore):
    """Base class for SQLStores"""

    def __new__(cls, project_context: ProjectContext) -> "SQLResourceStore":
        import logging

        FORMAT = orchestrator.utilities.logging.FORMAT
        LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
        logging.basicConfig(level=LOGLEVEL, format=FORMAT)
        log = logging.getLogger("SQLStore")

        log.debug("Creating SQL engine...")
        engine = engine_for_sql_store(configuration=project_context.metadataStore)

        # Get cache key from database connection string
        cache_key = (
            project_context.metadataStore.url().unicode_string()
            if project_context.metadataStore.scheme != "sqlite"
            else f"sqlite:///{project_context.metadataStore.path}"
        )

        # Check cache first to avoid network query
        if cache_key in _tables_exist_cache:
            tables_exist = _tables_exist_cache[cache_key]
            log.debug(
                f"Using cached table existence check result: tables_exist={tables_exist}"
            )
        else:
            # Prefer raw SQL via check_table_exists; falls back to inspect on error.
            log.debug("Checking if 'resources' table exists (network query)...")
            tables_exist = check_table_exists(engine, "resources")
            log.debug(f"Table existence check complete: tables_exist={tables_exist}")
            # Cache the result
            _tables_exist_cache[cache_key] = tables_exist

        # We set ensureExists manually by checking just one table.
        return SQLResourceStore(
            project_context=project_context,
            ensureExists=not tables_exist,
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
        self._engine = engine_for_sql_store(configuration=project_context.metadataStore)

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
            # Update cache after creating tables
            cache_key = (
                self.configuration.url().unicode_string()
                if self.configuration.scheme != "sqlite"
                else f"sqlite:///{self.configuration.path}"
            )
            _tables_exist_cache[cache_key] = True
            self.log.debug("Done")

        super().__init__()

    # The SQLAlchemy Engine is not picklable, so anything using
    # Ray would fail. To avoid this, we remove it before pickling
    # and create a new instance when unpickling.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_engine"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._engine = engine_for_sql_store(self.configuration)

    @property
    def engine(self) -> sqlalchemy.Engine:
        return self._engine

    def _deserialize_resource(
        self,
        kind: str,
        data: dict,
        *,
        ignore_plugin_validation: bool = True,
    ) -> orchestrator.core.resources.ADOResource:
        """Deserialize stored JSON into a typed resource model.

        Args:
            kind: Resource kind string from the metastore.
            data: Parsed JSON resource payload.
            ignore_plugin_validation: When True, skip plugin registry validation
                on nested operation and actuator configuration fields.

        Returns:
            Deserialized resource instance.
        """
        custom_model_loader = kind_custom_model_load.get(kind)
        if custom_model_loader:
            return custom_model_loader(data, self.configuration)

        context = merge_validation_context(
            ignore_plugin_validation_context if ignore_plugin_validation else None,
            do_not_populate_ado_provenance_context,
        )
        return orchestrator.core.kindmap[kind].model_validate(data, context=context)

    def get_resource_and_producers(
        self,
        identifier: str,
        kind: CoreResourceKinds,
        chain: list[tuple[str, CoreResourceKinds]],
        raise_error_if_no_resource: bool = False,
    ) -> list[orchestrator.core.resources.ADOResource | None]:
        """Fetch a resource and a chain of producer resources in a single SQL JOIN.

        Each entry in ``chain`` defines how to reach the next resource: the
        JSON path within the *previous* resource's ``data`` column that holds
        the identifier of the next resource, and the ``CoreResourceKinds`` of
        that next resource.  This traverses N hops in one round-trip rather
        than N+1 sequential queries.

        The ``->>`` operator is used for JSON path extraction; it is supported
        by both MySQL and SQLite 3.38+.

        Args:
            identifier: Identifier of the starting resource.
            kind: Kind of the starting resource.
            chain: Ordered list of ``(json_path, kind)`` pairs.  Each pair
                describes one hop: ``json_path`` (e.g.
                ``'$.config.spaces[0]'``) is a JSON path expression in the
                *previous* resource's ``data`` column whose unquoted value is
                the identifier of the next resource.
            raise_error_if_no_resource: When ``True``, raises
                :class:`ResourceDoesNotExistError` if any resource in the
                chain cannot be resolved.

        Returns:
            A list of ``len(chain) + 1`` deserialized
            :class:`~orchestrator.core.resources.ADOResource` instances.  The
            first element corresponds to the starting resource; subsequent
            elements correspond to each hop in ``chain``.

        Raises:
            ResourceDoesNotExistError: When ``raise_error_if_no_resource`` is
                ``True`` and any resource in the chain cannot be found.
        """
        n = len(chain)

        # Build SELECT with explicit per-resource column aliases to avoid
        # collisions when the same column name appears across table aliases.
        selects = ", ".join(
            f"r{i}.data AS r{i}_data, r{i}.kind AS r{i}_kind" for i in range(n + 1)
        )

        # Each JOIN hop resolves the next resource identifier from the JSON
        # field of the previous resource.  Kind filtering is included in the
        # ON clause so an incorrect kind never silently matches.
        joins = "\n".join(
            f"JOIN resources r{i + 1}"
            f"  ON r{i + 1}.identifier = r{i}.data->>'{json_path}'"
            f" AND r{i + 1}.kind = '{linked_kind.value}'"
            for i, (json_path, linked_kind) in enumerate(chain)
        )

        # selects and joins are built from CoreResourceKinds enum values and
        # literal JSON paths only; no user-supplied text is interpolated.
        query = sqlalchemy.text(
            f"SELECT {selects} FROM resources r0 {joins}"  # noqa: S608
            " WHERE r0.identifier = :identifier AND r0.kind = :kind"
        ).bindparams(identifier=identifier, kind=kind.value)

        with self.engine.connect() as connectable:
            row = connectable.execute(query).fetchone()

        if row is None:
            if raise_error_if_no_resource:
                raise ResourceDoesNotExistError(resource_id=identifier, kind=kind)
            return [None] * (n + 1)

        mapping = row._mapping
        all_kinds = [kind] + [k for _, k in chain]
        resources = []
        for i, _rk in enumerate(all_kinds):
            data_raw = mapping[f"r{i}_data"]
            kind_val = mapping[f"r{i}_kind"]
            d = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
            resource = self._deserialize_resource(kind_val, d)

            if orchestrator.core.resources.VersionIsGreaterThan(
                resource.version, d.get("version", "v0")
            ):
                self.updateResource(resource)

            resources.append(resource)

        return resources

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
        ignore_plugin_validation: bool = True,
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
            ignore_plugin_validation: When ``True`` (default), nested operation
                and actuator configuration fields skip plugin registry
                validation during deserialization. Set to ``False`` when
                loading resources for runtime use.

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
            resource = self._deserialize_resource(
                table.kind[0],
                d,
                ignore_plugin_validation=ignore_plugin_validation,
            )

            # The stored resource should always have a version - if somehow it doesn't we want this to fail
            if orchestrator.core.resources.VersionIsGreaterThan(
                resource.version, d.get("version", "v0")
            ):
                self.updateResource(resource)

        if not resource and raise_error_if_no_resource:
            raise ResourceDoesNotExistError(resource_id=identifier, kind=kind)

        return resource

    def getResources(
        self,
        identifiers: list[str],
        ignore_validation_errors: bool = True,
        ignore_plugin_validation: bool = True,
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

        The returned dictionary is sorted by the `created` timestamp of each
        resource in ascending order (oldest first), matching the AGE sorting
        behavior used in CLI commands.

        ``identifiers`` may be passed as either a plain list or a
        :class:`pandas.Series`; if a series is supplied it is converted
        to a list first.

        Args:
            identifiers: The list of resource identifiers to retrieve.
                Duplicate identifiers are ignored.
            ignore_validation_errors: If True (default), resources with validation
                errors are skipped and a warning is logged. If False, ValueError
                is raised when a resource fails validation.

        Returns:
            dict[str, orchestrator.core.resources.ADOResource]:
                A mapping where each key is an identifier found in the
                database and the value is the corresponding deserialized
                resource instance. Resources are ordered by their `created`
                timestamp in ascending order (oldest first). If a
                particular identifier does not exist, it will not appear
                in the returned dictionary.

        Raises:
            ValueError: If ignore_validation_errors is False and a resource
                fails validation.
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
                    try:
                        resource = self._deserialize_resource(
                            kind,
                            d,
                            ignore_plugin_validation=ignore_plugin_validation,
                        )
                    except Exception as error:
                        msg = f"Unable to create pydantic model for resource with id: {identifier} with data: {data}. {error}"
                        if ignore_validation_errors:
                            self.log.warning(msg)
                        else:
                            raise ValueError(msg) from error
                    else:
                        retval[identifier] = resource

        # Sort by resource.created ascending (oldest first, matching AGE sort behavior)
        return dict(sorted(retval.items(), key=lambda item: item[1].created))

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

        # Add the status and space to the resources that have it
        if kind == orchestrator.core.resources.CoreResourceKinds.OPERATION.value:
            select_status = (
                orchestrator.metastore.sql.statements.resource_select_data_field(
                    field_name="status",
                    needs_select=False,
                    dialect=self.engine.dialect.name,
                )
            )
            select_space = (
                orchestrator.metastore.sql.statements.resource_select_data_field(
                    field_name="config.spaces[0]",
                    needs_select=False,
                    dialect=self.engine.dialect.name,
                    output_field_name="space",
                )
            )
            select_statement = f"{select_statement} {select_status} {select_space}"

        # FROM
        from_statement = "FROM resources "

        field_selectors = field_selectors or {}

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
            columns.insert(-1, "SPACE")
            output_df["SPACE"] = table["space"]

        return output_df[columns]

    def get_latest_resource_identifiers_of_kinds(
        self,
        kinds: list[CoreResourceKinds],
    ) -> dict[CoreResourceKinds, str]:
        """Retrieve the identifiers of the most recently created resources for multiple kinds.

        This method executes a single database query to fetch the latest resource
        identifier for each specified kind, minimizing database round-trips.

        Args:
            kinds: List of resource kinds to query

        Returns:
            Dictionary mapping each kind to its most recent resource identifier.
            Kinds with no resources are omitted from the result.

        Raises:
            ValueError: If any supplied kind is not a known CoreResourceKinds value

        Example:
            >>> store.get_latest_resource_identifiers_of_kinds([
            ...     CoreResourceKinds.DISCOVERYSPACE,
            ...     CoreResourceKinds.ACTUATORCONFIGURATION
            ... ])
            {
                CoreResourceKinds.DISCOVERYSPACE: "space-abc123",
                CoreResourceKinds.ACTUATORCONFIGURATION: "actconf-def456"
            }
        """
        if not kinds:
            return {}

        # Validate all kinds are CoreResourceKinds instances
        invalid_kinds = [
            kind for kind in kinds if not isinstance(kind, CoreResourceKinds)
        ]

        if invalid_kinds:
            raise ValueError(
                f"All kinds must be CoreResourceKinds instances. Invalid: {invalid_kinds}"
            )

        # Convert CoreResourceKinds to string values for SQL query
        kind_values = [kind.value for kind in kinds]

        # Generate and execute the SQL query (returns bound TextClause)
        query = orchestrator.metastore.sql.statements.resource_select_latest_by_kinds(
            kinds=kind_values,
            dialect=self.engine.dialect.name,
        )

        with self.engine.connect() as connectable:
            result = connectable.execute(query)
            rows = result.fetchall()

        # Build dictionary mapping kind to identifier
        latest_ids: dict[CoreResourceKinds, str] = {}
        for row in rows:
            identifier, kind_str, _created = row
            # Convert string kind back to CoreResourceKinds enum
            kind_enum = CoreResourceKinds(kind_str)
            latest_ids[kind_enum] = identifier

        return latest_ids

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
        ignore_validation_errors: bool = True,
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
            ignore_validation_errors (bool): If True (default), resources with
                validation errors are skipped and a warning is logged. If False,
                ValueError is raised when a resource fails validation.

        Returns:
            dict[str, orchestrator.core.resources.ADOResource]: A mapping
            where the key is the resource identifier and the value is the
            fully-deserialized :class:`orchestrator.core.resources.ADOResource`
            instance.  An empty dictionary is returned when no matching
            resources are found.

        Raises:
            ValueError: If ``kind`` is not a recognised
                :class:`orchestrator.core.resources.CoreResourceKinds`
                value, or if ignore_validation_errors is False and a resource
                fails validation.

        See Also:
            - getResourceIdentifiersOfKind's documentation
            - https://dev.mysql.com/doc/refman/8.4/en/json-search-functions.html#function_json-contains
        """

        identifiers = self.getResourceIdentifiersOfKind(
            kind=kind, version=version, field_selectors=field_selectors
        )
        return self.getResources(
            identifiers=identifiers["IDENTIFIER"],
            ignore_validation_errors=ignore_validation_errors,
        )

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
                            "   WHERE object_identifier=:operation_identifier"
                            "   AND subject_identifier LIKE 'space-%')"
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

    # ---------------------------------------------------------------------------
    # Hierarchy traversal
    # ---------------------------------------------------------------------------

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
        | dict[CoreResourceKinds, dict[str, "orchestrator.core.resources.ADOResource"]]
        | dict[
            str,
            dict[
                CoreResourceKinds,
                dict[str, "orchestrator.core.resources.ADOResource"],
            ],
        ]
    ):
        """Walk the resource hierarchy stored in ``resource_relationships``.

        Issues at most three SQL queries: when ``identifier=None`` a seed query
        fetches all identifiers of ``kind`` via
        :meth:`getResourceIdentifiersOfKind`; then one recursive traversal query
        via :func:`orchestrator.metastore.sql.statements.graph_traversal_query`;
        and, when ``identifiers_only=False``, one additional batched resource
        query via :meth:`getResources`. When ``identifier`` is a ``str`` or
        ``set[str]`` only the latter two queries (or one, if
        ``identifiers_only=True``) are issued.

        Args:
            kind: The :class:`~orchestrator.core.resources.CoreResourceKinds` of
                the starting resources.
            identifier: Controls which resources are used as traversal origins.

                * ``str`` — a single start resource identifier; the return value
                  is unwrapped (no outer origin key).
                * ``set[str]`` — multiple explicit start resource identifiers.
                * ``None`` — all resources of ``kind`` are used as start
                  resources (seeded via :meth:`getResourceIdentifiersOfKind`).
                  Not supported when ``hierarchy_direction='both'``.
                * An **empty set** returns an empty result immediately.

            hierarchy_direction: ``'up'`` (child → parent), ``'down'``
                (parent → child), or ``'both'``.
            max_hops: Maximum number of relationship hops to follow from each
                start resource. When ``None`` the traversal runs to the full
                depth of the hierarchy. For ``hierarchy_direction='both'`` the
                limit is applied independently to each direction (e.g.
                ``max_hops=1`` yields one hop up *and* one hop down). Values
                exceeding the hierarchy maximum (currently 3, matching the 4
                resource levels) are silently capped at that maximum.
            identifiers_only: When ``False`` (default) discovered identifiers
                are hydrated into full
                :class:`~orchestrator.core.resources.ADOResource` objects via
                :meth:`getResources`. When ``True`` only discovered identifiers
                are returned.
            include_start_resources: When ``True``, the start resource(s)
                provided via ``identifier`` are included in the returned result
                under their own ``kind`` key, alongside the discovered related
                resources. Requires ``identifiers_only=False`` and
                ``identifier`` to be a ``str`` or ``set[str]`` (not ``None``);
                raises ``ValueError`` if either constraint is violated.

        Returns:
            The return type depends on whether a single identifier (``str``) or
            multiple identifiers (``set`` / ``None``) were requested, and
            whether ``identifiers_only`` is set:

            * single identifier, hydrated    → ``dict[CoreResourceKinds, dict[str, ADOResource]]``
            * multiple identifiers, hydrated → ``dict[str, dict[CoreResourceKinds, dict[str, ADOResource]]]``
            * single identifier, ids only    → ``dict[CoreResourceKinds, set[str]]``
            * multiple identifiers, ids only → ``dict[str, dict[CoreResourceKinds, set[str]]]``

            By default start identifiers are **excluded** from the returned
            results. Pass ``include_start_resources=True`` to include them.

        Raises:
            ValueError: If ``hierarchy_direction`` is not ``'up'``, ``'down'``
                or ``'both'``.
            ValueError: If ``identifier=None`` is used with
                ``hierarchy_direction='both'``.
            ValueError: If ``include_start_resources=True`` is used together
                with ``identifiers_only=True``.
            ValueError: If ``include_start_resources=True`` is used with
                ``identifier=None``.
        """
        # ------------------------------------------------------------------
        # 0. Validate parameters eagerly
        # ------------------------------------------------------------------
        if hierarchy_direction not in {"up", "down", "both"}:
            raise ValueError(
                f"hierarchy_direction must be 'up', 'down' or 'both', got {hierarchy_direction!r}"
            )

        if max_hops is not None and max_hops < 1:
            raise ValueError(f"max_hops must be a positive integer, got {max_hops!r}")

        if include_start_resources and identifiers_only:
            raise ValueError(
                "include_start_resources=True requires identifiers_only=False"
            )

        if include_start_resources and identifier is None:
            raise ValueError(
                "include_start_resources=True requires identifier to be a str or set[str], not None"
            )

        if identifier is None and hierarchy_direction == "both":
            raise ValueError(
                "identifier=None is not supported for hierarchy_direction='both'"
            )

        # ------------------------------------------------------------------
        # 1. Resolve the requested identifiers and record whether a single
        #    identifier was requested (determines the unwrapped return shape)
        # ------------------------------------------------------------------
        _single_identifier_requested: bool
        _identifiers_requested: set[str]

        if identifier is None:
            _single_identifier_requested = False
            df = self.getResourceIdentifiersOfKind(kind=kind.value)
            _identifiers_requested = set(df["IDENTIFIER"].tolist())
        elif isinstance(identifier, str):
            _single_identifier_requested = True
            _identifiers_requested = {identifier}
        else:
            # set[str]
            _single_identifier_requested = False
            _identifiers_requested = identifier

        # Empty identifier set → immediate empty result
        if not _identifiers_requested:
            return {}

        # ------------------------------------------------------------------
        # 2. Build and execute the single traversal query
        # ------------------------------------------------------------------
        # The hierarchy maximum (3 hops across 4 levels) is enforced inside
        # graph_traversal_query; passing max_hops=None lets it use the full cap.
        query = orchestrator.metastore.sql.statements.graph_traversal_query(
            kind=kind,
            hierarchy_direction=hierarchy_direction,
            origin_identifiers=_identifiers_requested,
            max_hops=max_hops,
        )

        with self.engine.connect() as connectable:
            raw_rows = connectable.execute(query).fetchall()

        # ------------------------------------------------------------------
        # 3. Build the mapping
        #    { origin_id -> { CoreResourceKinds -> {related_id, ...} } }
        # ------------------------------------------------------------------
        related_by_origin: dict[str, dict[CoreResourceKinds, set[str]]] = {}
        identifiers_to_fetch: set[str] = set()

        for row in raw_rows:
            origin_identifier = row.origin_identifier
            identifier_to = row.identifier
            identifier_to_kind = row.kind

            # Don't include the start identifiers in discovered results
            # This should never happen, if it does, we have a bug.
            if identifier_to in _identifiers_requested:
                continue

            identifiers_to_fetch.add(identifier_to)
            resource_kind = CoreResourceKinds(identifier_to_kind)
            related_by_origin.setdefault(origin_identifier, {}).setdefault(
                resource_kind, set()
            ).add(identifier_to)

        # ------------------------------------------------------------------
        # 4. Shape the result
        # ------------------------------------------------------------------
        if identifiers_only:
            if _single_identifier_requested:
                return related_by_origin.get(next(iter(_identifiers_requested)), {})
            return related_by_origin

        # Hydrated mode: fetch all discovered identifiers in one query,
        # then rebuild the graph with full resources.
        # When include_start_resources is True, also fetch the start resources.
        if include_start_resources:
            identifiers_to_fetch = identifiers_to_fetch.union(_identifiers_requested)

        resources = self.getResources(identifiers=list(identifiers_to_fetch))

        hydrated: dict[
            str,
            dict[CoreResourceKinds, dict[str, orchestrator.core.resources.ADOResource]],
        ] = {}

        for origin_identifier, related_identifiers_by_kind in related_by_origin.items():

            hydrated_related_resources_by_kind: dict[
                CoreResourceKinds,
                dict[str, orchestrator.core.resources.ADOResource],
            ] = {}

            for (
                resource_kind,
                related_identifiers,
            ) in related_identifiers_by_kind.items():

                hydrated_related_resources_by_kind[resource_kind] = {
                    identifier: resources[identifier]
                    for identifier in related_identifiers
                    if identifier in resources
                }

            if include_start_resources and origin_identifier in resources:
                start_resource = resources[origin_identifier]
                hydrated_related_resources_by_kind.setdefault(kind, {})[
                    origin_identifier
                ] = start_resource

            if hydrated_related_resources_by_kind:
                hydrated[origin_identifier] = hydrated_related_resources_by_kind

        # When include_start_resources is True but a start identifier had no
        # related resources, it won't appear in related_by_origin yet — ensure
        # it still gets an entry in hydrated.
        if include_start_resources:
            for start_id in _identifiers_requested:
                if start_id not in hydrated and start_id in resources:
                    hydrated[start_id] = {kind: {start_id: resources[start_id]}}

        if _single_identifier_requested:
            return hydrated.get(next(iter(_identifiers_requested)), {})

        return hydrated

    # ---------------------------------------------------------------------------
    # Space statistics
    # ---------------------------------------------------------------------------

    def get_space_metastore_stats(
        self,
        space_ids: str | set[str],
    ) -> "DiscoverySpaceStatistics | dict[str, DiscoverySpaceStatistics]":
        """Return lightweight metastore-level statistics for one or many spaces.

        Issues a single SQL query that anchors on each space's ``resources``
        row and left-joins to ``resource_relationships`` / ``resources`` to
        count operations.  The experiment count and operation counts are
        therefore fetched in one round-trip.

        Args:
            space_ids: A single space identifier (``str``) or a set of space
                identifiers (``set[str]``).

        Returns:
            :class:`~orchestrator.core.discoveryspace.stats.DiscoverySpaceStatistics`
            for a single ``str`` input, or a
            ``dict[str, DiscoverySpaceStatistics]`` for a ``set[str]`` input.

        Raises:
            SystemError: If the underlying SQL query fails.
        """
        single = isinstance(space_ids, str)
        _space_ids: set[str] = {space_ids} if single else set(space_ids)

        if not _space_ids:
            return {}  # type: ignore[return-value]

        # ------------------------------------------------------------------
        # Single query: anchor on the space row so every requested space is
        # returned even when it has no operations (LEFT JOIN).
        # The experiment list lives at $.config.experiments.experiments inside
        # the space's own resources.data column.
        # MySQL uses JSON_LENGTH(); SQLite uses json_array_length().
        # ------------------------------------------------------------------
        is_sqlite = self.engine.dialect.name == "sqlite"
        array_length_fn = "json_array_length" if is_sqlite else "JSON_LENGTH"

        query_text = f"""
            SELECT
                sp.identifier AS space_id,
                COALESCE({array_length_fn}(
                    JSON_EXTRACT(sp.data, '$.config.experiments.experiments')
                ), 0) AS num_experiments,
                COUNT(op.identifier) AS total_operations,
                COUNT(
                    CASE
                        WHEN JSON_EXTRACT(op.data, '$.operationType') = :explore_type
                        THEN 1
                    END
                ) AS explore_operations
            FROM resources sp
            LEFT JOIN resource_relationships rr
                ON rr.subject_identifier = sp.identifier
            LEFT JOIN resources op
                ON op.identifier = rr.object_identifier
                AND op.kind = :op_kind
            WHERE sp.identifier IN :space_ids
            GROUP BY sp.identifier, sp.data
        """  # noqa: S608 - identifier is an internal column name, not untrusted input

        try:
            with self.engine.begin() as conn:
                query = sqlalchemy.text(query_text).bindparams(
                    sqlalchemy.bindparam("space_ids", expanding=True),
                    space_ids=list(_space_ids),
                    explore_type=DiscoveryOperationEnum.SEARCH.value,
                    op_kind=CoreResourceKinds.OPERATION.value,
                )

                rows = {row.space_id: row for row in conn.execute(query)}

        except Exception as error:
            msg = f"Unable to get statistics for space(s) {space_ids}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        result: dict[str, DiscoverySpaceStatistics] = {
            sid: DiscoverySpaceStatistics(
                number_of_experiments=rows[sid].num_experiments if sid in rows else 0,
                number_of_operations=rows[sid].total_operations if sid in rows else 0,
                number_of_explore_operations=(
                    rows[sid].explore_operations if sid in rows else 0
                ),
                number_measured_entities=0,
            )
            for sid in _space_ids
        }

        if single:
            return result[space_ids]  # type: ignore[index,arg-type]

        return result

    # ---------------------------------------------------------------------------
    # DataContainer statistics
    # ---------------------------------------------------------------------------

    def get_datacontainer_stats(
        self,
        datacontainer_ids: set[str],
    ) -> dict[str, DataContainerStatistics]:
        """Return lightweight statistics for a set of DataContainer IDs.

        Args:
            datacontainer_ids: A set of DataContainer identifiers to query.

        Returns:
            A ``dict`` keyed by DataContainer ID mapping each to its
            :class:`~orchestrator.core.datacontainer.stats.DataContainerStatistics`.
            IDs that are not present in the database are returned with all-zero
            stats.  An empty input set returns an empty dict immediately (no
            query issued).

        Raises:
            SystemError: If the underlying SQL query fails.
        """
        if not datacontainer_ids:
            return {}

        # MySQL uses JSON_LENGTH() which counts object members correctly.
        # SQLite's json_array_length() only counts array elements and returns 0
        # for objects, so we use correlated subqueries with json_each() instead.
        is_sqlite = self.engine.dialect.name == "sqlite"

        if is_sqlite:
            query_text = """
                SELECT
                    identifier,
                    (SELECT count(*)
                     FROM json_each(json_extract(data, '$.config.tabularData'))
                    ) AS num_tables,
                    (SELECT count(*)
                     FROM json_each(json_extract(data, '$.config.locationData'))
                    ) AS num_locations,
                    (SELECT count(*)
                     FROM json_each(json_extract(data, '$.config.data'))
                    ) AS num_key_values,
                    COALESCE(
                        LENGTH(JSON_EXTRACT(data, '$.config'))
                        - LENGTH(JSON_EXTRACT(data, '$.config.metadata')),
                        0
                    ) AS data_bytes
                FROM resources
                WHERE identifier IN :ids
            """
        else:
            # On MySQL, JSON_LENGTH of a JSON null scalar returns 1 (scalar
            # length is 1 per the spec).  We must guard with JSON_TYPE to
            # return 0 for absent/null fields.
            # For byte count, JSON_STORAGE_SIZE returns the actual binary
            # storage size of the JSON value, which is more accurate than
            # LENGTH(JSON_EXTRACT(...)) (text representation length).
            query_text = """
                SELECT
                    identifier,
                    IF(JSON_TYPE(data->'$.config.tabularData') = 'NULL',
                       0, COALESCE(JSON_LENGTH(
                           data->'$.config.tabularData'
                       ), 0)) AS num_tables,
                    IF(JSON_TYPE(data->'$.config.locationData') = 'NULL',
                       0, COALESCE(JSON_LENGTH(
                           data->'$.config.locationData'
                       ), 0)) AS num_locations,
                    IF(JSON_TYPE(data->'$.config.data') = 'NULL',
                       0, COALESCE(JSON_LENGTH(
                           data->'$.config.data'
                       ), 0)) AS num_key_values,
                    COALESCE(
                        JSON_STORAGE_SIZE(JSON_EXTRACT(data, '$.config'))
                        - JSON_STORAGE_SIZE(JSON_EXTRACT(data, '$.config.metadata')),
                        0
                    ) AS data_bytes
                FROM resources
                WHERE identifier IN :ids
            """

        try:
            with self.engine.begin() as conn:
                query = sqlalchemy.text(query_text).bindparams(
                    sqlalchemy.bindparam("ids", expanding=True),
                    ids=list(datacontainer_ids),
                )
                rows_by_id = {row.identifier: row for row in conn.execute(query)}
        except Exception as error:
            msg = f"Unable to get statistics for datacontainer(s) {datacontainer_ids}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        empty_stats = DataContainerStatistics(
            number_of_tables=0,
            number_of_locations=0,
            number_of_key_values=0,
            total_data_bytes=0,
        )

        return {
            container_id: (
                DataContainerStatistics(
                    number_of_tables=row.num_tables,
                    number_of_locations=row.num_locations,
                    number_of_key_values=row.num_key_values,
                    total_data_bytes=row.data_bytes,
                )
                if (row := rows_by_id.get(container_id)) is not None
                else empty_stats
            )
            for container_id in datacontainer_ids
        }
