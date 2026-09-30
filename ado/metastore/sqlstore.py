# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
import logging
import os
from typing import TYPE_CHECKING, Literal

import pydantic
import sqlalchemy

import ado.core
import ado.metastore
import ado.metastore.sql.statements
import ado.utilities
from ado.core.datacontainer.stats import DataContainerStatistics
from ado.core.discoveryspace.stats import DiscoverySpaceStatistics
from ado.core.operation.config import DiscoveryOperationEnum
from ado.core.resources import ADOResourceEventEnum, CoreResourceKinds
from ado.metastore.base import (
    DeleteFromDatabaseError,
    NonEmptySampleStorePreventingDeletionError,
    NotSupportedOnSQLiteError,
    ResourceDoesNotExistError,
    ResourceHasChildrenError,
    ResourceStore,
    RunningOperationsPreventingDeletionError,
    kind_custom_model_dump,
    kind_custom_model_load,
)
from ado.metastore.project import ProjectContext
from ado.metastore.sql.utils import (
    check_table_exists,
    create_sql_resource_store,
    engine_for_sql_store,
    json_extract_field_as_string,
)
from ado.utilities.pydantic import (
    do_not_populate_ado_provenance_context,
    ignore_plugin_validation_context,
    merge_validation_context,
)

if TYPE_CHECKING:
    import pandas as pd

# Key: database connection string, Value: True if tables exist
_tables_exist_cache: dict[str, bool] = {}

# Key: engine URL string, Value: sqlalchemy.MetaData with reflected resources tables
_reflected_metadata_cache: dict[str, sqlalchemy.MetaData] = {}


class SQLStore(ResourceStore):
    """Base class for SQLStores"""

    def __new__(cls, project_context: ProjectContext) -> "SQLResourceStore":
        import logging

        FORMAT = ado.utilities.logging.FORMAT
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

        FORMAT = ado.utilities.logging.FORMAT
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
            cache_key = (
                self.configuration.url().unicode_string()
                if self.configuration.scheme != "sqlite"
                else f"sqlite:///{self.configuration.path}"
            )
            _tables_exist_cache[cache_key] = True
            self.log.debug("Done")

        self._reflect_tables()
        super().__init__()

    def _reflect_tables(self) -> None:
        """Reflect the resources and resource_relationships tables, caching the metadata by engine URL."""
        cache_key = str(self._engine.url)
        if cache_key not in _reflected_metadata_cache:
            metadata = sqlalchemy.MetaData()
            metadata.reflect(
                bind=self._engine, only=["resources", "resource_relationships"]
            )
            _reflected_metadata_cache[cache_key] = metadata
        metadata = _reflected_metadata_cache[cache_key]
        self._resources_table = metadata.tables["resources"]
        self._relationships_table = metadata.tables["resource_relationships"]

    # The SQLAlchemy Engine is not picklable, so anything using
    # Ray would fail. To avoid this, we remove it before pickling
    # and create a new instance when unpickling.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_engine"]
        del state["_resources_table"]
        del state["_relationships_table"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._engine = engine_for_sql_store(self.configuration)
        self._reflect_tables()

    @property
    def engine(self) -> sqlalchemy.Engine:
        return self._engine

    def _deserialize_resource(
        self,
        kind: str,
        data: dict,
        *,
        ignore_plugin_validation: bool = True,
    ) -> ado.core.resources.ADOResource:
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
        return ado.core.kindmap[kind].model_validate(data, context=context)

    def get_resource_and_producers(
        self,
        identifier: str,
        kind: CoreResourceKinds,
        chain: list[tuple[str, CoreResourceKinds]],
        raise_error_if_no_resource: bool = False,
    ) -> list[ado.core.resources.ADOResource | None]:
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
            :class:`~ado.core.resources.ADOResource` instances.  The
            first element corresponds to the starting resource; subsequent
            elements correspond to each hop in ``chain``.

        Raises:
            ResourceDoesNotExistError: When ``raise_error_if_no_resource`` is
                ``True`` and any resource in the chain cannot be found.
        """
        hop_count = len(chain)

        # Build one alias per hop.  alias() gives each copy of the table a
        # distinct name (r0, r1, …) so column references stay unambiguous.
        resource_aliases = [
            self._resources_table.alias(f"r{i}") for i in range(hop_count + 1)
        ]

        # Collect the columns we need: data and kind per alias.
        select_columns = []
        for i, resource_alias in enumerate(resource_aliases):
            select_columns.append(resource_alias.c.data.label(f"r{i}_data"))
            select_columns.append(resource_alias.c.kind.label(f"r{i}_kind"))

        # Build the FROM clause by chaining JOINs.  The ON condition uses the
        # ->> JSON path operator (supported by both SQLite ≥3.38 and MySQL).
        joined_from = resource_aliases[0]
        for i, (json_path, linked_kind) in enumerate(chain):
            current_alias = resource_aliases[i]
            next_alias = resource_aliases[i + 1]
            joined_from = joined_from.join(
                next_alias,
                sqlalchemy.and_(
                    next_alias.c.identifier
                    == current_alias.c.data.op("->>")(sqlalchemy.literal(json_path)),
                    next_alias.c.kind == linked_kind.value,
                ),
            )

        query = (
            sqlalchemy.select(*select_columns)
            .select_from(joined_from)
            .where(
                resource_aliases[0].c.identifier == identifier,
                resource_aliases[0].c.kind == kind.value,
            )
        )

        with self.engine.connect() as connectable:
            row = connectable.execute(query).fetchone()

        if row is None:
            if raise_error_if_no_resource:
                raise ResourceDoesNotExistError(resource_id=identifier, kind=kind)
            return [None] * (hop_count + 1)

        row_mapping = row._mapping
        all_kinds = [kind] + [linked_kind for _, linked_kind in chain]
        resources = []
        for i, _ in enumerate(all_kinds):
            raw_data = row_mapping[f"r{i}_data"]
            raw_kind = row_mapping[f"r{i}_kind"]
            data_dict = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            resource = self._deserialize_resource(raw_kind, data_dict)

            if ado.core.resources.VersionIsGreaterThan(
                resource.version, data_dict.get("version", "v0")
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
        stmt = sqlalchemy.select(self._resources_table).where(
            self._resources_table.c.identifier == identifier
        )
        with self.engine.connect() as connectable:
            row = connectable.execute(stmt).mappings().first()

        if row is None:
            return None
        data_raw = row["data"]
        return json.loads(data_raw) if isinstance(data_raw, str) else data_raw

    def getResource(
        self,
        identifier: str,
        kind: CoreResourceKinds,
        raise_error_if_no_resource: bool = False,
        ignore_plugin_validation: bool = True,
    ) -> ado.core.resources.ADOResource | None:
        """Retrieve a resource from the SQL store.

        This method selects the resource with the given *identifier* and
        *kind* from the ``resources`` table.  The JSON payload stored in
        the database is deserialized and converted into the appropriate
        :class:`~ado.core.resources.ADOResource` subclass.

        If the stored version is older than the resource instance being
        retrieved (`resource.version`) the object is automatically updated
        in the database.

        Args:
            identifier: The unique identifier of the resource to fetch.
            kind: The :class:`~ado.core.resources.CoreResourceKinds`
                enum value that specifies the expected resource kind.
            raise_error_if_no_resource: If ``True``, a
                :class:`~ado.metastore.base.ResourceDoesNotExistError`
                is raised when the resource cannot be found.  When ``False``
                (default) the method simply returns ``None``.
            ignore_plugin_validation: When ``True`` (default), nested operation
                and actuator configuration fields skip plugin registry
                validation during deserialization. Set to ``False`` when
                loading resources for runtime use.

        Returns:
            An instance of the appropriate
            :class:`~ado.core.resources.ADOResource` subclass if the
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
              the default Pydantic model from ``ado.core.kindmap``
              is instantiated.
        """

        stmt = sqlalchemy.select(self._resources_table).where(
            self._resources_table.c.identifier == identifier,
            self._resources_table.c.kind == kind.value,
        )
        with self.engine.connect() as connectable:
            row = connectable.execute(stmt).mappings().first()

        resource = None
        if row is not None:
            data_raw = row["data"]
            data_dict = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
            resource = self._deserialize_resource(
                row["kind"],
                data_dict,
                ignore_plugin_validation=ignore_plugin_validation,
            )

            # The stored resource should always have a version - if somehow it doesn't we want this to fail
            if ado.core.resources.VersionIsGreaterThan(
                resource.version, data_dict.get("version", "v0")
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
    ) -> dict[str, ado.core.resources.ADOResource]:
        """Retrieve multiple resources by identifier.

        This method queries the `resources` table for all rows whose
        ``identifier`` column matches an element of *identifiers*.  The
        JSON payload stored in the `data` column is deserialized and
        converted into the appropriate :class:`ado.core.resources.ADOResource`
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
            dict[str, ado.core.resources.ADOResource]:
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

            stmt = sqlalchemy.select(self._resources_table).where(
                self._resources_table.c.identifier.in_(identifiers)
            )
            with self.engine.connect() as connectable:
                rows = connectable.execute(stmt).mappings().all()

            for row in rows:
                identifier = row["identifier"]
                data_raw = row["data"]
                kind = row["kind"]
                data_dict = (
                    json.loads(data_raw) if isinstance(data_raw, str) else data_raw
                )
                try:
                    resource = self._deserialize_resource(
                        kind,
                        data_dict,
                        ignore_plugin_validation=ignore_plugin_validation,
                    )
                except Exception as error:
                    msg = f"Unable to create pydantic model for resource with id: {identifier} with data: {data_raw}. {error}"
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
                :class:`ado.core.resources.CoreResourceKinds`.
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
        import datetime
        import math

        import pandas as pd

        if kind not in [v.value for v in ado.core.resources.CoreResourceKinds]:
            raise ValueError(f"Unknown kind specified: {kind}")

        resources_table = self._resources_table
        dialect = self.engine.dialect.name

        # --- column expressions ---
        col_identifier = resources_table.c.identifier
        col_data = resources_table.c.data

        # name: $.config.metadata.name
        # MySQL returns JSON null as the string "null"; coerce it to SQL NULL.
        if dialect == "sqlite":
            col_name = json_extract_field_as_string(
                col_data, "$.config.metadata.name"
            ).label("name")
        else:
            col_name = sqlalchemy.func.nullif(
                json_extract_field_as_string(col_data, "$.config.metadata.name"),
                "null",
            ).label("name")

        # age in seconds from $.created
        if dialect == "sqlite":
            col_age = sqlalchemy.func.round(
                (
                    sqlalchemy.func.julianday(sqlalchemy.func.datetime("NOW"))
                    - sqlalchemy.func.julianday(
                        sqlalchemy.func.datetime(
                            json_extract_field_as_string(col_data, "$.created")
                        )
                    )
                )
                * 86400
            ).label("age")
        else:
            col_age = sqlalchemy.func.timestampdiff(
                sqlalchemy.text("SECOND"),
                sqlalchemy.func.str_to_date(
                    json_extract_field_as_string(col_data, "$.created"),
                    sqlalchemy.literal("%Y-%m-%dT%T.%fZ"),
                ),
                sqlalchemy.func.now(),
            ).label("age")

        selected_columns: list = [col_identifier, col_name, col_age]

        if details:
            if dialect == "sqlite":
                col_description = json_extract_field_as_string(
                    col_data, "$.config.metadata.description"
                ).label("description")
                col_labels = json_extract_field_as_string(
                    col_data, "$.config.metadata.labels"
                ).label("labels")
            else:
                col_description = sqlalchemy.func.nullif(
                    json_extract_field_as_string(
                        col_data, "$.config.metadata.description"
                    ),
                    "null",
                ).label("description")
                col_labels = sqlalchemy.func.nullif(
                    json_extract_field_as_string(col_data, "$.config.metadata.labels"),
                    "null",
                ).label("labels")
            # description and labels are inserted before age (the last element)
            selected_columns = [
                col_identifier,
                col_name,
                col_description,
                col_labels,
                col_age,
            ]

        if kind == ado.core.resources.CoreResourceKinds.OPERATION.value:
            col_status = json_extract_field_as_string(col_data, "$.status").label(
                "status"
            )
            col_space = json_extract_field_as_string(
                col_data, "$.config.spaces[0]"
            ).label("space")
            selected_columns = [*selected_columns, col_status, col_space]

        # --- build query ---
        query = sqlalchemy.select(*selected_columns).where(
            resources_table.c.kind == kind
        )

        if version is not None:
            query = query.where(resources_table.c.version == version)

        # field selectors
        for field_selector in field_selectors or []:
            for path, candidate in field_selector.items():
                if dialect == "sqlite":
                    where_fragment = (
                        ado.metastore.sql.statements.simulate_json_contains_on_sqlite(
                            path, candidate
                        )
                    )
                    query = query.where(sqlalchemy.text(where_fragment))
                else:
                    # MySQL: JSON_CONTAINS(data, candidate, path)
                    # Also handle null candidates: match rows where path does not exist
                    json_contains_expr = sqlalchemy.func.json_contains(
                        col_data, candidate, path
                    )
                    if candidate == "null":
                        not_contains_path_expr = sqlalchemy.not_(
                            sqlalchemy.func.json_contains_path(
                                col_data, sqlalchemy.literal("one"), path
                            )
                        )
                        query = query.where(
                            sqlalchemy.or_(json_contains_expr, not_contains_path_expr)
                        )
                    else:
                        query = query.where(json_contains_expr)

        # ORDER BY age DESC, NULLs last
        if dialect == "sqlite":
            query = query.order_by(col_age.is_(None), col_age.desc())
        else:
            query = query.order_by(
                sqlalchemy.func.isnull(col_age),
                col_age.desc(),
            )

        with self.engine.connect() as connection:
            result_rows = connection.execute(query).fetchall()

        # Build output DataFrame from query results
        row_dicts = [row._mapping for row in result_rows]

        columns = (
            ["IDENTIFIER", "NAME", "DESCRIPTION", "LABELS", "AGE"]
            if details
            else ["IDENTIFIER", "NAME", "AGE"]
        )

        output_df = pd.DataFrame(
            data={
                "IDENTIFIER": [r["identifier"] for r in row_dicts],
                "NAME": [r["name"] for r in row_dicts],
                "AGE": [r["age"] for r in row_dicts],
            }
        )

        # The DB returns age in seconds; convert to timedelta (NaN values are preserved)
        output_df["AGE"] = output_df["AGE"].apply(
            lambda x: (
                datetime.timedelta(seconds=x)
                if x is not None and not math.isnan(x)
                else x
            )
        )

        if details:
            output_df["DESCRIPTION"] = [r["description"] for r in row_dicts]
            output_df["LABELS"] = [r["labels"] for r in row_dicts]

        if kind == ado.core.resources.CoreResourceKinds.OPERATION.value:
            columns.insert(-1, "STATUS")
            output_df["STATUS"] = [r["status"] for r in row_dicts]
            columns.insert(-1, "SPACE")
            output_df["SPACE"] = [r["space"] for r in row_dicts]

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

        # Convert CoreResourceKinds to string values for the IN clause
        kind_values = [kind.value for kind in kinds]

        # Build CTE: rank resources within each kind by their created timestamp
        # descending so row_rank=1 identifies the most recently created one.
        resources_table = self._resources_table
        created_at_col = json_extract_field_as_string(
            resources_table.c.data, "$.created"
        )
        ranked_resources_cte = (
            sqlalchemy.select(
                resources_table.c.identifier,
                resources_table.c.kind,
                created_at_col.label("created"),
                sqlalchemy.func.row_number()
                .over(
                    partition_by=resources_table.c.kind,
                    order_by=created_at_col.desc(),
                )
                .label("row_rank"),
            )
            .where(resources_table.c.kind.in_(kind_values))
            .cte("ranked_resources")
        )

        query = sqlalchemy.select(
            ranked_resources_cte.c.identifier,
            ranked_resources_cte.c.kind,
            ranked_resources_cte.c.created,
        ).where(ranked_resources_cte.c.row_rank == 1)

        with self.engine.connect() as connectable:
            rows = connectable.execute(query).fetchall()

        # Build dictionary mapping kind enum to its most recently created identifier
        latest_ids: dict[CoreResourceKinds, str] = {}
        for row in rows:
            kind_enum = CoreResourceKinds(row.kind)
            latest_ids[kind_enum] = row.identifier

        return latest_ids

    def resourceTable(self) -> "pd.DataFrame":
        """Return all rows of the resources table as a DataFrame.

        Returns:
            A DataFrame containing all columns and rows of the resources table.
        """
        import pandas as pd

        stmt = sqlalchemy.select(self._resources_table)
        with self.engine.connect() as connectable:
            return pd.read_sql(stmt, con=connectable)

    def getResourcesOfKind(
        self,
        kind: str,
        version: str | None = None,
        field_selectors: list[dict[str, str]] | None = None,
        ignore_validation_errors: bool = True,
    ) -> dict[str, ado.core.resources.ADOResource]:
        """
        Retrieve all resources of a given kind.

        The method first obtains the identifiers of matching resources by
        calling :meth:`getResourceIdentifiersOfKind`. The identifiers are
        then used to fetch the full resource objects via
        :meth:`getResources`.

        Args:
            kind (str): The kind of resources to fetch. Must be one of
                :class:`ado.core.resources.CoreResourceKinds`.
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
            dict[str, ado.core.resources.ADOResource]: A mapping
            where the key is the resource identifier and the value is the
            fully-deserialized :class:`ado.core.resources.ADOResource`
            instance.  An empty dictionary is returned when no matching
            resources are found.

        Raises:
            ValueError: If ``kind`` is not a recognised
                :class:`ado.core.resources.CoreResourceKinds`
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

        relationships_table = self._relationships_table
        resources_table = self._resources_table
        stmt = (
            sqlalchemy.select(
                relationships_table.c.subject_identifier,
                resources_table.c.kind,
            )
            .join(
                resources_table,
                relationships_table.c.subject_identifier
                == resources_table.c.identifier,
            )
            .where(relationships_table.c.object_identifier == identifier)
        )
        if kind is not None:
            stmt = stmt.where(resources_table.c.kind == kind)
        if version is not None:
            stmt = stmt.where(resources_table.c.version == version)

        with self.engine.connect() as connectable:
            rows = connectable.execute(stmt).fetchall()

        related_identifiers = [row[0] for row in rows]
        related_kinds = [row[1] for row in rows]
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

        relationships_table = self._relationships_table
        resources_table = self._resources_table
        stmt = (
            sqlalchemy.select(
                relationships_table.c.object_identifier,
                resources_table.c.kind,
            )
            .join(
                resources_table,
                relationships_table.c.object_identifier == resources_table.c.identifier,
            )
            .where(relationships_table.c.subject_identifier == identifier)
        )
        if kind is not None:
            stmt = stmt.where(resources_table.c.kind == kind)
        if version is not None:
            stmt = stmt.where(resources_table.c.version == version)

        with self.engine.connect() as connectable:
            rows = connectable.execute(stmt).fetchall()

        related_identifiers = [row[0] for row in rows]
        related_kinds = [row[1] for row in rows]
        return pd.DataFrame({"IDENTIFIER": related_identifiers, "TYPE": related_kinds})

    def containsResourceWithIdentifier(
        self, identifier: str, kind: CoreResourceKinds | None = None
    ) -> bool:
        """Check whether the resources table contains a row for the given identifier.

        Args:
            identifier: The resource identifier to look up.
            kind: When provided, also filters by resource kind.

        Returns:
            True if a matching row exists, False otherwise.
        """
        stmt = sqlalchemy.select(sqlalchemy.func.count()).where(
            self._resources_table.c.identifier == identifier
        )
        if kind is not None:
            stmt = stmt.where(self._resources_table.c.kind == kind.value)
        stmt = stmt.select_from(self._resources_table)

        with self.engine.connect() as connectable:
            row_count = connectable.execute(stmt).scalar()

        return row_count != 0

    def addResource(self, resource: ado.core.resources.ADOResource) -> None:
        """Insert a new resource row into the resources table.

        Args:
            resource: The resource to insert.

        Raises:
            ValueError: If resource is not an ADOResource subclass, or if a
                row with the same identifier already exists.
        """
        if not isinstance(resource, ado.core.resources.ADOResource):
            raise ValueError(
                f"Cannot add resource, {resource}, that is not a subclass of ADOResource"
            )

        if self.containsResourceWithIdentifier(resource.identifier):
            raise ValueError(
                f"Resource with id {resource.identifier} already present. "
                f"Use updateResource if you want to overwrite it"
            )
        resource.status.append(
            ado.core.resources.ADOResourceStatus(event=ADOResourceEventEnum.ADDED)
        )
        custom_model_dump = kind_custom_model_dump.get(resource.kind)
        if custom_model_dump:
            representation = custom_model_dump(resource)
        else:
            representation = resource.model_dump_json()

        stmt = self._resources_table.insert().values(
            identifier=resource.identifier,
            kind=resource.kind.value,
            version=resource.version,
            data=json.loads(representation),
        )
        with self.engine.begin() as connectable:
            connectable.execute(stmt)

    def addRelationship(
        self,
        subjectIdentifier: str,
        objectIdentifier: str,
    ) -> None:
        """Insert a row into the resource_relationships table.

        Args:
            subjectIdentifier: Identifier of the subject resource.
            objectIdentifier: Identifier of the object resource.
        """
        stmt = self._relationships_table.insert().values(
            subject_identifier=subjectIdentifier,
            object_identifier=objectIdentifier,
        )
        with self.engine.begin() as connectable:
            connectable.execute(stmt)

    def addRelationshipForResources(
        self, subjectResource: pydantic.BaseModel, objectResource: pydantic.BaseModel
    ) -> None:

        self.addRelationship(
            subjectIdentifier=subjectResource.identifier,
            objectIdentifier=objectResource.identifier,
        )

    def addResourceWithRelationships(
        self,
        resource: ado.core.resources.ADOResource,
        relatedIdentifiers: list,
    ) -> None:
        """For the relationship, the resource id is stored as object and the other ids as subjects

        This is because the others ids must already exist"""

        # Test that the relatedIdentifiers exist before adding
        resource_exists_checks = [
            self.containsResourceWithIdentifier(identifier=ident)
            for ident in relatedIdentifiers
        ]
        if False in resource_exists_checks:
            raise ValueError(f"Unknown resource identifier passed {relatedIdentifiers}")

        self.addResource(resource=resource)
        for identifier in relatedIdentifiers:
            self.addRelationship(
                subjectIdentifier=identifier, objectIdentifier=resource.identifier
            )

    def updateResource(self, resource: ado.core.resources.ADOResource) -> None:
        """Replace any data stored against ``resource.identifier`` with ``resource``.

        Uses a dialect-specific upsert so that the row is inserted if absent
        or updated in-place if it already exists.

        Args:
            resource: The resource whose stored data should be overwritten.
        """
        resource.status.append(
            ado.core.resources.ADOResourceStatus(event=ADOResourceEventEnum.UPDATED)
        )
        custom_model_dump = kind_custom_model_dump.get(resource.kind)
        if custom_model_dump:
            representation = custom_model_dump(resource)
        else:
            representation = resource.model_dump_json()

        values = {
            "identifier": resource.identifier,
            "kind": resource.kind.value,
            "version": resource.version,
            "data": json.loads(representation),
        }
        if self.engine.dialect.name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            stmt = sqlite_insert(self._resources_table).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["identifier"],
                set_={"data": stmt.excluded.data},
            )
        else:
            from sqlalchemy.dialects.mysql import insert as mysql_insert

            stmt = mysql_insert(self._resources_table).values(**values)
            stmt = stmt.on_duplicate_key_update(data=stmt.inserted.data)

        with self.engine.begin() as connectable:
            connectable.execute(stmt)

    def deleteResource(self, identifier: str) -> None:
        """Delete a resource and its object-side relationships from the store.

        Args:
            identifier: The identifier of the resource to delete.

        Raises:
            ValueError: If the resource does not exist, or if relationships
                exist where this resource is the subject.
        """
        if not self.containsResourceWithIdentifier(identifier):
            raise ValueError(
                f"Cannot delete resource with id {identifier} - it is not present"
            )

        relatedAsObject = self.getRelatedObjectResourceIdentifiers(
            identifier=identifier
        )
        if len(relatedAsObject) > 0:
            raise ValueError(
                f"Cannot delete resource {identifier} as there are existing relationships where it is the subject. "
                f"You must delete all the related object resources first:\n{relatedAsObject['IDENTIFIER']}"
            )
        self.deleteObjectRelationships(identifier=identifier)
        stmt = sqlalchemy.delete(self._resources_table).where(
            self._resources_table.c.identifier == identifier
        )
        with self.engine.begin() as connectable:
            connectable.execute(stmt)

    def deleteObjectRelationships(self, identifier: str) -> None:
        """Delete all relationship rows where ``identifier`` is the object.

        Args:
            identifier: The object-side identifier whose relationship rows
                should be removed.

        Raises:
            ValueError: If relationships exist where this identifier is also
                the subject, which would break provenance.
        """
        relatedAsObject = self.getRelatedObjectResourceIdentifiers(
            identifier=identifier
        )
        if len(relatedAsObject) > 0:
            raise ValueError(
                f"Cannot delete relationships where {identifier} is the object as there are existing relationships where it is the subject. "
                f"You must delete all the related object resources first:\n{relatedAsObject['IDENTIFIER']}"
            )
        stmt = sqlalchemy.delete(self._relationships_table).where(
            self._relationships_table.c.object_identifier == identifier
        )
        with self.engine.begin() as connectable:
            connectable.execute(stmt)

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
                        sqlalchemy.delete(self._relationships_table).where(
                            self._relationships_table.c.object_identifier == identifier
                        )
                    )

                    session.execute(
                        sqlalchemy.delete(self._resources_table).where(
                            self._resources_table.c.identifier == identifier,
                            self._resources_table.c.kind
                            == CoreResourceKinds.SAMPLESTORE.value,
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
                        sqlalchemy.text(
                            f"DROP TABLE sqlsource_{identifier}_measurement_requests_results"
                        )
                    )

                    session.execute(
                        sqlalchemy.text(
                            f"DROP TABLE sqlsource_{identifier}_measurement_results"
                        )
                    )

                    session.execute(
                        sqlalchemy.text(
                            f"DROP TABLE sqlsource_{identifier}_measurement_requests"
                        )
                    )

                    session.execute(
                        sqlalchemy.text(f"DROP TABLE sqlsource_{identifier}")
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
                    relationships_table = self._relationships_table
                    resources_table = self._resources_table

                    space_subquery = (
                        sqlalchemy.select(relationships_table.c.subject_identifier)
                        .where(
                            relationships_table.c.object_identifier == identifier,
                            relationships_table.c.subject_identifier.like("space-%"),
                        )
                        .scalar_subquery()
                    )
                    sample_store_id = session.execute(
                        sqlalchemy.select(
                            json_extract_field_as_string(
                                resources_table.c.data,
                                "$.config.sampleStoreIdentifier",
                            )
                        ).where(resources_table.c.identifier == space_subquery)
                    ).first()[0]

                    # The user might choose to ignore running operations
                    # <--------- START CHECKS FOR RUNNING OPERATIONS --------->
                    if not ignore_running_operations:
                        spaces_in_sample_store = [
                            result[0]
                            for result in session.execute(
                                sqlalchemy.select(
                                    relationships_table.c.object_identifier
                                ).where(
                                    relationships_table.c.subject_identifier
                                    == sample_store_id,
                                    relationships_table.c.object_identifier.like(
                                        "space-%"
                                    ),
                                )
                            )
                        ]

                        spaces_json = sqlalchemy.literal(
                            json.dumps(spaces_in_sample_store)
                        )
                        data_col = resources_table.c.data
                        running_operations = [
                            result[0]
                            for result in session.execute(
                                sqlalchemy.select(resources_table.c.identifier).where(
                                    resources_table.c.kind
                                    == CoreResourceKinds.OPERATION.value,
                                    sqlalchemy.func.JSON_OVERLAPS(
                                        data_col.op("->")(
                                            sqlalchemy.literal("$.config.spaces")
                                        ),
                                        spaces_json,
                                    ),
                                    sqlalchemy.func.JSON_CONTAINS(
                                        data_col.op("->")(
                                            sqlalchemy.literal("$.status")
                                        ),
                                        sqlalchemy.literal('{"event":"started"}'),
                                    ),
                                    sqlalchemy.not_(
                                        sqlalchemy.func.JSON_CONTAINS(
                                            data_col.op("->")(
                                                sqlalchemy.literal("$.status")
                                            ),
                                            sqlalchemy.literal('{"event":"finished"}'),
                                        )
                                    ),
                                )
                            )
                        ]

                        if running_operations:
                            raise RunningOperationsPreventingDeletionError(
                                operation_id=identifier,
                                running_operations=running_operations,
                            )

                    # <--------- END CHECKS FOR RUNNING OPERATIONS --------->

                    # <--------- CASCADE DELETE DATACONTAINER CHILDREN --------->
                    # Query all direct children of this operation
                    import pandas as pd

                    child_rows = session.execute(
                        sqlalchemy.select(
                            relationships_table.c.object_identifier,
                            resources_table.c.kind,
                        )
                        .join(
                            resources_table,
                            relationships_table.c.object_identifier
                            == resources_table.c.identifier,
                        )
                        .where(relationships_table.c.subject_identifier == identifier)
                    ).fetchall()

                    child_resources_df = pd.DataFrame(
                        child_rows, columns=["IDENTIFIER", "TYPE"]
                    )

                    if not child_resources_df.empty:
                        non_data_container_children = child_resources_df[
                            child_resources_df["TYPE"]
                            != CoreResourceKinds.DATACONTAINER.value
                        ]
                        if not non_data_container_children.empty:
                            raise ResourceHasChildrenError(
                                resource_id=identifier,
                                kind=CoreResourceKinds.OPERATION,
                                children_resources=non_data_container_children,
                            )

                        # All children are DataContainers - check each has no
                        # grandchildren
                        for data_container_id in child_resources_df["IDENTIFIER"]:
                            grandchildren_rows = session.execute(
                                sqlalchemy.select(
                                    relationships_table.c.object_identifier,
                                    resources_table.c.kind,
                                )
                                .join(
                                    resources_table,
                                    relationships_table.c.object_identifier
                                    == resources_table.c.identifier,
                                )
                                .where(
                                    relationships_table.c.subject_identifier
                                    == data_container_id
                                )
                            ).fetchall()

                            if grandchildren_rows:
                                grandchildren_df = pd.DataFrame(
                                    grandchildren_rows, columns=["IDENTIFIER", "TYPE"]
                                )
                                raise ResourceHasChildrenError(
                                    resource_id=data_container_id,
                                    kind=CoreResourceKinds.DATACONTAINER,
                                    children_resources=grandchildren_df,
                                )

                        # Safe to delete all DataContainer children
                        data_container_ids = child_resources_df["IDENTIFIER"].tolist()
                        session.execute(
                            sqlalchemy.delete(self._relationships_table).where(
                                self._relationships_table.c.object_identifier.in_(
                                    data_container_ids
                                )
                            )
                        )
                        session.execute(
                            sqlalchemy.delete(self._resources_table).where(
                                self._resources_table.c.identifier.in_(
                                    data_container_ids
                                ),
                                self._resources_table.c.kind
                                == CoreResourceKinds.DATACONTAINER.value,
                            )
                        )
                    # <--------- END CASCADE DELETE DATACONTAINER CHILDREN --------->

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
                        sqlalchemy.text(f"""
                            DELETE
                            FROM sqlsource_{sample_store_id}_measurement_results
                            WHERE uid NOT IN (
                                SELECT DISTINCT(result_uid)
                                FROM sqlsource_{sample_store_id}_measurement_requests_results
                            )
                            """)  # noqa: S608 - sample store id is not a user input
                    )

                    # The requests that have no link to results anymore
                    # can now be safely deleted.
                    session.execute(
                        sqlalchemy.text(f"""
                            DELETE
                            FROM sqlsource_{sample_store_id}_measurement_requests
                            WHERE uid NOT IN (
                                SELECT DISTINCT(request_uid)
                                FROM sqlsource_{sample_store_id}_measurement_requests_results
                            )
                            """)  # noqa: S608 - sample store id is not a user input
                    )

                    session.execute(
                        sqlalchemy.delete(self._relationships_table).where(
                            self._relationships_table.c.object_identifier == identifier
                        )
                    )

                    session.execute(
                        sqlalchemy.delete(self._resources_table).where(
                            self._resources_table.c.identifier == identifier,
                            self._resources_table.c.kind
                            == CoreResourceKinds.OPERATION.value,
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
        """Delete a discovery space resource and its object-side relationships.

        Args:
            identifier: The identifier of the discovery space to delete.

        Raises:
            DeleteFromDatabaseError: If the delete transaction fails.
        """
        import sqlalchemy.orm

        with sqlalchemy.orm.Session(self.engine) as session:
            try:
                with session.begin():
                    session.execute(
                        sqlalchemy.delete(self._relationships_table).where(
                            self._relationships_table.c.object_identifier == identifier
                        )
                    )
                    session.execute(
                        sqlalchemy.delete(self._resources_table).where(
                            self._resources_table.c.identifier == identifier,
                            self._resources_table.c.kind
                            == CoreResourceKinds.DISCOVERYSPACE.value,
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
        """Delete a data container resource and its object-side relationships.

        Args:
            identifier: The identifier of the data container to delete.

        Raises:
            DeleteFromDatabaseError: If the delete transaction fails.
        """
        import sqlalchemy.orm

        with sqlalchemy.orm.Session(self.engine) as session:
            try:
                with session.begin():
                    session.execute(
                        sqlalchemy.delete(self._relationships_table).where(
                            self._relationships_table.c.object_identifier == identifier
                        )
                    )
                    session.execute(
                        sqlalchemy.delete(self._resources_table).where(
                            self._resources_table.c.identifier == identifier,
                            self._resources_table.c.kind
                            == CoreResourceKinds.DATACONTAINER.value,
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
        """Delete an actuator configuration resource and its object-side relationships.

        Args:
            identifier: The identifier of the actuator configuration to delete.

        Raises:
            DeleteFromDatabaseError: If the delete transaction fails.
        """
        import sqlalchemy.orm

        with sqlalchemy.orm.Session(self.engine) as session:
            try:
                with session.begin():
                    session.execute(
                        sqlalchemy.delete(self._relationships_table).where(
                            self._relationships_table.c.object_identifier == identifier
                        )
                    )
                    session.execute(
                        sqlalchemy.delete(self._resources_table).where(
                            self._resources_table.c.identifier == identifier,
                            self._resources_table.c.kind
                            == CoreResourceKinds.ACTUATORCONFIGURATION.value,
                        )
                    )
            except Exception as e:
                session.rollback()
                raise DeleteFromDatabaseError(
                    resource_id=identifier,
                    resource_kind=CoreResourceKinds.ACTUATORCONFIGURATION,
                    rollback_occurred=True,
                ) from e

    def delete_document(self, identifier: str) -> None:
        """Delete a document resource and all its relationships.

        Args:
            identifier: The identifier of the document to delete.

        Raises:
            DeleteFromDatabaseError: If the delete transaction fails.
        """
        import sqlalchemy.orm

        with sqlalchemy.orm.Session(self.engine) as session:
            try:
                with session.begin():
                    session.execute(
                        sqlalchemy.delete(self._relationships_table).where(
                            self._relationships_table.c.object_identifier == identifier
                        )
                    )
                    session.execute(
                        sqlalchemy.delete(self._relationships_table).where(
                            self._relationships_table.c.subject_identifier == identifier
                        )
                    )
                    session.execute(
                        sqlalchemy.delete(self._resources_table).where(
                            self._resources_table.c.identifier == identifier,
                            self._resources_table.c.kind
                            == CoreResourceKinds.DOCUMENT.value,
                        )
                    )
            except Exception as e:
                session.rollback()
                raise DeleteFromDatabaseError(
                    resource_id=identifier,
                    resource_kind=CoreResourceKinds.DOCUMENT,
                    rollback_occurred=True,
                ) from e

    # ---------------------------------------------------------------------------
    # Hierarchy traversal
    # ---------------------------------------------------------------------------

    def get_resources_by_relationship(
        self,
        kind: CoreResourceKinds,
        identifier: str | set[str] | None,
        relationship: Literal["child", "parent", "both"] = "both",
        result_kinds: "set[CoreResourceKinds] | None" = None,
        max_hops: int | None = None,
        identifiers_only: bool = False,
        include_start_resources: bool = False,
    ) -> (
        dict[CoreResourceKinds, set[str]]
        | dict[str, dict[CoreResourceKinds, set[str]]]
        | dict[CoreResourceKinds, dict[str, "ado.core.resources.ADOResource"]]
        | dict[
            str,
            dict[
                CoreResourceKinds,
                dict[str, "ado.core.resources.ADOResource"],
            ],
        ]
    ):
        """Walk the resource graph stored in ``resource_relationships``.

        Issues at most three SQL queries: when ``identifier=None`` a seed query
        fetches all identifiers of ``kind`` via
        :meth:`getResourceIdentifiersOfKind`; then one recursive traversal query
        built as a SQLAlchemy Core recursive CTE;
        and, when ``identifiers_only=False``, one additional batched resource
        query via :meth:`getResources`. When ``identifier`` is a ``str`` or
        ``set[str]`` only the latter two queries (or one, if
        ``identifiers_only=True``) are issued.

        Args:
            kind: The :class:`~ado.core.resources.CoreResourceKinds` of
                the starting resources.
            identifier: Controls which resources are used as traversal origins.

                * ``str`` — a single start resource identifier; the return value
                  is unwrapped (no outer origin key).
                * ``set[str]`` — multiple explicit start resource identifiers.
                * ``None`` — all resources of ``kind`` are used as start
                  resources (seeded via :meth:`getResourceIdentifiersOfKind`).
                * An **empty set** returns an empty result immediately.

            relationship: ``'child'`` (follow edges where the current node is
                the source), ``'parent'`` (follow edges where the current node
                is the target), or ``'both'`` (default).
            result_kinds: When not ``None``, only resources whose kind is in
                this set are included in the returned result. ``None`` returns
                every reachable kind.
            max_hops: Maximum number of relationship hops to follow from each
                start resource. When ``None`` the traversal runs to the full
                depth cap. Values exceeding the cap are silently capped.
            identifiers_only: When ``False`` (default) discovered identifiers
                are hydrated into full
                :class:`~ado.core.resources.ADOResource` objects via
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
            ValueError: If ``relationship`` is not ``'child'``, ``'parent'``
                or ``'both'``.
            ValueError: If ``include_start_resources=True`` is used together
                with ``identifiers_only=True``.
            ValueError: If ``include_start_resources=True`` is used with
                ``identifier=None``.
        """
        # ------------------------------------------------------------------
        # 0. Validate parameters eagerly
        # ------------------------------------------------------------------
        if relationship not in {"child", "parent", "both"}:
            raise ValueError(
                f"relationship must be 'child', 'parent' or 'both', got {relationship!r}"
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
        # The hierarchy maximum is enforced by capping max_hops; passing
        # max_hops=None lets the traversal run to the full depth cap.
        from ado.metastore.sql.utils import _MAX_HIERARCHY_HOPS

        effective_max_hops = (
            _MAX_HIERARCHY_HOPS
            if max_hops is None
            else min(max_hops, _MAX_HIERARCHY_HOPS)
        )

        resources_table = self._resources_table
        relationships_table = self._relationships_table

        # logical_edges_cte: join relationships with resources on both ends so we
        # have (from_id, from_kind, to_id, to_kind) for each stored edge.
        subject_resource_alias = resources_table.alias("le_parent")
        object_resource_alias = resources_table.alias("le_child")
        logical_edges_cte = (
            sqlalchemy.select(
                subject_resource_alias.c.identifier.label("from_identifier"),
                subject_resource_alias.c.kind.label("from_kind"),
                object_resource_alias.c.identifier.label("to_identifier"),
                object_resource_alias.c.kind.label("to_kind"),
            )
            .select_from(relationships_table)
            .join(
                subject_resource_alias,
                subject_resource_alias.c.identifier
                == relationships_table.c.subject_identifier,
            )
            .join(
                object_resource_alias,
                object_resource_alias.c.identifier
                == relationships_table.c.object_identifier,
            )
            .cte("logical_edges")
        )

        # Seed: one row per origin identifier of the requested kind.
        # visited_path is seeded as ',id,' so membership checks are unambiguous.
        is_sqlite = self.engine.dialect.name == "sqlite"

        if is_sqlite:
            seed_visited_path = (
                sqlalchemy.literal(",")
                + resources_table.c.identifier
                + sqlalchemy.literal(",")
            )
        else:
            seed_visited_path = sqlalchemy.func.CONCAT(
                sqlalchemy.literal(","),
                resources_table.c.identifier,
                sqlalchemy.literal(","),
            )

        traversal_seed = sqlalchemy.select(
            resources_table.c.identifier.label("origin_identifier"),
            resources_table.c.kind.label("current_kind"),
            resources_table.c.identifier.label("current_identifier"),
            sqlalchemy.literal(0).label("depth"),
            seed_visited_path.label("visited_path"),
        ).where(
            resources_table.c.kind == kind.value,
            resources_table.c.identifier.in_(list(_identifiers_requested)),
        )

        traversal_cte = traversal_seed.cte("traversal", recursive=True)

        # next_identifier/next_kind expressions depend on traversal direction.
        if relationship == "child":
            step_join_condition = (
                logical_edges_cte.c.from_identifier
                == traversal_cte.c.current_identifier
            )
            next_identifier = logical_edges_cte.c.to_identifier
            next_kind = logical_edges_cte.c.to_kind
        elif relationship == "parent":
            step_join_condition = (
                logical_edges_cte.c.to_identifier == traversal_cte.c.current_identifier
            )
            next_identifier = logical_edges_cte.c.from_identifier
            next_kind = logical_edges_cte.c.from_kind
        else:  # "both"
            step_join_condition = sqlalchemy.or_(
                logical_edges_cte.c.from_identifier
                == traversal_cte.c.current_identifier,
                logical_edges_cte.c.to_identifier == traversal_cte.c.current_identifier,
            )
            next_identifier = sqlalchemy.case(
                (
                    logical_edges_cte.c.from_identifier
                    == traversal_cte.c.current_identifier,
                    logical_edges_cte.c.to_identifier,
                ),
                else_=logical_edges_cte.c.from_identifier,
            )
            next_kind = sqlalchemy.case(
                (
                    logical_edges_cte.c.from_identifier
                    == traversal_cte.c.current_identifier,
                    logical_edges_cte.c.to_kind,
                ),
                else_=logical_edges_cte.c.from_kind,
            )

        # visited_path cycle guard: append next_identifier and a trailing comma.
        if is_sqlite:
            next_visited_path = (
                traversal_cte.c.visited_path + next_identifier + sqlalchemy.literal(",")
            )
            cycle_guard_pattern = (
                sqlalchemy.literal("%,") + next_identifier + sqlalchemy.literal(",%")
            )
        else:
            next_visited_path = sqlalchemy.func.CONCAT(
                traversal_cte.c.visited_path, next_identifier, sqlalchemy.literal(",")
            )
            cycle_guard_pattern = sqlalchemy.func.CONCAT(
                sqlalchemy.literal("%,"), next_identifier, sqlalchemy.literal(",%")
            )

        recursive_step = (
            sqlalchemy.select(
                traversal_cte.c.origin_identifier,
                next_kind.label("current_kind"),
                next_identifier.label("current_identifier"),
                (traversal_cte.c.depth + 1).label("depth"),
                next_visited_path.label("visited_path"),
            )
            .select_from(traversal_cte)
            .join(logical_edges_cte, step_join_condition)
            .where(
                traversal_cte.c.depth < effective_max_hops,
                traversal_cte.c.visited_path.notlike(cycle_guard_pattern),
            )
        )

        traversal_cte = traversal_cte.union_all(recursive_step)

        query = sqlalchemy.select(
            traversal_cte.c.origin_identifier,
            traversal_cte.c.current_identifier.label("identifier"),
            traversal_cte.c.current_kind.label("kind"),
        ).where(traversal_cte.c.depth > 0)

        with self.engine.connect() as connection:
            raw_rows = connection.execute(query).fetchall()

        # ------------------------------------------------------------------
        # 3. Build the mapping
        #    { origin_id -> { CoreResourceKinds -> {related_id, ...} } }
        # ------------------------------------------------------------------
        related_by_origin: dict[str, dict[CoreResourceKinds, set[str]]] = {}
        identifiers_to_fetch: set[str] = set()

        for row in raw_rows:
            origin_identifier = row.origin_identifier
            related_identifier = row.identifier
            related_kind = row.kind

            # Don't include the start identifiers in discovered results
            # This should never happen, if it does, we have a bug.
            if related_identifier in _identifiers_requested:
                continue

            resource_kind = CoreResourceKinds(related_kind)

            # Apply result_kinds filter if specified
            if result_kinds is not None and resource_kind not in result_kinds:
                continue

            identifiers_to_fetch.add(related_identifier)
            related_by_origin.setdefault(origin_identifier, {}).setdefault(
                resource_kind, set()
            ).add(related_identifier)

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
            dict[CoreResourceKinds, dict[str, ado.core.resources.ADOResource]],
        ] = {}

        for origin_identifier, related_identifiers_by_kind in related_by_origin.items():
            hydrated_related_resources_by_kind: dict[
                CoreResourceKinds,
                dict[str, ado.core.resources.ADOResource],
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
            :class:`~ado.core.discoveryspace.stats.DiscoverySpaceStatistics`
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
        # Both MySQL JSON_LENGTH and SQLite json_array_length are called via
        # func so SQLAlchemy emits the right name per dialect.
        # ------------------------------------------------------------------
        is_sqlite = self.engine.dialect.name == "sqlite"
        space_alias = self._resources_table.alias("space_table")
        relationship_alias = self._relationships_table.alias("relationship_table")
        operation_alias = self._resources_table.alias("operation_table")

        json_array_length_func = (
            sqlalchemy.func.json_array_length
            if is_sqlite
            else sqlalchemy.func.JSON_LENGTH  # noqa: E501
        )
        num_experiments_col = sqlalchemy.func.coalesce(
            json_array_length_func(
                sqlalchemy.func.JSON_EXTRACT(
                    space_alias.c.data,
                    sqlalchemy.literal("$.config.experiments.experiments"),
                )
            ),
            0,
        )

        explore_type_literal = sqlalchemy.literal(DiscoveryOperationEnum.EXPLORE.value)
        explore_legacy_literal = sqlalchemy.literal("search")
        operation_type_extract = sqlalchemy.func.JSON_EXTRACT(
            operation_alias.c.data, sqlalchemy.literal("$.operationType")
        )
        is_explore_operation_case = sqlalchemy.case(
            (
                operation_type_extract.in_(
                    [explore_type_literal, explore_legacy_literal]
                ),
                1,
            ),
        )

        query = (
            sqlalchemy.select(
                space_alias.c.identifier.label("space_id"),
                num_experiments_col.label("num_experiments"),
                sqlalchemy.func.count(operation_alias.c.identifier).label(
                    "total_operations"
                ),
                sqlalchemy.func.count(is_explore_operation_case).label(
                    "explore_operations"
                ),
            )
            .select_from(space_alias)
            .outerjoin(
                relationship_alias,
                relationship_alias.c.subject_identifier == space_alias.c.identifier,
            )
            .outerjoin(
                operation_alias,
                sqlalchemy.and_(
                    operation_alias.c.identifier
                    == relationship_alias.c.object_identifier,
                    operation_alias.c.kind == CoreResourceKinds.OPERATION.value,
                ),
            )
            .where(space_alias.c.identifier.in_(list(_space_ids)))
            .group_by(space_alias.c.identifier, space_alias.c.data)
        )

        try:
            with self.engine.begin() as conn:
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
            :class:`~ado.core.datacontainer.stats.DataContainerStatistics`.
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
        # The dialect-specific JSON column expressions are wrapped with
        # literal_column() so they are emitted verbatim; the table reference
        # comes from self._resources_table to avoid hard-coding the table name.
        is_sqlite = self.engine.dialect.name == "sqlite"
        resources_table = self._resources_table

        if is_sqlite:

            def _sqlite_json_count(path: str) -> sqlalchemy.ColumnElement:  # type: ignore[type-arg]
                """Count entries in a JSON object/array at the given path via json_each.

                Args:
                    path: A JSON path expression (e.g. ``$.config.tabularData``).

                Returns:
                    A correlated scalar subquery that counts rows returned by
                    ``json_each`` for the JSON value at ``path``.
                """
                json_each_alias = sqlalchemy.func.json_each(
                    sqlalchemy.func.json_extract(
                        resources_table.c.data, sqlalchemy.literal(path)
                    )
                ).alias("json_each_entries")
                return (
                    sqlalchemy.select(sqlalchemy.func.count())
                    .select_from(json_each_alias)
                    .correlate(resources_table)
                    .scalar_subquery()
                )

            num_tables_col = _sqlite_json_count("$.config.tabularData").label(
                "num_tables"
            )
            num_locations_col = _sqlite_json_count("$.config.locationData").label(
                "num_locations"
            )
            num_key_values_col = _sqlite_json_count("$.config.data").label(
                "num_key_values"
            )
            data_bytes_col = sqlalchemy.func.coalesce(
                sqlalchemy.func.LENGTH(
                    sqlalchemy.func.JSON_EXTRACT(
                        resources_table.c.data, sqlalchemy.literal("$.config")
                    )
                )
                - sqlalchemy.func.LENGTH(
                    sqlalchemy.func.JSON_EXTRACT(
                        resources_table.c.data,
                        sqlalchemy.literal("$.config.metadata"),
                    )
                ),
                0,
            ).label("data_bytes")
        else:
            # On MySQL, JSON_LENGTH of a JSON null scalar returns 1 (scalar
            # length is 1 per the spec).  We must guard with JSON_TYPE to
            # return 0 for absent/null fields.
            # For byte count, JSON_STORAGE_SIZE returns the actual binary
            # storage size of the JSON value, which is more accurate than
            # LENGTH(JSON_EXTRACT(...)) (text representation length).
            def _mysql_json_count(path: str) -> sqlalchemy.ColumnElement:  # type: ignore[type-arg]
                """Return JSON_LENGTH of the value at path, or 0 if absent or JSON null.

                Args:
                    path: A JSON path expression (e.g. ``$.config.tabularData``).

                Returns:
                    An expression that evaluates to 0 when the field is absent
                    (``->>`` returns SQL NULL, handled by COALESCE) or is a
                    JSON null literal (guarded by JSON_TYPE check), otherwise
                    returns JSON_LENGTH of the value.
                """
                extracted_value = resources_table.c.data.op("->>")(
                    sqlalchemy.literal(path)
                )
                return sqlalchemy.func.IF(
                    sqlalchemy.func.JSON_TYPE(extracted_value) == "NULL",
                    0,
                    sqlalchemy.func.coalesce(
                        sqlalchemy.func.JSON_LENGTH(extracted_value), 0
                    ),
                )

            num_tables_col = _mysql_json_count("$.config.tabularData").label(
                "num_tables"
            )
            num_locations_col = _mysql_json_count("$.config.locationData").label(
                "num_locations"
            )
            num_key_values_col = _mysql_json_count("$.config.data").label(
                "num_key_values"
            )
            data_bytes_col = sqlalchemy.func.coalesce(
                sqlalchemy.func.JSON_STORAGE_SIZE(
                    sqlalchemy.func.JSON_EXTRACT(
                        resources_table.c.data, sqlalchemy.literal("$.config")
                    )
                )
                - sqlalchemy.func.JSON_STORAGE_SIZE(
                    sqlalchemy.func.JSON_EXTRACT(
                        resources_table.c.data,
                        sqlalchemy.literal("$.config.metadata"),
                    )
                ),
                0,
            ).label("data_bytes")

        query = sqlalchemy.select(
            resources_table.c.identifier,
            num_tables_col,
            num_locations_col,
            num_key_values_col,
            data_bytes_col,
        ).where(resources_table.c.identifier.in_(list(datacontainer_ids)))

        try:
            with self.engine.begin() as conn:
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
