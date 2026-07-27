# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
import logging
import typing
import uuid
import warnings
from typing import TYPE_CHECKING, Annotated, Literal

import pydantic
import sqlalchemy
from sqlalchemy.exc import InvalidRequestError, SQLAlchemyError

import ado.core.samplestore.config
import ado.core.samplestore.csv
import ado.metastore.sql.statements
from ado.core.samplestore.base import (
    ActiveSampleStore,
    FailedToDecodeStoredEntityError,
    FailedToDecodeStoredMeasurementResultForEntityError,
)
from ado.metastore.sql.utils import engine_for_sql_store
from ado.modules.actuators.catalog import ExperimentCatalog
from ado.schema.entity import Entity
from ado.schema.experiment import Experiment
from ado.schema.property import (
    ConstitutiveProperty,
)
from ado.schema.reference import ExperimentReference
from ado.schema.request import (
    MeasurementRequest,
    MeasurementRequestStateEnum,
    ReplayedMeasurement,
)
from ado.schema.result import (
    DuplicateMeasurementResultError,
    InvalidMeasurementResult,
    MeasurementResult,
    MeasurementResultStateEnum,
    ValidMeasurementResult,
)
from ado.schema.virtual_property import (
    PropertyAggregationMethod,
    PropertyAggregationMethodEnum,
)
from ado.utilities.location import (
    SQLiteStoreConfiguration,
    SQLStoreConfiguration,
)
from ado.utilities.pandas import (
    filter_dataframe_columns,
    reorder_dataframe_columns,
)

if TYPE_CHECKING:
    import pandas as pd
    from rich.console import RenderableType

    from ado.core.operation.stats import OperationMeasurementStatistics
    from ado.core.samplestore.stats import (  # noqa: F401 — used in annotations
        SampleStoreStatistics,
    )

# Process-level cache of (db_url, tablename) pairs for which the four DDL tables
# have already been verified to exist, along with their reflected metadata.
# Skips the four `CREATE TABLE IF NOT EXISTS` round-trips and metadata reflection
# on every subsequent SQLSampleStore construction for the same store.
# The db_url is included so that two stores with the same identifier but pointing
# to different databases are treated independently.
_source_tables_verified: set[tuple[str, str]] = set()
_reflected_metadata_cache: dict[tuple[str, str], sqlalchemy.MetaData] = {}

# Sentinel for the instance-level experiment-catalog cache.
# Using a distinct object lets us cache ``None`` (empty store) without
# confusing it with "not yet computed".
_UNSET = object()


class SQLSampleStoreConfiguration(pydantic.BaseModel):
    identifier: Annotated[
        str | None, pydantic.Field(description="id for this sample store")
    ]
    configuration: Annotated[
        SQLStoreConfiguration | None,
        pydantic.Field(description="connection information for database"),
    ] = None


class SQLSampleStore(ActiveSampleStore):
    """
    Provides a non-optimized, non-production DB for storing entities

    Each source is specific to a DiscoverySpace i.e. has a specific measurement space.
    You cannot add entities that do not conform to this space
    """

    @classmethod
    def from_csv(
        cls,
        csvPath: str,
        idColumn: str,
        storeConfiguration: SQLStoreConfiguration | SQLiteStoreConfiguration,
        generatorIdentifier: str | None = None,
        experimentIdentifier: str | None = None,
        actuatorIdentifier: str = "replay",
        observedPropertyColumns: list[str] | None = None,
        constitutivePropertyColumns: list[str] | None = None,
        propertyFormat: Literal["target", "observed"] = "target",
    ) -> "SQLSampleStore":

        csv_sample_store = ado.core.samplestore.csv.CSVSampleStore.from_csv(
            csvPath=csvPath,
            idColumn=idColumn,
            generatorIdentifier=generatorIdentifier,
            experimentIdentifier=experimentIdentifier,
            actuatorIdentifier=actuatorIdentifier,
            observedPropertyColumns=observedPropertyColumns,
            constitutivePropertyColumns=constitutivePropertyColumns,
            propertyFormat=propertyFormat,
        )

        sql_sample_store = cls(
            identifier=None,
            storageLocation=storeConfiguration,
            parameters={},
        )
        sql_sample_store.add_external_entities(csv_sample_store.entities)

        return sql_sample_store

    def __rich__(self) -> "RenderableType":
        """Render this SQL sample store using rich."""
        from rich.console import Group
        from rich.text import Text

        from ado.utilities.rich import get_rich_repr

        return Group(
            Text.assemble(("Identifier: ", "bold"), (self.uri, "bold green")),
            Text("Number of entities:", style="bold", end=" "),
            get_rich_repr(self.numberOfEntities),
        )

    def commit(self) -> None:
        pass

    @classmethod
    def experimentCatalogFromReference(
        cls, reference: ado.core.samplestore.config.SampleStoreReference
    ) -> ExperimentCatalog:
        """Return the experiment catalog for the store described by *reference*.

        Delegates to :meth:`experimentCatalog` on a live ``SQLSampleStore``
        instance so that the fix, caching, and future improvements are
        inherited automatically.
        """
        if reference.identifier is None:
            raise ValueError(
                f"No identifier provided for SQLSampleStore - cannot read catalog. Data passed: {reference}"
            )
        if reference.storageLocation is None:
            raise ValueError(
                "SQLSampleStore.experimentCatalog requires valid location parameters."
            )
        store = cls(
            identifier=reference.identifier,
            storageLocation=reference.storageLocation,
            parameters={},
        )
        return store.experimentCatalog()

    def experimentCatalog(
        self,
    ) -> ExperimentCatalog | None:
        """Return the experiment catalog derived from replay measurement results.

        The catalog is built by querying the measurement-results table for rows
        whose ``actuatorIdentifier`` is ``"replay"``.  One row per distinct
        entity is loaded; the entity representation is decoded to recover the
        ``constitutiveProperties`` and ``observedProperties`` needed to
        reconstruct each :class:`~ado.schema.experiment.Experiment`.

        The result is cached in ``self._experiment_catalog`` so that repeated
        calls do not hit the database.  The cache is invalidated by
        :meth:`add_external_entities` and the ``ReplayedMeasurement`` branch of
        :meth:`addMeasurement`.

        Returns:
            An :class:`~ado.modules.actuators.catalog.ExperimentCatalog` when
            replay results are present, or ``None`` when the store is empty or
            contains no replay results.
        """
        if self._experiment_catalog is not _UNSET:
            return self._experiment_catalog  # type: ignore[return-value]

        # Step 1: collect the distinct entity_ids that have at least one replay
        # measurement result.  We use COALESCE over the two known serialisation
        # formats (compressed: top-level experimentReference; legacy: nested
        # inside measurements[0].property.experimentReference) so both are
        # handled without fetching and deserialising every row in Python.
        query = sqlalchemy.text(f"""
            SELECT DISTINCT entity_id
            FROM {self._tablename}_measurement_results
            WHERE COALESCE(
                JSON_EXTRACT(data, '$.experimentReference.actuatorIdentifier'),
                JSON_EXTRACT(data, '$.measurements[0].property.experimentReference.actuatorIdentifier')
            ) IN ('replay', '"replay"')
        """)  # noqa: S608 - self._tablename is not untrusted

        try:
            with self.engine.begin() as connectable:
                entity_ids = {row[0] for row in connectable.execute(query)}
        except SQLAlchemyError as error:
            msg = f"Unable to query replay entity IDs for catalog from sample store {self._tablename}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        if not entity_ids:
            self._experiment_catalog = None
            return None

        # Step 2: load those entities (with their measurement results) via the
        # existing method which handles caching, decoding, and result attachment.
        entities = self.entities_with_identifiers(entity_ids)

        # Step 3: build the catalog from the fully-loaded entities.
        experiments: dict[str, Experiment] = {}
        for entity in entities:
            for r in entity.experimentReferences:
                if (
                    r.actuatorIdentifier != "replay"
                    or r.experimentIdentifier in experiments
                ):
                    continue
                props = [
                    p for p in entity.observedProperties if p.experimentReference == r
                ]
                experiments[r.experimentIdentifier] = Experiment(
                    identifier=r.experimentIdentifier,
                    actuatorIdentifier=r.actuatorIdentifier,
                    targetProperties=[p.targetProperty for p in props],
                    requiredProperties=tuple(
                        ConstitutiveProperty.from_descriptor(p)
                        for p in entity.constitutiveProperties
                    ),
                )

        catalog: ExperimentCatalog | None = (
            ExperimentCatalog(
                experiments=experiments, catalogIdentifier="sqlstore_catalog"
            )
            if experiments
            else None
        )
        self._experiment_catalog = catalog
        return catalog

    def _create_source_table(self) -> sqlalchemy.MetaData:

        from sqlalchemy import CHAR, JSON, DateTime, Integer, String, Text

        # Create the tables if they don't exist
        meta = sqlalchemy.MetaData()

        sqlalchemy.Table(
            f"{self._tablename}",
            meta,
            sqlalchemy.Column("identifier", String(768), primary_key=True),
            sqlalchemy.Column("representation", Text(3145728)),
        )

        # Measurement-related tables
        sqlalchemy.Table(
            f"{self._tablename}_measurement_requests",
            meta,
            # Columns
            sqlalchemy.Column(
                "insert_id", Integer, primary_key=True, autoincrement=True
            ),
            sqlalchemy.Column("uid", CHAR(36), nullable=False, unique=True, index=True),
            sqlalchemy.Column("experiment_reference", Text(3145728), nullable=False),
            sqlalchemy.Column("operation_id", String(256), nullable=False),
            sqlalchemy.Column("request_index", Integer, nullable=False),
            sqlalchemy.Column("request_id", String(256), nullable=False),
            sqlalchemy.Column("type", String(256), nullable=False),
            sqlalchemy.Column("status", String(256), nullable=False),
            sqlalchemy.Column("metadata", JSON(False)),
            sqlalchemy.Column(
                "timestamp",
                DateTime(timezone=True),
                nullable=False,
                default=sqlalchemy.func.now(),
            ),
        )

        sqlalchemy.Table(
            f"{self._tablename}_measurement_results",
            meta,
            # Columns
            sqlalchemy.Column(
                "insert_id", Integer, primary_key=True, autoincrement=True
            ),
            sqlalchemy.Column("uid", CHAR(36), nullable=False, unique=True, index=True),
            sqlalchemy.Column("entity_id", Text(3145728), nullable=False),
            sqlalchemy.Column("data", JSON(False), nullable=False),
        )

        sqlalchemy.Table(
            f"{self._tablename}_measurement_requests_results",
            meta,
            # Columns
            sqlalchemy.Column(
                "insert_id", Integer, primary_key=True, autoincrement=True
            ),
            sqlalchemy.Column("uid", CHAR(36), nullable=False, unique=True, index=True),
            sqlalchemy.Column(
                "request_uid",
                CHAR(36),
                sqlalchemy.ForeignKey(f"{self._tablename}_measurement_requests.uid"),
                index=True,
                nullable=False,
            ),
            sqlalchemy.Column(
                "result_uid",
                CHAR(36),
                sqlalchemy.ForeignKey(f"{self._tablename}_measurement_results.uid"),
                index=True,
                nullable=False,
            ),
            sqlalchemy.Column("entity_index", Integer, nullable=False),
        )

        meta.create_all(self.engine, checkfirst=True)
        return meta

    def __init__(
        self,
        identifier: str | None,
        storageLocation: (
            ado.utilities.location.SQLStoreConfiguration | SQLiteStoreConfiguration
        ),
        parameters: dict,
    ) -> None:

        import uuid

        if identifier is None:
            # AP 26/09/2025:
            # This identifier could be a string that gets
            # parsed by --set as an int/float.
            # Examples are:
            # - 344846 -> interpreted as the number
            # - 5013e3 -> interpreted as 5013000.0
            # We check if this would happen and re-generate
            # the identifier if that's the case
            while True:
                identifier = str(uuid.uuid4())[:6]
                try:
                    float(identifier)
                except ValueError:
                    break

            parameters["identifier"] = identifier

        self._identifier = identifier
        self.log = logging.getLogger(f"sqlsource-{identifier}")
        self._parameters = parameters
        self._configuration = storageLocation
        if self._configuration is None:
            raise ValueError("SQLSampleStore requires valid location parameters.")

        self._tablename = f"sqlsource_{self._identifier}"
        self._engine = engine_for_sql_store(storageLocation)

        # Initialize entities cache as empty dict for lazy loading
        self._entities = {}
        self._all_entities_loaded = False
        self._last_insert_id = (
            0  # Track last processed insert_id for incremental refresh
        )
        # Tracks which entity identifiers have had all measurements fetched and attached.
        self._entities_with_measurements_loaded: set[str] = set()
        # Experiment-catalog cache.  _UNSET means "not yet computed".
        self._experiment_catalog: object = _UNSET

        # Create the four backing tables only when they do not yet exist.
        # The module level _source_tables_verified cache enables skipping
        # table creation checks for subsequent constructions within the same process.
        # The _reflected_metadata_cache stores the reflected metadata to avoid
        # repeated reflection operations.
        _cache_key = (str(self._engine.url), self._tablename)
        if _cache_key not in _source_tables_verified:
            # Reflect only the 4 tables related to this sample store
            metadata = sqlalchemy.MetaData()
            table_names = [
                self._tablename,
                f"{self._tablename}_measurement_requests",
                f"{self._tablename}_measurement_results",
                f"{self._tablename}_measurement_requests_results",
            ]

            try:
                metadata.reflect(bind=self.engine, only=table_names)
            except InvalidRequestError:
                # metadata.reflect raises an InvalidRequestError if one of the tables in only does not exist
                # Create tables and use the returned metadata which already has table definitions
                metadata = self._create_source_table()

            # Cache the metadata (already contains all tables including measurement tables)
            _reflected_metadata_cache[_cache_key] = metadata
            _source_tables_verified.add(_cache_key)

        # Use cached metadata
        self._metadata = _reflected_metadata_cache[_cache_key]
        self._request_table = self._metadata.tables[
            f"{self._tablename}_measurement_requests"
        ]
        self._request_result_table = self._metadata.tables[
            f"{self._tablename}_measurement_requests_results"
        ]
        self._result_table = self._metadata.tables[
            f"{self._tablename}_measurement_results"
        ]

        self.log.debug(f"SQLSampleStore id {self.uri}")

    # The SQLAlchemy Engine is not picklable, so anything using
    # Ray would fail. To avoid this, we remove it before pickling
    # and create a new instance when unpickling.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_engine"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._engine = engine_for_sql_store(self._configuration)

    @property
    def engine(self) -> sqlalchemy.Engine:
        return self._engine

    @property
    def config(self) -> dict:
        """Returns the parameters used to initialise the receiver"""

        return self._parameters.copy()

    @property
    def location(self) -> ado.utilities.location.SQLStoreConfiguration:

        return self._configuration.model_copy()

    @property
    def entities(self) -> list[Entity]:
        """Deprecated: use ``get_entities(require_measurements=True)`` instead.

        Returns all entities with all their measurement results attached.
        This property is deprecated and will be removed in a future release.
        """
        warnings.warn(
            "SQLSampleStore.entities is deprecated. "
            "Use get_entities(require_measurements=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_entities(require_measurements=True)

    def _fetch_entities(self, entity_ids: set[str] | None = None) -> dict[str, Entity]:
        """Fetch entities from the database and merge them into ``self._entities``.

        Parameters:
            entity_ids: Optional set of entity identifiers to fetch.
                       If None or empty set, fetches all entities.

        Returns:
            Dictionary mapping entity_identifier -> Entity object for the
            newly fetched entities (callers may use this for counting).

        Raises:
            SystemError: If database query fails
            FailedToDecodeStoredEntityError: If entity JSON is invalid
        """
        from sqlalchemy import select

        # Treat empty set same as None - fetch all entities
        if entity_ids is not None and len(entity_ids) == 0:
            entity_ids = None

        entity_table = self._metadata.tables[self._tablename]
        stmt = select(entity_table.c.identifier, entity_table.c.representation)
        if entity_ids is not None:
            stmt = stmt.where(entity_table.c.identifier.in_(entity_ids))

        try:
            with self.engine.begin() as connectable:
                cur = connectable.execute(stmt)
        except SQLAlchemyError as error:
            msg = f"Unable to fetch entities from sample store {self._tablename}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        entities = {}
        for entity_identifier, entity_representation in cur:
            try:
                entities[entity_identifier] = Entity.model_validate(
                    json.loads(entity_representation)
                )
            except Exception as error:  # noqa: PERF203
                raise FailedToDecodeStoredEntityError(
                    entity_identifier=entity_identifier,
                    entity_representation=entity_representation,
                    cause=error,
                ) from error

        self.log.debug(
            f"Fetched {len(entities)} entities"
            + (f" (filtered from {len(entity_ids)} requested)" if entity_ids else "")
        )
        # Always merge fetched entities into the cache.
        self._entities.update(entities)
        return entities

    def _fetch_measurement_results(
        self,
        min_insert_id: int = 0,
        entity_ids: set[str] | None = None,
    ) -> tuple[dict[str, list[ValidMeasurementResult]], int, int]:
        """Fetch measurement results from the database and attach them to cached entities.

        Fetches results, validates them, groups them by entity_id, attaches
        ``ValidMeasurementResult`` objects to the corresponding entities in
        ``self._entities``, and updates ``self._entities_with_measurements_loaded``
        for every entity that was in scope (even those with no results).

        Parameters:
            min_insert_id: Minimum insert_id to fetch (exclusive).
                          Use 0 to fetch all results.
            entity_ids: Optional set of entity identifiers to restrict the
                       query to.  When ``None`` (default) all entities are
                       considered.

        Returns:
            Tuple of:
            - Dictionary mapping entity_id -> list of ValidMeasurementResult objects
            - Maximum insert_id seen (or min_insert_id if no results)
            - Total number of measurement results successfully attached to entities

        Raises:
            SystemError: If database query fails
            FailedToDecodeStoredMeasurementResultForEntityError: If result JSON is invalid
        """
        from collections import defaultdict

        from sqlalchemy import select

        if entity_ids is not None and len(entity_ids) == 0:
            entity_ids = None

        res_table = self._result_table
        stmt = (
            select(res_table.c.insert_id, res_table.c.entity_id, res_table.c.data)
            .where(res_table.c.insert_id > min_insert_id)
            .order_by(res_table.c.insert_id)
        )
        if entity_ids is not None:
            stmt = stmt.where(res_table.c.entity_id.in_(entity_ids))

        try:
            with self.engine.begin() as connectable:
                cur = connectable.execute(stmt)
        except SQLAlchemyError as error:
            msg = f"Unable to fetch measurement results from sample store {self._tablename}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error
        results_by_entity = defaultdict(list)
        max_insert_id = min_insert_id

        for insert_id, entity_id, result_data in cur:
            max_insert_id = max(max_insert_id, insert_id)

            if result_data is None:
                self.log.debug(
                    f"Measurement result {insert_id} for entity {entity_id} had no data, skipping"
                )
                continue

            try:
                if not result_data.get("measurements", None):
                    continue

                measurement_result = ValidMeasurementResult.model_validate(result_data)
                results_by_entity[entity_id].append(measurement_result)
            except Exception as error:
                raise FailedToDecodeStoredMeasurementResultForEntityError(
                    entity_identifier=entity_id,
                    result_representation=result_data,
                    cause=error,
                ) from error

        # Attach results to cached entities and update the measurement-loaded tracking set.
        total_attached = 0
        for entity_id, measurement_results in results_by_entity.items():
            if entity_id not in self._entities:
                continue
            for measurement_result in measurement_results:
                try:
                    self._entities[entity_id].add_measurement_result(
                        result=measurement_result
                    )
                except DuplicateMeasurementResultError:  # noqa: PERF203
                    pass
                else:
                    total_attached += 1

        # Mark every requested entity (including those with no results) as loaded.
        requested_entity_ids = (
            entity_ids if entity_ids is not None else set(self._entities)
        )
        self._entities_with_measurements_loaded.update(
            requested_entity_ids.intersection(self._entities)
        )

        total_results = sum(len(results) for results in results_by_entity.values())
        self.log.debug(
            f"Fetched {total_results} measurement results for {len(results_by_entity)} entities "
            f"(insert_id range: {min_insert_id + 1} to {max_insert_id}), "
            f"attached {total_attached}"
        )

        return dict(results_by_entity), max_insert_id, total_attached

    def refresh(self, force_fetch_all_entities: bool = False) -> tuple[int, int]:
        """Refresh entities and fetch new measurement results.

        This method efficiently syncs the local cache with the database by:
        1. Fetching only new measurement results (insert_id > _last_insert_id)
        2. Fetching only entities that don't exist in cache yet (or all if forced)
        3. Attaching new measurements to existing or new entities

        Parameters:
            force_fetch_all_entities: If True, fetches all entities from database
                                     (used for initial load). If False, only fetches
                                     missing entities (used for incremental refresh).

        Returns:
            Tuple of (number of new entities fetched, number of new measurement results processed)

        Raises:
            SystemError: If database queries fail

        Example:
            >>> store = SQLSampleStore(...)
            >>> new_entities, new_results = store.refresh()
            >>> print(f"Fetched {new_entities} new entities and {new_results} new measurements")
        """
        self.log.debug(
            f"Refreshing entities for {self._tablename} "
            f"(last insert_id: {self._last_insert_id}, force_fetch_all={force_fetch_all_entities})"
        )

        new_entities_count = 0

        # Phase 1: Fetch entities
        if force_fetch_all_entities:
            # Initial load: clear cache then re-fetch everything.
            self._entities.clear()
            self._entities_with_measurements_loaded.clear()
            new_entities = self._fetch_entities(entity_ids=None)
            new_entities_count = len(new_entities)
            self._all_entities_loaded = True
            self.log.debug(f"Fetched all {new_entities_count} entities")

        # Phase 2: Fetch new measurement results (already validated, grouped, and attached
        # to self._entities as a side effect of _fetch_measurement_results).
        _results_by_entity, max_insert_id, total_measurements = (
            self._fetch_measurement_results(min_insert_id=self._last_insert_id)
        )

        if not _results_by_entity:
            self.log.debug("No new measurement results found")
            return (new_entities_count, 0)

        # Phase 3: Fetch missing entities.
        # Done even when force_fetch_all_entities=True to handle the race where
        # another process inserts an entity+results after our entity fetch but
        # before our measurement fetch.
        new_entity_ids = set(_results_by_entity.keys())
        missing_entity_ids = new_entity_ids - set(self._entities.keys())

        if missing_entity_ids:
            self.log.debug(f"Fetching {len(missing_entity_ids)} new entities")
            new_entities = self._fetch_entities(entity_ids=missing_entity_ids)
            fetched_count = len(new_entities)
            if not force_fetch_all_entities:
                new_entities_count = fetched_count

            if len(missing_entity_ids) != fetched_count:
                self.log.warning(
                    f"Expected to find {len(missing_entity_ids)} new entities but "
                    f"{fetched_count} were retrieved. This suggests another process "
                    f"is updating the sample store concurrently."
                )

            # Attach the already-fetched results to the newly cached entities.
            # _fetch_measurement_results ran before these entities were in cache,
            # so they were skipped during the attachment pass.
            for entity_id in missing_entity_ids:
                if entity_id not in self._entities:
                    continue
                for measurement_result in _results_by_entity.get(entity_id, []):
                    try:
                        self._entities[entity_id].add_measurement_result(
                            result=measurement_result
                        )
                    except DuplicateMeasurementResultError:  # noqa: PERF203
                        pass
                    else:
                        total_measurements += 1
                self._entities_with_measurements_loaded.add(entity_id)

        # Update tracking
        self._last_insert_id = max_insert_id

        self.log.info(
            f"Refresh complete: fetched {new_entities_count} new entities, "
            f"processed {total_measurements} new measurements "
            f"(last insert_id: {max_insert_id})"
        )

        return (new_entities_count, total_measurements)

    def entities_with_identifiers(
        self, entity_identifiers: set[str] | list[str]
    ) -> list[Entity]:
        """Efficiently fetch entities by their identifiers without loading all entities.

        This method queries only the specified entities from the database, making it
        much more efficient than loading all entities and filtering in Python.

        Args:
            entity_identifiers: Set or list of entity identifiers to fetch

        Returns:
            List of Entity objects matching the provided identifiers
        """
        if not entity_identifiers:
            return []

        # Convert to set for deduplication and efficient lookup
        entity_ids_set = (
            set(entity_identifiers)
            if isinstance(entity_identifiers, list)
            else entity_identifiers
        )

        # Partition into cached and uncached IDs
        cached_keys = (
            entity_ids_set.intersection(self._entities.keys())
            if self._entities
            else set()
        )
        uncached_ids = entity_ids_set.difference(cached_keys)
        cached_entities = [self._entities[k] for k in cached_keys]

        # All requested entities were already cached
        if not uncached_ids:
            return cached_entities

        # Query database only for the uncached entities
        # Use SQLAlchemy's expanding bindparam for IN clause
        # This automatically handles the parameter expansion for the IN clause
        query = sqlalchemy.text(f"""
            SELECT ent.identifier, ent.representation, res.data
            FROM {self._tablename} ent
            LEFT OUTER JOIN {self._tablename}_measurement_results res ON res.entity_id = ent.identifier
            WHERE ent.identifier IN :entity_ids
        """).bindparams(  # noqa: S608 - self._tablename is not untrusted
            sqlalchemy.bindparam(
                key="entity_ids", value=list(uncached_ids), expanding=True
            )
        )

        try:
            with self.engine.begin() as connectable:
                cur = connectable.execute(query)
        except SQLAlchemyError as error:
            msg = f"Unable to fetch entities by identifiers from sample store {self._tablename}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        # Build result dictionary to handle multiple measurement results per entity
        entities_dict: dict[str, Entity] = {}
        for entity_identifier, entity_representation, result_data in cur:
            if entity_identifier not in entities_dict:
                try:
                    entities_dict[entity_identifier] = Entity.model_validate(
                        json.loads(entity_representation)
                    )
                    # Update cache if it exists
                    if self._entities is not None:
                        self._entities[entity_identifier] = entities_dict[
                            entity_identifier
                        ]
                except Exception as error:
                    raise FailedToDecodeStoredEntityError(
                        entity_identifier=entity_identifier,
                        entity_representation=entity_representation,
                        cause=error,
                    ) from error

            if result_data is None:
                self.log.debug(
                    f"Entity {entity_identifier} had no measurements associated to it."
                )
                continue

            try:
                result_dict = json.loads(result_data)
                if not result_dict.get("measurements", None):
                    continue

                measurement_result = ValidMeasurementResult.model_validate(result_dict)
            except Exception as error:
                raise FailedToDecodeStoredMeasurementResultForEntityError(
                    entity_identifier=entity_identifier,
                    result_representation=result_data,
                    cause=error,
                ) from error

            # Add measurement result to entity
            entities_dict[entity_identifier].add_measurement_result(
                result=measurement_result
            )

        return cached_entities + list(entities_dict.values())

    def get_entities(
        self,
        identifiers: str | set[str] | None = None,
        *,
        require_measurements: bool,
        refresh: bool = False,
    ) -> list[Entity]:
        """Retrieve entities from the store with optional measurement attachment.

        This is the preferred method for fetching entities.  It maximises cache
        reuse — only data that is not already cached is fetched from the database.

        Args:
            identifiers: Which entities to return.
                - ``None`` (default): all entities.
                - ``str``: a single entity identifier (normalised to a one-item set).
                - ``set[str]``: an explicit subset of entity identifiers.
            require_measurements: When ``True``, measurement results are fetched
                and attached to every returned entity before the list is returned.
                Entities that already have their measurements loaded (tracked via
                ``_entities_with_measurements_loaded``) are not re-queried.
                Defaults to ``False``.
            refresh: When ``True``, the relevant cache entries are evicted before
                fetching so that the database is always re-queried.
                - For ``identifiers=None``: clears all cached data and resets
                  ``_last_insert_id`` to 0.
                - For a specific subset: removes only those ids from ``_entities``
                  and ``_entities_with_measurements_loaded``; the global
                  ``_last_insert_id`` cursor is preserved.
                Defaults to ``False``.

        Returns:
            List of ``Entity`` objects.  When ``identifiers`` is a subset, only
            entities that exist in the store are included (missing ids are silently
            omitted).
        """
        # --- normalise identifiers ---
        if isinstance(identifiers, str):
            requested_ids: set[str] | None = {identifiers}
        else:
            requested_ids = identifiers  # None or set[str]

        # --- selective cache invalidation ---
        if refresh:
            if requested_ids is None:
                self._entities.clear()
                self._entities_with_measurements_loaded.clear()
                self._all_entities_loaded = False
                self._last_insert_id = 0
            else:
                for eid in requested_ids:
                    self._entities.pop(eid, None)
                self._entities_with_measurements_loaded.difference_update(requested_ids)

        # --- entity fetch ---
        if requested_ids is None:
            if not self._all_entities_loaded:
                self._fetch_entities(entity_ids=None)
                self._all_entities_loaded = True
        else:
            uncached_ids = requested_ids.difference(self._entities)
            if uncached_ids:
                self._fetch_entities(entity_ids=uncached_ids)

        # --- optional measurement fetch ---
        if require_measurements:
            if requested_ids is None:
                missing_measurement_ids = set(self._entities).difference(
                    self._entities_with_measurements_loaded
                )
            else:
                missing_measurement_ids = requested_ids.difference(
                    self._entities_with_measurements_loaded
                )

            if missing_measurement_ids:
                self._fetch_measurement_results(
                    min_insert_id=0,
                    entity_ids=missing_measurement_ids,
                )

        # --- build result list ---
        if requested_ids is None:
            return list(self._entities.values())
        return [self._entities[eid] for eid in requested_ids if eid in self._entities]

    def entities_in_operations(self, operation_ids: str | set[str]) -> list[Entity]:
        """Get entities sampled in one or more operations.

        Args:
            operation_ids: A single operation identifier or a set of operation
                identifiers to fetch entities for.

        Returns:
            List of Entity objects that were sampled in the specified operation(s)
        """
        # Use entity_identifiers_in_operations + entities_with_identifiers so that
        # the entity cache is used when fetching entities.
        entity_ids = self.entity_identifiers_in_operations(operation_ids)
        return self.entities_with_identifiers(entity_ids)

    def entities_in_operation(self, operation_ids: str | set[str]) -> list[Entity]:
        """Deprecated: use entities_in_operations instead."""
        warnings.warn(
            "entities_in_operation is deprecated, use entities_in_operations instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.entities_in_operations(operation_ids)

    @property
    def numberOfEntities(self) -> int:

        with self.engine.connect() as connectable:
            query = sqlalchemy.text(
                f"SELECT count(*) FROM {self._tablename}"  # noqa: S608 - self._tablename is not untrusted
            )
            exe = connectable.execute(query)
            return exe.scalar()

    def containsEntityWithIdentifier(self, entity_id: str) -> bool:
        query = sqlalchemy.text(
            "SELECT COUNT(1) FROM :table_name WHERE identifier=:identifier"
        ).bindparams(table_name=self._tablename, identifier=entity_id)

        with self.engine.connect() as connectable:
            exe = connectable.execute(query)
            row_count = exe.scalar()

        return row_count != 0

    @property
    def identifier(self) -> str:
        """Return a unique identifier for this configuration of the sample store"""

        return self._identifier

    def addEntities(self, entities: list[Entity]) -> None:
        """
        Add the entities to the sample store.

        Entities that are added to the database are stripped of the observed property values.
        This does not affect the list that is passed to the function.
        """

        for entity in entities:
            self._entities[entity.identifier] = entity

        for index in range(0, len(entities), 5000):
            values = [
                {
                    "identifier": e.identifier,
                    "representation": e.model_dump_json(
                        exclude_defaults=True, exclude={"measurement_results"}
                    ),
                }
                for e in entities
            ]

            self.log.debug(f"Inserting {len(values)} entities")

            try:
                # Remote
                with self.engine.begin() as connectable:
                    query = ado.metastore.sql.statements.insert_entities_ignore_on_duplicate(
                        sample_store_name=self._tablename,
                        dialect=self.engine.dialect.name,
                    )
                    connectable.execute(query, values)
            except SQLAlchemyError as error:
                self.log.critical(
                    f"Failed to insert entity batch starting from {index}. Error: {error}"
                )
                raise SystemError(
                    f"Failed to insert entity batch starting from {index}. Error: {error}"
                ) from error

    def upgrade_entities(self) -> int:
        """Re-serialize all entity rows through the current Pydantic model in a single atomic UPDATE.

        Returns:
            Number of entity rows upgraded.

        Raises:
            FailedToDecodeStoredEntityError: If any entity cannot be deserialized.
            SystemError: If the SQL UPDATE does not match the expected row count.
        """
        from sqlalchemy import select

        entity_table = self._metadata.tables[self._tablename]
        stmt = select(entity_table.c.identifier, entity_table.c.representation)

        with self.engine.begin() as conn:
            rows = list(conn.execute(stmt))
            expected_count = len(rows)
            if expected_count == 0:
                return 0

            values = []
            for entity_identifier, entity_representation in rows:
                try:
                    entity = Entity.model_validate(json.loads(entity_representation))
                except Exception as error:
                    raise FailedToDecodeStoredEntityError(
                        entity_identifier=entity_identifier,
                        entity_representation=entity_representation,
                        cause=error,
                    ) from error
                values.append(
                    {
                        "identifier": entity_identifier,
                        "representation": entity.model_dump_json(
                            exclude_defaults=True, exclude={"measurement_results"}
                        ),
                    }
                )

            # b_ prefix avoids a name collision: SQLAlchemy uses column names as
            # implicit bind parameters in the executemany VALUES clause, so the
            # explicit WHERE-clause parameters need distinct names.
            result = conn.execute(
                entity_table.update()
                .where(
                    entity_table.c.identifier == sqlalchemy.bindparam("b_identifier")
                )
                .values(representation=sqlalchemy.bindparam("b_representation")),
                [
                    {
                        "b_identifier": v["identifier"],
                        "b_representation": v["representation"],
                    }
                    for v in values
                ],
            )

        if result.rowcount != expected_count:
            raise SystemError(
                f"upgrade_entities: expected to update {expected_count} rows "
                f"but updated {result.rowcount}"
            )

        self.log.debug(f"upgrade_entities: upgraded {expected_count} entity rows")
        return expected_count

    def add_external_entities(self, entities: list[Entity]) -> None:

        existing_entity_ids = self.entity_identifiers()
        missing_entities = [
            entity
            for entity in entities
            if entity.identifier not in existing_entity_ids
        ]
        missing_measurements = []
        for entity in missing_entities:
            missing_measurements.extend(entity.measurement_results)

        # Importing replay results invalidates the experiment-catalog cache.
        self._experiment_catalog = _UNSET
        self.addEntities(entities=missing_entities)
        self.add_measurement_results(
            results=missing_measurements, skip_relationship_to_request=True
        )

    def addMeasurement(
        self,
        measurementRequest: ado.schema.request.MeasurementRequest,
    ) -> None:
        """Adds the results of a measurement to a set of entities

        Implementations of this method can require that the results have been already added to the
        Entities OR that measurementRequest.results is required instead.

        """

        for entity in measurementRequest.entities:
            self._entities[entity.identifier] = entity

        request_db_id = self.add_measurement_request(request=measurementRequest)

        if isinstance(measurementRequest, ReplayedMeasurement):
            # Replay results invalidate the experiment-catalog cache.
            self._experiment_catalog = _UNSET
            try:
                self.add_relationship_between_request_and_results(
                    request_db_id, measurementRequest.measurements
                )
                return
            except SystemError as e:
                # We're likely in the case where the result has been deleted
                # while the operation was running. We will try to add the
                # results again
                self.log.exception(
                    "Exception while trying to add a relationship between "
                    "measurement requests and results",
                    e,
                )

        self.add_measurement_results(
            results=measurementRequest.measurements,
            skip_relationship_to_request=False,
            request_db_id=request_db_id,
        )

    def close(self) -> None:

        pass

    def delete(self) -> None:

        pass

    def entityWithIdentifier(self, entityIdentifier: str) -> Entity | None:
        """Returns entity if its in receiver otherwise returns None"""

        query = sqlalchemy.text(f"""
                SELECT ent.identifier, ent.representation, res.data
                FROM (
                    SELECT identifier, representation
                    FROM {self._tablename} ent
                    WHERE identifier = :identifier
                ) ent
                LEFT OUTER JOIN {self._tablename}_measurement_results res ON ent.identifier = res.entity_id
            """).bindparams(  # noqa: S608 - self._tablename is not untrusted
            identifier=entityIdentifier
        )

        try:
            with self.engine.begin() as connectable:
                cur = connectable.execute(query)
        except SQLAlchemyError as error:
            msg = f"Unable to fetch entity {entityIdentifier} and measurements from sample store {self._tablename}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        entity = None
        failures = 0
        for entity_identifier, entity_representation, result_data in cur:
            if entity is None:
                try:
                    entity = Entity.model_validate(json.loads(entity_representation))
                except Exception as error:
                    self.log.warning(
                        f"Unable to decode representation for entity {entity_identifier}.\n"
                        f"Representation was: {entity_representation}.\n"
                        f"Error was {error}"
                    )
                    return None

            if result_data is None:
                self.log.info(
                    f"Entity {entity_identifier} had no measurements associated to it."
                )
                continue

            try:
                result_dict = json.loads(result_data)
                if not result_dict.get("measurements", None):
                    continue

                measurement_result = ValidMeasurementResult.model_validate(result_dict)
            except Exception as error:
                self.log.warning(
                    f"Unable to decode a measurement result for entity {entity_identifier}.\n"
                    f"Data was: {result_data}.\n"
                    f"Error was {error}"
                )
                failures += 1
                continue

            # We need to manually add valid measurements to the entity
            entity.add_measurement_result(result=measurement_result)

        return entity

    @property
    def uri(self) -> str:
        """Returns a URI for the Active Source - password is elided"""

        return (
            f"sqlite:///{self._configuration.path}"
            if self._configuration.scheme == "sqlite"
            else self._configuration.url(hide_pw=True).unicode_string()
        ) + f"/{self._tablename}"

    @staticmethod
    def validate_parameters(parameters: dict) -> dict:

        # No parameters to validate
        return parameters

    @staticmethod
    def storage_location_class() -> type[
        SQLiteStoreConfiguration | SQLStoreConfiguration
    ]:
        return SQLiteStoreConfiguration | SQLStoreConfiguration

    def add_measurement_request(self, request: MeasurementRequest) -> uuid.uuid4:

        db_id = uuid.uuid4()

        # We need to add entities in case they're missing
        # We use the "ignore" semantic on duplicates provided by
        # addEntities to just try to insert them
        if not isinstance(request, ReplayedMeasurement):
            self.addEntities(request.entities)

        try:
            with self.engine.begin() as connectable:
                query = sqlalchemy.text(f"""
                    INSERT INTO {self._tablename}_measurement_requests
                    (uid, experiment_reference, operation_id, request_index, request_id, type, status, metadata, timestamp)
                    VALUES (:uid, :experiment_reference, :operation_id, :request_index, :request_id, :type, :status, :metadata, :timestamp)
                    """).bindparams(  # noqa: S608 - self._tablename is not untrusted
                    uid=str(db_id),
                    experiment_reference=str(request.experimentReference),
                    operation_id=request.operation_id,
                    request_index=request.requestIndex,
                    request_id=request.requestid,
                    type=request.__class__.__name__,
                    status=request.status.value,
                    metadata=json.dumps(request.metadata),
                    timestamp=request.timestamp,
                )
                connectable.execute(query)

                return db_id
        except SQLAlchemyError as error:
            self.log.critical(f"Failed to add measurement request. Error: {error}")
            raise SystemError(
                f"Failed to add measurement request. Error: {error}"
            ) from error

    def entity_identifiers(self) -> set[str]:

        query = sqlalchemy.text(
            f"""SELECT identifier FROM {self._tablename}"""  # noqa: S608 - self._tablename is not untrusted
        )

        try:
            with self.engine.begin() as connectable:
                cur = connectable.execute(query)
        except SQLAlchemyError as error:
            msg = f"Failed to load identifiers from sample store {self._tablename}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        return {row[0] for row in cur}

    def add_measurement_results(
        self,
        results: list[MeasurementResult],
        skip_relationship_to_request: bool,
        request_db_id: uuid.UUID | None = None,
    ) -> None:
        if len(results) == 0:
            return

        if not request_db_id and not skip_relationship_to_request:
            raise ValueError(
                "request_db_id cannot be None when skip_relationship_to_request is false"
            )

        prepared_results = [
            {
                "uid": r.uid,
                "entity_id": r.entityIdentifier,
                "data": r.model_dump_json(),
            }
            for r in results
        ]

        try:
            with self.engine.begin() as connectable:
                query = sqlalchemy.text(f"""
                    INSERT INTO {self._tablename}_measurement_results
                    (uid, entity_id, data)
                    VALUES (:uid, :entity_id, :data)
                    """)  # noqa: S608 - self._tablename is not untrusted
                connectable.execute(query, prepared_results)
        except SQLAlchemyError as error:
            self.log.critical(f"Failed to add measurement results. Error: {error}")
            raise SystemError(
                f"Failed to add measurement results. Error: {error}"
            ) from error

        if skip_relationship_to_request:
            return

        self.add_relationship_between_request_and_results(request_db_id, results)

    def upgrade_measurement_results(self) -> int:
        """Re-serialize all measurement result rows through the current Pydantic model in a single atomic UPDATE.

        Returns:
            Number of measurement result rows upgraded.

        Raises:
            FailedToDecodeStoredMeasurementResultForEntityError: If any result
                cannot be deserialized.
            SystemError: If the SQL UPDATE does not match the expected row count.
        """
        from sqlalchemy import select

        res_table = self._result_table
        stmt = select(res_table.c.uid, res_table.c.data)

        with self.engine.begin() as conn:
            rows = list(conn.execute(stmt))
            expected_count = len(rows)
            if expected_count == 0:
                return 0

            values = []
            for uid, result_data in rows:
                try:
                    if "reason" in result_data:
                        measurement_result = InvalidMeasurementResult.model_validate(
                            result_data
                        )
                    else:
                        measurement_result = ValidMeasurementResult.model_validate(
                            result_data
                        )
                except Exception as error:
                    raise FailedToDecodeStoredMeasurementResultForEntityError(
                        entity_identifier=result_data.get("entityIdentifier", uid),
                        result_representation=result_data,
                        cause=error,
                    ) from error
                values.append(
                    {
                        "uid": uid,
                        "data": json.loads(measurement_result.model_dump_json()),
                    }
                )

            # b_ prefix avoids a name collision: SQLAlchemy uses column names as
            # implicit bind parameters in the executemany VALUES clause, so the
            # explicit WHERE-clause parameters need distinct names.
            result = conn.execute(
                res_table.update()
                .where(res_table.c.uid == sqlalchemy.bindparam("b_uid"))
                .values(data=sqlalchemy.bindparam("b_data")),
                [{"b_uid": v["uid"], "b_data": v["data"]} for v in values],
            )

        if result.rowcount != expected_count:
            raise SystemError(
                f"upgrade_measurement_results: expected to update {expected_count} rows "
                f"but updated {result.rowcount}"
            )

        self.log.debug(
            f"upgrade_measurement_results: upgraded {expected_count} result rows"
        )
        return expected_count

    def add_relationship_between_request_and_results(
        self,
        request_db_id: uuid.uuid4,
        results: list[MeasurementResult],
    ) -> None:

        # 24/04/2025 AP:
        # casting the UUIDs to string because SQLite
        # can't otherwise do it. MySQL worked.
        # Note that result_uid was already a string.
        prepared_relationships = [
            {
                "uid": str(uuid.uuid4()),
                "request_uid": str(request_db_id),
                "result_uid": r.uid,
                "entity_index": idx,
            }
            for idx, r in enumerate(results)
        ]

        try:
            with self.engine.begin() as connectable:
                query = sqlalchemy.text(f"""
                    INSERT INTO {self._tablename}_measurement_requests_results
                    (uid, request_uid, result_uid, entity_index)
                    VALUES (:uid, :request_uid, :result_uid, :entity_index)
                    """)  # noqa: S608 - self._tablename is not untrusted
                connectable.execute(query, prepared_relationships)
        except SQLAlchemyError as error:
            self.log.critical(
                f"Failed to add link between measurement requests and results. Error: {error}"
            )
            raise SystemError(
                f"Failed to add link between measurement requests and results. Error: {error}"
            ) from error

    def measurement_requests_count_for_operation(
        self,
        operation_id: str,
        experiment_filter: str | None = None,
        status_filter: MeasurementRequestStateEnum | None = None,
    ) -> int:

        query_text = f"""
                        SELECT COUNT(uid)
                        FROM {self._tablename}_measurement_requests
                        WHERE operation_id = :operation_id
                    """  # noqa: S608 - self._tablename is not untrusted
        query_parameters = {"operation_id": operation_id}

        if status_filter:
            query_text += "AND status = :status_filter "
            query_parameters["status_filter"] = status_filter.value

        if experiment_filter:
            query_text += "AND experiment_reference = :experiment_filter "
            query_parameters["experiment_filter"] = experiment_filter

        try:
            with self.engine.begin() as connectable:
                query = sqlalchemy.text(query_text).bindparams(**query_parameters)
                return connectable.execute(query).first()[0]
        except SQLAlchemyError as error:
            msg = f"Unable to get the count of measurement requests for operation {operation_id}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

    def measurement_results_count_for_operation(
        self,
        operation_id: str,
        experiment_filter: str | None = None,
        status_filter: MeasurementResultStateEnum | None = None,
    ) -> int:
        result_state_map = {
            MeasurementResultStateEnum.VALID: "measurements",
            MeasurementResultStateEnum.INVALID: "reason",
        }

        query_parameters = {"operation_id": operation_id}
        inner_query = f"""
                        SELECT uid
                        FROM {self._tablename}_measurement_requests
                        WHERE operation_id = :operation_id
                        """  # noqa: S608 - self._tablename is not untrusted

        if experiment_filter:
            inner_query += "AND experiment_reference = :experiment_filter"
            query_parameters["experiment_filter"] = experiment_filter

        query_text = f"""
                        SELECT COUNT(uid)
                        FROM {self._tablename}_measurement_requests_results
                        WHERE request_uid IN ({inner_query})
                    """  # noqa: S608 - self._tablename is not untrusted and inner_query has been sanitized

        if status_filter:
            query_text += "AND :status_filter MEMBER OF(JSON_KEYS(data))"
            query_parameters["status_filter"] = result_state_map[status_filter]

        try:
            with self.engine.begin() as connectable:
                query = sqlalchemy.text(query_text).bindparams(**query_parameters)
                return connectable.execute(query).first()[0]
        except SQLAlchemyError as error:
            msg = f"Unable to get the count of measurement results for operation {operation_id}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

    def operation_entity_statistics(self, operation_id: str) -> dict[str, int]:
        """
        Compute entity-level statistics for an operation.

        This method efficiently computes statistics without fetching all measurement
        results, using SQL COUNT queries instead of loading data into Python.

        Args:
            operation_id: The operation identifier

        Returns:
            Dictionary with keys:
            - 'entities_with_all_successful_measurements': count of entities where
              ALL measurements succeeded
            - 'entities_with_at_least_one_successful_measurement': count of entities
              with at least one successful measurement
            - 'total_entities': total distinct entities measured in the operation
        """
        try:
            with self.engine.begin() as connectable:
                # Only the request has information on what operation it belongs to, as
                # results can be associated with multiple requests. For this reason, we
                # start by selecting all the requests that belong to the operation we
                # are interested in, then join them to their results via the reqres table.
                #
                # We then select the entity identifier, along with the total number of
                # results associated to it (counting on the index for speed) and the
                # number of valid measurement results by checking how many results do not
                # have the `reason` field (only found in invalid measurement results).
                #
                # We can then use this information to compute the fields we are interested in.
                query_text = f"""
                    SELECT
                        COUNT(DISTINCT entity_stats.entity_id) as total_entities,
                        COUNT(
                            DISTINCT CASE
                            WHEN valid_measurements > 0
                            THEN entity_stats.entity_id END
                        ) as entities_with_at_least_one_successful,
                        COUNT(
                            DISTINCT CASE
                            WHEN valid_measurements = total_measurements
                            AND total_measurements > 0
                            THEN entity_stats.entity_id END
                        ) as entities_with_all_successful
                    FROM (
                        SELECT
                            res.entity_id,
                            COUNT(res.uid) as total_measurements,
                            SUM(CASE WHEN JSON_EXTRACT(res.data, '$.reason') IS NULL THEN 1 ELSE 0 END) as valid_measurements
                        FROM {self._tablename}_measurement_requests req
                        JOIN {self._tablename}_measurement_requests_results reqres ON reqres.request_uid = req.uid
                        JOIN {self._tablename}_measurement_results res ON reqres.result_uid = res.uid
                        WHERE req.operation_id = :operation_id
                        GROUP BY res.entity_id
                    ) AS entity_stats
                """  # noqa: S608 - self._tablename is not untrusted

                query = sqlalchemy.text(query_text).bindparams(
                    operation_id=operation_id
                )
                cur = connectable.execute(query)
                (
                    total_entities,
                    entities_with_at_least_one_successful,
                    entities_with_all_successful,
                ) = cur.one()

                return {
                    "entities_with_all_successful_measurements": entities_with_all_successful,
                    "entities_with_at_least_one_successful_measurement": entities_with_at_least_one_successful,
                    "total_entities": total_entities,
                }
        except SQLAlchemyError as error:
            msg = f"Unable to get entity statistics for operation {operation_id}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

    def entities_with_valid_measurements(
        self, entity_identifiers: set[str]
    ) -> set[str]:
        """Return the subset of entity_identifiers that has at least one valid measurement.

        Args:
            entity_identifiers: Set of entity identifier strings to check.

        Returns:
            Subset of *entity_identifiers* for which at least one valid measurement
            result exists.

        Raises:
            SystemError: If the underlying SQL query fails.
        """
        if not entity_identifiers:
            return set()

        try:
            from sqlalchemy import select

            res_table = self._result_table
            # A measurement result is valid when the ``reason`` field is absent
            # from its stored JSON blob.
            stmt = (
                select(res_table.c.entity_id)
                .where(res_table.c.entity_id.in_(entity_identifiers))
                .where(
                    sqlalchemy.func.json_extract(res_table.c.data, "$.reason").is_(None)
                )
                .distinct()
            )
            with self.engine.begin() as connectable:
                rows = connectable.execute(stmt).fetchall()
            return {row.entity_id for row in rows}
        except SQLAlchemyError as error:
            msg = "Unable to get entities with valid measurement"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

    def entity_experiment_references(
        self, entity_identifiers: set[str]
    ) -> "dict[str, set[ExperimentReference]]":
        """Return the experiment references covered by valid measurements per entity.

        Args:
            entity_identifiers: Set of entity identifier strings to query.

        Returns:
            ``dict`` mapping each entity identifier (only those with ≥1 valid result)
            to the set of :class:`~ado.schema.reference.ExperimentReference` objects
            for which they have a valid result.

        Raises:
            SystemError: If the underlying SQL query fails.
        """
        if not entity_identifiers:
            return {}

        # Uses COALESCE to handle both the current (compressed) and legacy
        # serialization formats stored in the database:
        # - Current format: ``experimentReference`` at top level of the ``data`` blob.
        # - Legacy format: ``experimentReference`` inside the first measurement's property.
        try:
            from sqlalchemy import select

            res_table = self._result_table
            exp_ref_col = sqlalchemy.func.coalesce(
                sqlalchemy.func.json_extract(res_table.c.data, "$.experimentReference"),
                sqlalchemy.func.json_extract(
                    res_table.c.data,
                    "$.measurements[0].property.experimentReference",
                ),
            ).label("experiment_reference_json")
            stmt = (
                select(res_table.c.entity_id, exp_ref_col)
                .where(res_table.c.entity_id.in_(entity_identifiers))
                .where(
                    sqlalchemy.func.json_extract(res_table.c.data, "$.reason").is_(None)
                )
                .distinct()
            )

            with self.engine.begin() as connectable:
                rows = connectable.execute(stmt).fetchall()
            result: dict[str, set[ExperimentReference]] = {}
            for row in rows:
                blob = row.experiment_reference_json
                if blob is None:
                    continue
                exp_ref = ExperimentReference.model_validate_json(blob)
                result.setdefault(row.entity_id, set()).add(exp_ref)
            return result
        except SQLAlchemyError as error:
            msg = "Unable to get entity experiment references"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

    def operation_measurement_statistics(
        self, operation_ids: set[str] | None = None
    ) -> "list[OperationMeasurementStatistics]":
        """Compute aggregated measurement statistics for one or more operations.

        Computes all statistics in a single query grouped by operation_id to
        avoid multiple DB round-trips: request counts (total/failed/successful),
        result counts (total/valid/invalid), and distinct measured entity count.

        Args:
            operation_ids: Set of operation identifiers to aggregate. Pass
                ``None`` to aggregate across all operations in the store.
                Passing an empty set raises ``ValueError``.

        Returns:
            A list of OperationMeasurementStatistics instances, one per
            operation found in the store.

        Raises:
            ValueError: If ``operation_ids`` is an empty set.
        """
        if operation_ids is not None and len(operation_ids) == 0:
            raise ValueError("operation_ids must be a non-empty set or None")

        from sqlalchemy import case, func, select

        from ado.core.operation.stats import OperationMeasurementStatistics

        req_table = self._request_table
        reqres_table = self._request_result_table
        res_table = self._result_table

        # Equivalent SQL:
        #
        #   SELECT
        #     req.operation_id,
        #     COUNT(DISTINCT req.uid) AS total_requests,
        #     COUNT(DISTINCT CASE WHEN req.status = 'Failed' THEN req.uid END) AS failed_requests,
        #     COUNT(DISTINCT CASE WHEN req.status = 'Success' THEN req.uid END) AS successful_requests,
        #     COUNT(res.uid) AS total_results,
        #     SUM(CASE WHEN JSON_EXTRACT(res.data, '$.reason') IS NULL     THEN 1 ELSE 0 END) AS successful_results,
        #     SUM(CASE WHEN JSON_EXTRACT(res.data, '$.reason') IS NOT NULL THEN 1 ELSE 0 END) AS failed_results,
        #     COUNT(DISTINCT res.entity_id) AS measured_entities
        #   FROM <tablename>_measurement_requests req
        #   LEFT JOIN <tablename>_measurement_requests_results reqres ON reqres.request_uid = req.uid
        #   LEFT JOIN <tablename>_measurement_results            res  ON reqres.result_uid  = res.uid
        #   [WHERE req.operation_id IN (...)]          -- only when operation_ids is not None
        #   GROUP BY req.operation_id

        reason_is_null = func.json_extract(res_table.c.data, "$.reason").is_(None)

        stmt = (
            select(
                # req.operation_id
                req_table.c.operation_id,
                # COUNT(DISTINCT req.uid) AS total_requests,
                func.count(func.distinct(req_table.c.uid)).label("total_requests"),
                # COUNT(DISTINCT CASE WHEN req.status = 'Failed' THEN req.uid END) AS failed_requests,
                func.count(
                    func.distinct(
                        case(
                            (
                                req_table.c.status
                                == MeasurementRequestStateEnum.FAILED.value,
                                req_table.c.uid,
                            ),
                            else_=None,
                        )
                    )
                ).label("failed_requests"),
                # COUNT(DISTINCT CASE WHEN req.status = 'Success' THEN req.uid END) AS successful_requests,
                func.count(
                    func.distinct(
                        case(
                            (
                                req_table.c.status
                                == MeasurementRequestStateEnum.SUCCESS.value,
                                req_table.c.uid,
                            ),
                            else_=None,
                        )
                    )
                ).label("successful_requests"),
                # COUNT(res.uid) AS total_results,
                func.count(res_table.c.uid).label("total_results"),
                # SUM(CASE WHEN JSON_EXTRACT(res.data, '$.reason') IS NULL THEN 1 ELSE 0 END) AS successful_results,
                func.sum(case((reason_is_null, 1), else_=0)).label(
                    "successful_results"
                ),
                # SUM(CASE WHEN JSON_EXTRACT(res.data, '$.reason') IS NOT NULL THEN 1 ELSE 0 END) AS failed_results,
                func.sum(case((~reason_is_null, 1), else_=0)).label("failed_results"),
                # COUNT(DISTINCT res.entity_id) AS measured_entities
                func.count(func.distinct(res_table.c.entity_id)).label(
                    "measured_entities"
                ),
            )
            # FROM <tablename>_measurement_requests req
            .select_from(req_table)
            # LEFT JOIN <tablename>_measurement_requests_results reqres ON reqres.request_uid = req.uid
            .outerjoin(reqres_table, reqres_table.c.request_uid == req_table.c.uid)
            # LEFT JOIN <tablename>_measurement_results res ON reqres.result_uid = res.uid
            .outerjoin(res_table, reqres_table.c.result_uid == res_table.c.uid)
            # GROUP BY req.operation_id
            .group_by(req_table.c.operation_id)
        )

        if operation_ids is not None:
            stmt = stmt.where(req_table.c.operation_id.in_(operation_ids))

        try:
            with self.engine.begin() as connectable:
                rows = connectable.execute(stmt).all()
                return [
                    OperationMeasurementStatistics(
                        operation_id=row.operation_id,
                        total_requests=row.total_requests or 0,
                        failed_requests=row.failed_requests or 0,
                        successful_requests=row.successful_requests or 0,
                        total_results=row.total_results or 0,
                        successful_results=row.successful_results or 0,
                        failed_results=row.failed_results or 0,
                        measured_entities=row.measured_entities or 0,
                    )
                    for row in rows
                ]
        except SQLAlchemyError as error:
            msg = f"Unable to get measurement statistics for operation IDs {operation_ids}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

    def samplestore_statistics(self) -> "SampleStoreStatistics":
        """Compute aggregate statistics for this sample store in a single query.

        Issues exactly one query that returns three scalar counts via subqueries:
        total entities, total measurement results, and distinct experiment
        references.

        Returns:
            A :class:`~ado.core.samplestore.stats.SampleStoreStatistics`
            instance with all three counters populated.

        Raises:
            SystemError: If the underlying database query fails.
        """
        from sqlalchemy import func, select

        from ado.core.samplestore.stats import SampleStoreStatistics

        entity_table = self._metadata.tables[self._tablename]
        req_table = self._request_table
        res_table = self._result_table

        # Equivalent SQL:
        #
        #   SELECT
        #     (SELECT COUNT(identifier)
        #        FROM sqlsource_{id})                              AS number_of_entities,
        #     (SELECT COUNT(uid)
        #        FROM sqlsource_{id}_measurement_results)          AS number_of_results,
        #     (SELECT COUNT(DISTINCT experiment_reference)
        #        FROM sqlsource_{id}_measurement_requests)         AS number_of_experiments

        stmt = select(
            select(func.count(entity_table.c.identifier))
            .scalar_subquery()
            .label("number_of_entities"),
            select(func.count(res_table.c.uid))
            .scalar_subquery()
            .label("number_of_results"),
            select(func.count(func.distinct(req_table.c.experiment_reference)))
            .scalar_subquery()
            .label("number_of_experiments"),
        )

        try:
            with self.engine.begin() as connectable:
                row = connectable.execute(stmt).one()
                return SampleStoreStatistics(
                    number_of_entities=row.number_of_entities or 0,
                    number_of_results=row.number_of_results or 0,
                    number_of_experiments=row.number_of_experiments or 0,
                )
        except SQLAlchemyError as error:
            msg = f"Unable to get statistics for sample store {self._tablename}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

    def measurement_requests_for_operation(
        self,
        operation_id: str | set[str],
        filters: list[dict[str, str]] | None = None,
    ) -> list[MeasurementRequest] | dict[str, list[MeasurementRequest]]:
        """
        Fetch measurement requests for one or more operations, optionally filtered.

        Args:
            operation_id: The operation identifier, or a set of operation identifiers
            filters: Optional DB-level filters from prepare_query_filters_for_db()
                     Format: [{"$.path": "value"}, ...]
                     Supports filtering on request fields (status, requestIndex, etc.)
                     and nested JSON fields (metadata.*, experimentReference.*)

        Returns:
            A list of MeasurementRequest objects when passed a single operation ID,
            or a dict keyed by operation ID when passed a set of operation IDs
        """
        from sqlalchemy import and_, select

        from ado.core.samplestore.orm.measurements_filtering import (
            MeasurementFilterBuilder,
        )

        try:
            req_table = self._request_table
            reqres_table = self._request_result_table
            res_table = self._result_table

            stmt = (
                select(
                    req_table.c.uid,
                    req_table.c.experiment_reference,
                    req_table.c.operation_id,
                    req_table.c.request_index,
                    req_table.c.request_id,
                    req_table.c.type,
                    req_table.c.status,
                    req_table.c.metadata,
                    req_table.c.timestamp,
                    res_table.c.entity_id,
                    res_table.c.data,
                )
                .select_from(req_table)
                .join(reqres_table, reqres_table.c.request_uid == req_table.c.uid)
                .join(res_table, reqres_table.c.result_uid == res_table.c.uid)
            )

            if isinstance(operation_id, set):
                stmt = stmt.where(req_table.c.operation_id.in_(operation_id))
            else:
                stmt = stmt.where(req_table.c.operation_id == operation_id)

            if filters:
                filter_builder = MeasurementFilterBuilder(
                    dialect=self.engine.dialect.name
                )
                filter_conditions = filter_builder.apply_filters(
                    filters=filters,
                    table=req_table,
                    filter_type="request",
                )
                if filter_conditions:
                    stmt = stmt.where(and_(*filter_conditions))

            stmt = stmt.order_by(
                req_table.c.request_index,
                req_table.c.insert_id,
                reqres_table.c.entity_index,
                reqres_table.c.insert_id,
            )

            with self.engine.begin() as connectable:
                cur = connectable.execute(stmt)
        except SQLAlchemyError as error:
            msg = (
                f"Unable to get the measurement requests for operations {operation_id}"
                if isinstance(operation_id, set)
                else f"Unable to get the measurement requests for operation {operation_id}"
            )
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        requests = self._measurement_requests_cursor_to_pydantic(db_cursor=cur)
        if not isinstance(operation_id, set):
            return requests

        requests_by_operation_id: dict[str, list[MeasurementRequest]] = {
            requested_operation_id: [] for requested_operation_id in operation_id
        }
        for request in requests:
            requests_by_operation_id[request.operation_id].append(request)

        return requests_by_operation_id

    def measurement_request_by_id(
        self, measurement_request_id: str
    ) -> MeasurementRequest:

        try:
            with self.engine.begin() as connectable:
                query = sqlalchemy.text(f"""
                    SELECT req.uid, req.experiment_reference, req.operation_id,
                           req.request_index, req.request_id, req.type, req.status,
                           req.metadata, req.timestamp, res.entity_id, res.data
                    FROM (
                        SELECT *
                        FROM {self._tablename}_measurement_requests
                        WHERE request_id = :measurement_request_id
                    ) req
                    JOIN {self._tablename}_measurement_requests_results reqres ON reqres.request_uid = req.uid
                    JOIN {self._tablename}_measurement_results res ON reqres.result_uid = res.uid
                    ORDER BY req.request_index, req.insert_id , reqres.entity_index , reqres.insert_id
                    """).bindparams(  # noqa: S608 - self._tablename is not untrusted
                    measurement_request_id=measurement_request_id
                )
                cur = connectable.execute(query)
        except SQLAlchemyError as error:
            msg = f"Unable to get the measurement request for measurement request id {measurement_request_id}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        request = self._measurement_requests_cursor_to_pydantic(db_cursor=cur)
        return request[0] if request else None

    def measurement_results_for_operation(
        self,
        operation_id: str,
        filters: list[dict[str, str]] | None = None,
    ) -> list[MeasurementResult]:
        """
        Fetch measurement results for an operation, optionally filtered.

        Args:
            operation_id: The operation identifier
            filters: Optional DB-level filters from prepare_query_filters_for_db()
                     Format: [{"$.path": "value"}, ...]
                     Supports filtering on result fields (uid, entityIdentifier, etc.)
                     and nested JSON fields in the data column

        Returns:
            List of MeasurementResult objects matching the filters
        """
        from sqlalchemy import and_, select

        from ado.core.samplestore.orm.measurements_filtering import (
            MeasurementFilterBuilder,
        )

        try:
            # Use cached table metadata from initialization
            req_table = self._request_table
            reqres_table = self._request_result_table
            res_table = self._result_table

            # Build query using SQLAlchemy
            stmt = (
                select(res_table.c.data)
                .select_from(req_table)
                .join(reqres_table, reqres_table.c.request_uid == req_table.c.uid)
                .join(res_table, reqres_table.c.result_uid == res_table.c.uid)
                .where(req_table.c.operation_id == operation_id)
            )

            # Apply filters if provided
            if filters:
                filter_builder = MeasurementFilterBuilder(
                    dialect=self.engine.dialect.name
                )
                filter_conditions = filter_builder.apply_filters(
                    filters=filters,
                    table=res_table,
                    filter_type="result",
                )
                if filter_conditions:
                    stmt = stmt.where(and_(*filter_conditions))

            # Add ordering
            stmt = stmt.order_by(
                req_table.c.request_index,
                req_table.c.insert_id,
                reqres_table.c.entity_index,
                reqres_table.c.insert_id,
            )

            with self.engine.begin() as connectable:
                cur = connectable.execute(stmt)
        except SQLAlchemyError as error:
            msg = f"Unable to get the measurement results for operation {operation_id}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        parsed_results = []
        for row in cur:
            # Handle data - may already be parsed by SQLAlchemy
            row_data = row[0]
            row_dict = json.loads(row_data) if isinstance(row_data, str) else row_data
            if "reason" in row_dict:
                parsed_results.append(InvalidMeasurementResult.model_validate(row_dict))
            else:
                parsed_results.append(ValidMeasurementResult.model_validate(row_dict))

        return parsed_results

    def _measurement_requests_cursor_to_pydantic(
        self, db_cursor: sqlalchemy.CursorResult[typing.Any]
    ) -> list[MeasurementRequest]:

        # We consume the whole result cursor early to extract the set of entities
        # identifiers for which we have results and fetch missing ones in a batch
        # to reduce network overhead. The fallback entityWithIdentifier call should
        # only be reached for entities added by a concurrent distributed process
        # in the window between the batch fetch and now.
        rows = db_cursor.fetchall()

        # row[9] is entity id
        missing_entity_ids = {row[9] for row in rows if row[9] not in self._entities}
        if missing_entity_ids:
            fetched = self._fetch_entities(entity_ids=missing_entity_ids)
            self._entities.update(fetched)

        entries = {}
        measurement_results_for_entities = {}

        for entry in rows:
            (
                uid,
                experiment_reference,
                operation_id,
                request_index,
                request_id,
                request_type,
                request_status,
                metadata,
                timestamp,
                entity_id,
                result_data,
            ) = entry

            # Handle metadata - may already be parsed by SQLAlchemy
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            # Parse the result - we will always need it
            # May already be parsed by SQLAlchemy
            if isinstance(result_data, str):
                result_data = json.loads(result_data)
            if "reason" in result_data:
                result = InvalidMeasurementResult.model_validate(result_data)
            else:
                result = ValidMeasurementResult.model_validate(result_data)

            # For MeasurementRequests and related subclasses, we do not support
            # reassigning measurements. We must use a support structure to then
            # assign them just once.
            if uid in measurement_results_for_entities:
                measurement_results_for_entities[uid].append(result)
            else:
                measurement_results_for_entities[uid] = [result]

            # We also need the entity referenced by the measurement
            if entity_id not in self._entities:
                self._entities[entity_id] = self.entityWithIdentifier(entity_id)

            entity = self._entities[entity_id]

            # If we have already seen this measurement request
            # we are only interested in the entity associated to it
            if uid in entries:
                if not any(e.identifier == entity_id for e in entries[uid].entities):
                    entries[uid].entities.append(entity)

                continue

            #
            if request_type == ReplayedMeasurement.__name__:
                request = ReplayedMeasurement(
                    experimentReference=ExperimentReference.referenceFromString(
                        experiment_reference
                    ),
                    entities=[entity],
                    status=MeasurementRequestStateEnum(request_status),
                    requestid=request_id,
                    operation_id=operation_id,
                    requestIndex=request_index,
                    timestamp=timestamp,
                    metadata=metadata,
                )
            else:
                request = MeasurementRequest(
                    experimentReference=ExperimentReference.referenceFromString(
                        experiment_reference
                    ),
                    entities=[entity],
                    status=MeasurementRequestStateEnum(request_status),
                    requestid=request_id,
                    operation_id=operation_id,
                    requestIndex=request_index,
                    timestamp=timestamp,
                    metadata=metadata,
                )

            entries[uid] = request

        # We make sure we assign measurements just once
        for uid, results in measurement_results_for_entities.items():
            entries[uid].measurements = results

        return list(entries.values())

    def experiments_in_operation(self, operation_id: str) -> list[Experiment]:
        try:
            with self.engine.begin() as connectable:
                query = sqlalchemy.text(f"""
                    SELECT DISTINCT(experiment_reference)
                    FROM {self._tablename}_measurement_requests
                    WHERE operation_id = :operation_id
                    """).bindparams(  # noqa: S608 - self._tablename is not untrusted
                    operation_id=operation_id
                )
                cur = connectable.execute(query)
        except SQLAlchemyError as error:
            msg = f"Unable to get the experiments for operation {operation_id}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        return [
            self.experimentCatalog().experimentForReference(
                ExperimentReference.referenceFromString(e[0])
            )
            for e in cur
        ]

    def entity_identifiers_in_operations(
        self,
        operation_ids: str | set[str],
        group_by_operation: bool = False,
    ) -> set[str] | dict[str, set[str]]:
        """Get the set of entity identifiers sampled in one or more operations.

        Args:
            operation_ids: A single operation identifier or a set of operation
                identifiers to look up entity identifiers for.
            group_by_operation: When True, return a dict mapping each operation
                ID to its set of entity identifiers. When False (default),
                return a flat set of entity identifiers across all operations.

        Returns:
            A flat set of entity identifier strings when group_by_operation is
            False, or a dict mapping operation ID to set of entity identifiers
            when group_by_operation is True.
        """
        if isinstance(operation_ids, str):
            operation_ids = {operation_ids}

        try:
            from sqlalchemy import select

            req_table = self._request_table
            reqres_table = self._request_result_table
            res_table = self._result_table

            stmt = (
                select(
                    req_table.c.operation_id,
                    res_table.c.entity_id,
                )
                .select_from(req_table)
                .join(reqres_table, reqres_table.c.request_uid == req_table.c.uid)
                .join(res_table, reqres_table.c.result_uid == res_table.c.uid)
                .where(req_table.c.operation_id.in_(operation_ids))
                .distinct()
            )

            with self.engine.begin() as connectable:
                rows = connectable.execute(stmt).fetchall()
        except SQLAlchemyError as error:
            msg = f"Unable to get the entity ids for operations {operation_ids}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        entity_ids_by_operation_id: dict[str, set[str]] = {}
        for row in rows:
            entity_ids_by_operation_id.setdefault(row.operation_id, set()).add(
                row.entity_id
            )

        if group_by_operation:
            return entity_ids_by_operation_id
        return {
            entity_id
            for ids in entity_ids_by_operation_id.values()
            for entity_id in ids
        }

    def entity_identifiers_in_operation(
        self, operation_ids: str | set[str]
    ) -> set[str]:
        """Deprecated: use entity_identifiers_in_operations instead."""
        warnings.warn(
            "entity_identifiers_in_operation is deprecated, use entity_identifiers_in_operations instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.entity_identifiers_in_operations(operation_ids)

    def complete_measurement_request_with_results_timeseries(
        self,
        operation_id: str,
        output_format: typing.Literal["target", "observed"],
        limit_to_properties: list[str] | None = None,
        aggregation_method: PropertyAggregationMethodEnum | None = None,
        filters: list[dict[str, str]] | None = None,
    ) -> "pd.DataFrame":
        import pandas as pd

        """
        Returns the complete timeseries of measurement requests and measurement results.

        Parameters:
        - operation_id (str): The ID of the operation to retrieve measurement requests and results for.
        - output_format (typing.Literal["target", "observed"]): The format of the output data.
        - limit_to_properties (typing.Optional[list[str]]): A list of properties to limit the output to.
        - aggregation_method (PropertyAggregationMethodEnum | None): If set, aggregate list-valued
          property columns (e.g. mean of multiple runs) to a single scalar per cell.
        - filters (list[dict[str, str]] | None): Optional DB-level filters from prepare_query_filters_for_db()
          Can filter on both request and result fields

        Returns:
        pd.DataFrame: The timeseries of measurement requests and results for the operation.
        """
        measurement_requests = self.measurement_requests_for_operation(
            operation_id, filters=filters
        )
        rows = []

        for m in measurement_requests:
            rows.extend(m.series_representation(output_format=output_format))

        # We want to distinguish between values that weren't measured
        # and values that are meant to be NaN/None.
        # We do this by getting all the columns and reindexing each series
        # to have all the columns, filling their missing values with
        # `not_measured`, so that we can filter it out once we build the
        # full dataframe
        columns_in_rows = pd.DataFrame(rows).columns
        rows = [r.reindex(columns_in_rows, fill_value="not_measured") for r in rows]

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        columns_at_the_start = [
            "request_index",
            "result_index",
            "identifier",
            "experiment_id",
        ]
        columns_at_the_end = [
            "request_id",
            "entity_index",
            "valid",
        ]

        if limit_to_properties:
            for p in limit_to_properties:
                if p not in df.columns:
                    raise ValueError(
                        f"Property {p} is not in the timeseries. "
                        f"Available columns were: "
                        f"{set(df.columns).difference(set(columns_at_the_start + columns_at_the_end))}"
                    )

            df = filter_dataframe_columns(
                df,
                columns_to_keep=columns_at_the_start
                + columns_at_the_end
                + limit_to_properties,
            )

        if output_format == "observed":
            columns_at_the_start = ["request_index", "result_index", "identifier"]
            columns_at_the_end = ["valid"]
            df = df.drop(
                ["request_id", "entity_index", "experiment_id"], axis="columns"
            )

            def _aggregate_to_list_if_meaningful(series: pd.Series) -> pd.Series:
                filtered_series = list(
                    filter(lambda val: val != "not_measured", series)
                )

                if len(filtered_series) == 0:
                    return ""
                if len(filtered_series) == 1:
                    return filtered_series[0]
                return filtered_series

            df = df.groupby(
                by=["identifier", "valid", "request_index", "result_index"],
                as_index=False,
            ).agg(_aggregate_to_list_if_meaningful)

        df = reorder_dataframe_columns(
            df=df,
            move_to_start=columns_at_the_start,
            move_to_end=columns_at_the_end,
        )
        df = df.sort_values(by=["request_index", "result_index"])

        if aggregation_method is not None:
            property_columns = [
                c
                for c in df.columns
                if c not in columns_at_the_start and c not in columns_at_the_end
            ]
            pam = PropertyAggregationMethod(identifier=aggregation_method)

            def _aggregate_cell(value: object) -> object:
                if value == "not_measured":
                    return value
                if not isinstance(value, list):
                    return value
                try:
                    result, _ = pam.function(value)
                    return result
                except (ValueError, TypeError):
                    return value

            for col in property_columns:
                df[col] = df[col].apply(_aggregate_cell)

        return df.set_index("request_index")
