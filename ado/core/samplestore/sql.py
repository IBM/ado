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
from ado.core.discoveryspace.stats import DiscoverySpaceStatistics
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
        import pandas as pd

        if reference.identifier is not None:
            if reference.storageLocation is None:
                raise ValueError(
                    "SQLSampleStore.experimentCatalog requires valid location parameters. "
                )

            query = f"""SELECT * FROM sqlsource_{reference.identifier} LIMIT 1;"""  # noqa: S608 - reference.identifier is not untrusted
            engine = engine_for_sql_store(configuration=reference.storageLocation)

            with engine.connect() as connectable:
                table = pd.read_sql(query, con=connectable)

            j = table.representation[0]

            d = json.loads(j)
            entity = Entity.model_validate(d)
            refs = [
                e
                for e in entity.experimentReferences
                if e.actuatorIdentifier == "replay"
            ]
            experiments = {}
            for r in refs:
                props = [
                    p for p in entity.observedProperties if p.experimentReference == r
                ]
                experiment = Experiment(
                    identifier=r.experimentIdentifier,
                    actuatorIdentifier=r.actuatorIdentifier,
                    targetProperties=[p.targetProperty for p in props],
                )
                experiments[experiment.identifier] = experiment

            catalog = ExperimentCatalog(
                experiments=experiments, catalogIdentifier="sqlstore_catalog"
            )
        else:
            raise ValueError(
                f"No identifier provided for SQLSampleStore - cannot read catalog. Data passed: {reference}"
            )

        return catalog

    def experimentCatalog(
        self,
    ) -> ExperimentCatalog | None:

        # TODO: This is not the right way to do this.
        # Here we're using the descriptors of the first entity to create the catalog
        # if this entity has an experiment with "replay" actuators
        # This works in the case every entity in sampletore was imported from an external source
        # and all had the same external experiment.
        # A better way would be to find all results from a replay experiment and then
        # get the set of those

        # Optimized: Query just one entity directly instead of loading all entities
        query = sqlalchemy.text(f"""
            SELECT ent.identifier, ent.representation, res.data
            FROM {self._tablename} ent
            LEFT OUTER JOIN {self._tablename}_measurement_results res ON res.entity_id = ent.identifier
            LIMIT 1
        """).bindparams()  # noqa: S608 - self._tablename is not untrusted

        try:
            with self.engine.begin() as connectable:
                cur = connectable.execute(query)
                row = cur.fetchone()
        except SQLAlchemyError as error:
            msg = f"Unable to fetch first entity for catalog from sample store {self._tablename}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        if row is None:
            # There are no entities
            return None

        entity_identifier, entity_representation, result_data = row

        try:
            entity = Entity.model_validate(json.loads(entity_representation))
        except Exception as error:
            self.log.warning(
                f"Unable to decode representation for entity {entity_identifier} when building catalog.\n"
                f"Representation was: {entity_representation}.\n"
                f"Error was {error}"
            )
            return None

        # If there's a measurement result, add it to the entity
        if result_data is not None:
            try:
                result_dict = json.loads(result_data)
                if result_dict.get("measurements", None):
                    measurement_result = ValidMeasurementResult.model_validate(
                        result_dict
                    )
                    entity.add_measurement_result(result=measurement_result)
            except Exception as error:
                self.log.debug(
                    f"Unable to decode measurement result for entity {entity_identifier} when building catalog: {error}"
                )
                # Continue without the measurement result - catalog doesn't strictly need it

        refs = [
            e for e in entity.experimentReferences if e.actuatorIdentifier == "replay"
        ]
        experiments = {}
        for r in refs:
            props = [p for p in entity.observedProperties if p.experimentReference == r]
            experiment = Experiment(
                identifier=r.experimentIdentifier,
                actuatorIdentifier=r.actuatorIdentifier,
                targetProperties=[p.targetProperty for p in props],
                requiredProperties=tuple(
                    [
                        ConstitutiveProperty.from_descriptor(p)
                        for p in entity.constitutiveProperties
                    ]
                ),
            )
            experiments[experiment.identifier] = experiment

        return ExperimentCatalog(
            experiments=experiments, catalogIdentifier="sqlstore_catalog"
        )

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
        if not self._all_entities_loaded:
            # Initial load: delegate to refresh with force_fetch_all_entities=True
            self.log.debug(f"Initial load of entities for {self._tablename}")
            self.refresh(force_fetch_all_entities=True)

        return list(self._entities.values())

    def _fetch_entities(self, entity_ids: set[str] | None = None) -> dict[str, Entity]:
        """
        Fetch entities from the database.

        Parameters:
            entity_ids: Optional set of entity identifiers to fetch.
                       If None or empty set, fetches all entities.

        Returns:
            Dictionary mapping entity_identifier -> Entity object

        Raises:
            SystemError: If database query fails
            FailedToDecodeStoredEntityError: If entity JSON is invalid
        """
        # Treat empty set same as None - fetch all entities
        if entity_ids is not None and len(entity_ids) == 0:
            entity_ids = None

        # Build query based on whether we're filtering
        if entity_ids is None:
            query = sqlalchemy.text(
                f"SELECT identifier, representation FROM {self._tablename}"  # noqa: S608 - self._tablename is not untrusted
            )
            params = {}
        else:
            # Use parameterized query for filtering
            placeholders = ", ".join([f":id{i}" for i in range(len(entity_ids))])
            query = sqlalchemy.text(
                f"SELECT identifier, representation FROM {self._tablename} "  # noqa: S608 - self._tablename is not untrusted
                f"WHERE identifier IN ({placeholders})"
            )
            params = {f"id{i}": eid for i, eid in enumerate(entity_ids)}

        try:
            with self.engine.begin() as connectable:
                cur = connectable.execute(query, params)
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
        return entities

    def _fetch_measurement_results(
        self, min_insert_id: int = 0
    ) -> tuple[dict[str, list[ValidMeasurementResult]], int]:
        """
        Fetch measurement results from database starting from a specific insert_id.

        This method fetches results, validates them, and groups them by entity_id.
        Only valid measurement results are included in the returned dictionary.
        All validation happens here, so callers don't need to validate again.

        Parameters:
            min_insert_id: Minimum insert_id to fetch (exclusive).
                          Use 0 to fetch all results.

        Returns:
            Tuple of:
            - Dictionary mapping entity_id -> list of ValidMeasurementResult objects
            - Maximum insert_id seen (or min_insert_id if no results)

        Raises:
            SystemError: If database query fails
            FailedToDecodeStoredMeasurementResultForEntityError: If result JSON is invalid
        """
        from collections import defaultdict

        query = sqlalchemy.text(f"""
            SELECT insert_id, entity_id, data
            FROM {self._tablename}_measurement_results
            WHERE insert_id > :min_insert_id
            ORDER BY insert_id
            """)  # noqa: S608 - self._tablename is not untrusted

        try:
            with self.engine.begin() as connectable:
                cur = connectable.execute(query, {"min_insert_id": min_insert_id})
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
                result_dict = json.loads(result_data)
                if not result_dict.get("measurements", None):
                    continue

                measurement_result = ValidMeasurementResult.model_validate(result_dict)
                results_by_entity[entity_id].append(measurement_result)
            except Exception as error:
                raise FailedToDecodeStoredMeasurementResultForEntityError(
                    entity_identifier=entity_id,
                    result_representation=result_data,
                    cause=error,
                ) from error

        total_results = sum(len(results) for results in results_by_entity.values())
        self.log.debug(
            f"Fetched {total_results} measurement results for {len(results_by_entity)} entities "
            f"(insert_id range: {min_insert_id + 1} to {max_insert_id})"
        )

        return dict(results_by_entity), max_insert_id

    def refresh(self, force_fetch_all_entities: bool = False) -> tuple[int, int]:
        """
        Refresh entities and fetch new measurement results.

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
            >>> initial_count = len(store.entities)
            >>> # Another process adds measurements
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
            # Initial load: fetch all entities
            self._entities = self._fetch_entities(entity_ids=None)
            new_entities_count = len(self._entities)
            self._all_entities_loaded = True
            self.log.debug(f"Fetched all {new_entities_count} entities")

        # Phase 2: Fetch new measurement results (already validated and grouped)
        results_by_entity, max_insert_id = self._fetch_measurement_results(
            min_insert_id=self._last_insert_id
        )

        if not results_by_entity:
            self.log.debug("No new measurement results found")
            return (new_entities_count, 0)

        # Phase 3: Fetch missing entities
        # Doing it every time even if force_fetch_all_entities is True to avoid
        # the off-chance where another process adds an entity and some results
        # in the time it takes to fetch all the entities + all the measurements.
        # This avoid the chance of having results for which we have no entity.
        new_entity_ids = set(results_by_entity.keys())
        missing_entity_ids = new_entity_ids - set(self._entities.keys())

        if missing_entity_ids:
            self.log.debug(f"Fetching {len(missing_entity_ids)} new entities")
            new_entities = self._fetch_entities(entity_ids=missing_entity_ids)
            self._entities.update(new_entities)
            new_entities_count = len(new_entities)

            if len(missing_entity_ids) != new_entities_count:
                self.log.warning(
                    f"Expected to find {len(missing_entity_ids)} new entities but "
                    f"{new_entities_count} were retrieved. This suggests another process "
                    f"is updating the sample store concurrently."
                )

        # Phase 4: Attach measurements to entities (no validation needed - already done)
        total_measurements = 0
        for entity_id, measurement_results in results_by_entity.items():
            for measurement_result in measurement_results:
                # We have fetched results starting from self._last_insert_id, which
                # means:
                #   1.  Somebody else (e.g., another distributed process) could have
                #       added results to the sample store.
                #   2.  We ourselves could've added results to the sample store via
                #       add_measurement_results.
                # At the moment we can't know the `insert_id` of the results we add
                # to avoid them. If we did, we would still have to fetch results
                # starting from self._last_insert_id because someone else could have
                # added results, but we would also be able to add a NOT IN to avoid
                # ones we are already aware of.
                # As it stands, then, we need to be careful not to add measurement
                # results twice.
                try:
                    self._entities[entity_id].add_measurement_result(
                        result=measurement_result
                    )
                except DuplicateMeasurementResultError:  # noqa: PERF203
                    pass
                else:
                    total_measurements += 1

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

    def upsertExperimentResults(
        self,
        entities: list[Entity],
        experiment: Experiment,
    ) -> None:

        self.upsertEntities(entities, [experiment])

    def upsertEntities(
        self,
        entities: list[Entity],
        experiments: list[Experiment] | None = None,
    ) -> None:
        """Raises:
        SystemError: If there are any errors encountered with upserting entities to SQL DB
        """

        # Local
        for entity in entities:
            storedEntity = self._entities.get(entity.identifier)  # type: Entity
            if storedEntity is not None:
                # Merge the entities property values measured here and upsert the result
                if experiments is not None and len(experiments) != 0:
                    for experiment in experiments:
                        values = entity.propertyValuesFromExperiment(experiment)
                        for v in values:
                            storedEntity.add_measurement_result(
                                ValidMeasurementResult(
                                    entityIdentifier=storedEntity.identifier,
                                    measurements=[v],
                                )
                            )
                else:
                    # if no experiments are specified we add everything.
                    values = entity.propertyValues
                    for v in values:
                        if storedEntity.valueForProperty(v.property) is None:
                            storedEntity.add_measurement_result(
                                ValidMeasurementResult(
                                    entityIdentifier=storedEntity.identifier,
                                    measurements=[v],
                                )
                            )
            else:
                self._entities[entity.identifier] = entity

        # Retrieve stored version of all the entities

        for index in range(0, len(entities), 5000):
            # Replace entities passed with the stored equivalent as that was the one that's updated
            selectedEntities = [
                self._entities[entity.identifier]
                for entity in entities[index : index + 5000]
            ]

            values = [
                {
                    "identifier": e.identifier,
                    "representation": e.model_dump_json(
                        exclude_defaults=True, exclude_unset=True
                    ),
                }
                for e in selectedEntities
            ]

            self.log.debug(f"Inserting {len(values)} entities")

            try:
                # Remote
                with self.engine.begin() as connectable:
                    query = ado.metastore.sql.statements.upsert_entities(
                        sample_store_name=self._tablename,
                        dialect=self.engine.dialect.name,
                    )
                    connectable.execute(query, values)
            except SQLAlchemyError as error:
                self.log.critical(
                    f"Failed to upsert entity batch starting from {index}. Error: {error}"
                )
                raise SystemError(
                    f"Failed to upsert entity batch starting from {index}. Error: {error}"
                ) from error

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

    def space_entity_statistics(
        self,
        space_ids_to_operation_ids: dict[str, set[str]],
    ) -> "dict[str, DiscoverySpaceStatistics]":
        """Compute entity-level statistics for one or more discovery spaces.

        Issues a single SQL query that fetches all distinct
        ``(operation_id, entity_id)`` pairs across every operation referenced
        in *space_ids_to_operation_ids*, then groups the results by space ID
        in Python.  This approach is portable across all supported backends
        (SQLite, MySQL).

        ``number_matching_entities`` and
        ``number_matching_entities_with_measurements`` are not computed here
        (they require Python-side ``isEntityInSpace`` evaluation) and are
        always ``None`` in the returned models.

        Args:
            space_ids_to_operation_ids: Mapping of space ID to the set of
                operation IDs that belong to that space.  Spaces with an empty
                operation-ID set are returned with ``number_measured_entities``
                equal to ``0``.  An empty mapping returns an empty dict.

        Returns:
            A ``dict`` keyed by space ID.  Each value is a
            :class:`~ado.core.discoveryspace.stats.DiscoverySpaceStatistics`
            with ``number_measured_entities`` populated and all other fields at
            their defaults (``None``).

        Raises:
            SystemError: If the underlying SQL query fails.
        """
        if not space_ids_to_operation_ids:
            return {}

        # Separate spaces that have no operations (return 0 immediately) from
        # those that need a DB query.
        empty_space_ids = {
            space_id
            for space_id, operation_ids in space_ids_to_operation_ids.items()
            if not operation_ids
        }
        spaces_to_query = {
            space_id: operation_ids
            for space_id, operation_ids in space_ids_to_operation_ids.items()
            if operation_ids
        }

        result: dict[str, DiscoverySpaceStatistics] = {
            space_id: DiscoverySpaceStatistics(
                number_of_experiments=0,
                number_of_operations=0,
                number_of_explore_operations=0,
                number_measured_entities=0,
            )
            for space_id in empty_space_ids
        }

        if not spaces_to_query:
            return result

        # Flat set of all operation IDs across all queried spaces.
        operation_ids = {
            operation_id
            for operation_ids in spaces_to_query.values()
            for operation_id in operation_ids
        }
        # Reverse map: operation_id → space_id (each operation belongs to one space).
        operation_id_to_space_id: dict[str, str] = {
            operation_id: space_id
            for space_id, operation_ids in spaces_to_query.items()
            for operation_id in operation_ids
        }

        try:
            from sqlalchemy import select

            req_table = self._request_table
            reqres_table = self._request_result_table
            res_table = self._result_table

            # Fetch all distinct (operation_id, entity_id) pairs in one query.
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

            # Group distinct entity IDs per space in Python.
            entity_ids_by_space_id: dict[str, set[str]] = {
                space_id: set() for space_id in spaces_to_query
            }
            for row in rows:
                space_id = operation_id_to_space_id[row.operation_id]
                entity_ids_by_space_id[space_id].add(row.entity_id)

            for space_id, entity_ids in entity_ids_by_space_id.items():
                result[space_id] = DiscoverySpaceStatistics(
                    number_of_experiments=0,
                    number_of_operations=0,
                    number_of_explore_operations=0,
                    number_measured_entities=len(entity_ids),
                )

        except SQLAlchemyError as error:
            msg = f"Unable to get entity statistics for spaces {set(spaces_to_query)}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        return result

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
