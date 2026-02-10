# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import contextlib
import json
import logging
import typing
import uuid
from typing import TYPE_CHECKING, Annotated, Literal

import pydantic
import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError

import orchestrator.core.samplestore.config
import orchestrator.core.samplestore.csv
import orchestrator.metastore.sql.statements
from orchestrator.core.samplestore.base import (
    ActiveSampleStore,
    FailedToDecodeStoredEntityError,
    FailedToDecodeStoredMeasurementResultForEntityError,
)
from orchestrator.metastore.sql.utils import engine_for_sql_store
from orchestrator.modules.actuators.catalog import ExperimentCatalog
from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.property import (
    ConstitutiveProperty,
)
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import (
    MeasurementRequest,
    MeasurementRequestStateEnum,
    ReplayedMeasurement,
)
from orchestrator.schema.result import (
    DuplicateMeasurementResultError,
    InvalidMeasurementResult,
    MeasurementResult,
    MeasurementResultStateEnum,
    ValidMeasurementResult,
)
from orchestrator.utilities.location import (
    SQLiteStoreConfiguration,
    SQLStoreConfiguration,
)
from orchestrator.utilities.pandas import (
    filter_dataframe_columns,
    reorder_dataframe_columns,
)

if TYPE_CHECKING:
    import pandas as pd
    from rich.console import RenderableType


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

        csv_sample_store = orchestrator.core.samplestore.csv.CSVSampleStore.from_csv(
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

        from orchestrator.utilities.rich import get_rich_repr

        return Group(
            Text.assemble(("Identifier: ", "bold"), (self.uri, "bold green")),
            Text("Number of entities:", style="bold", end=" "),
            get_rich_repr(self.numberOfEntities),
        )

    def commit(self) -> None:
        pass

    @classmethod
    def experimentCatalogFromReference(
        cls, reference: orchestrator.core.samplestore.config.SampleStoreReference
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

    def _create_source_table(self) -> None:

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

    def __init__(
        self,
        identifier: str | None,
        storageLocation: (
            orchestrator.utilities.location.SQLStoreConfiguration
            | SQLiteStoreConfiguration
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

        # Create a table for this sample store
        self._create_source_table()

        # Initialize entities cache as empty dict for lazy loading
        # Empty dict is falsy, so lazy loading check `if not self._entities:` still works
        # But it's also a valid dict that can be used for assignments
        self._entities = {}
        self._last_insert_id = (
            0  # Track last processed insert_id for incremental refresh
        )

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
    def location(self) -> orchestrator.utilities.location.SQLStoreConfiguration:

        return self._configuration.model_copy()

    @property
    def entities(self) -> list[Entity]:
        if not self._entities:
            # Initial load: delegate to refresh with force_fetch_all_entities=True
            self.log.debug(f"Initial load of entities for {self._tablename}")
            self._entities = {}
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

        # Check cache first - if all requested entities are cached, return them
        if self._entities:
            cached_entities = [
                self._entities[entity_id]
                for entity_id in entity_ids_set
                if entity_id in self._entities
            ]
            # If we got all requested entities from cache, return them
            if len(cached_entities) == len(entity_ids_set):
                return cached_entities

        # Query database for entities by identifiers
        # Use SQLAlchemy's expanding bindparam for IN clause
        # This automatically handles the parameter expansion for the IN clause
        query = sqlalchemy.text(f"""
            SELECT ent.identifier, ent.representation, res.data
            FROM {self._tablename} ent
            LEFT OUTER JOIN {self._tablename}_measurement_results res ON res.entity_id = ent.identifier
            WHERE ent.identifier IN :entity_ids
        """).bindparams(  # noqa: S608 - self._tablename is not untrusted
            sqlalchemy.bindparam(
                key="entity_ids", value=list(entity_ids_set), expanding=True
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

        return list(entities_dict.values())

    def entities_in_operation(self, operation_id: str) -> list[Entity]:
        """Get entities directly from a single operation in one query.

        This method is optimized for the common case of fetching entities from
        a single operation. It performs the entire operation in a single database
        query, avoiding the need to first fetch entity IDs and then fetch entities.

        Args:
            operation_id: The operation identifier to fetch entities for

        Returns:
            List of Entity objects that were sampled in the specified operation
        """
        query = sqlalchemy.text(f"""
            SELECT
                ent.identifier,
                ent.representation,
                res.data
            FROM {self._tablename} ent
            JOIN {self._tablename}_measurement_results res ON res.entity_id = ent.identifier
            JOIN {self._tablename}_measurement_requests_results reqres ON reqres.result_uid = res.uid
            JOIN {self._tablename}_measurement_requests req ON reqres.request_uid = req.uid
            WHERE req.operation_id = :operation_id
        """).bindparams(  # noqa: S608 - self._tablename is not untrusted
            operation_id=operation_id
        )

        try:
            with self.engine.begin() as connectable:
                cur = connectable.execute(query)
        except SQLAlchemyError as error:
            msg = f"Unable to fetch entities for operation {operation_id} from sample store {self._tablename}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        entities_dict: dict[str, Entity] = {}
        for entity_identifier, entity_representation, result_data in cur:

            if entity_identifier in self._entities:
                entities_dict[entity_identifier] = self._entities[entity_identifier]
                continue

            try:
                entity = Entity.model_validate(json.loads(entity_representation))
                self._entities[entity_identifier] = entity
                entities_dict[entity_identifier] = entity
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
                    self.log.debug(
                        f"Entity {entity_identifier} had no valid measurements associated to it."
                    )
                    continue

                measurement_result = ValidMeasurementResult.model_validate(result_dict)
            except Exception as error:
                raise FailedToDecodeStoredMeasurementResultForEntityError(
                    entity_identifier=entity_identifier,
                    result_representation=result_data,
                    cause=error,
                ) from error

            with contextlib.suppress(DuplicateMeasurementResultError):
                entities_dict[entity_identifier].add_measurement_result(
                    result=measurement_result
                )

        return list(entities_dict.values())

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
                    query = orchestrator.metastore.sql.statements.insert_entities_ignore_on_duplicate(
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
        measurementRequest: orchestrator.schema.request.MeasurementRequest,
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
                    query = orchestrator.metastore.sql.statements.upsert_entities(
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
    def storage_location_class() -> (
        type[SQLiteStoreConfiguration | SQLStoreConfiguration]
    ):
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

    def measurement_requests_for_operation(
        self, operation_id: str
    ) -> list[MeasurementRequest]:

        try:
            with self.engine.begin() as connectable:
                query = sqlalchemy.text(f"""
                    SELECT req.uid, req.experiment_reference, req.operation_id,
                           req.request_index, req.request_id, req.type, req.status,
                           req.metadata, req.timestamp, res.entity_id, res.data
                    FROM (
                        SELECT *
                        FROM {self._tablename}_measurement_requests
                        WHERE operation_id = :operation_id
                    ) req
                    JOIN {self._tablename}_measurement_requests_results reqres ON reqres.request_uid = req.uid
                    JOIN {self._tablename}_measurement_results res ON reqres.result_uid = res.uid
                    ORDER BY req.request_index, req.insert_id , reqres.entity_index , reqres.insert_id
                    """).bindparams(  # noqa: S608 - self._tablename is not untrusted
                    operation_id=operation_id
                )
                cur = connectable.execute(query)
        except SQLAlchemyError as error:
            msg = f"Unable to get the measurement results for operation {operation_id}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        return self._measurement_requests_cursor_to_pydantic(db_cursor=cur)

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
        self, operation_id: str
    ) -> list[MeasurementResult]:
        try:
            with self.engine.begin() as connectable:
                query = sqlalchemy.text(f"""
                    SELECT res.data
                    FROM (
                        SELECT *
                        FROM {self._tablename}_measurement_requests
                        WHERE operation_id = :operation_id
                    ) req
                    JOIN {self._tablename}_measurement_requests_results reqres ON reqres.request_uid = req.uid
                    JOIN {self._tablename}_measurement_results res ON reqres.result_uid = res.uid
                    ORDER BY req.request_index, req.insert_id , reqres.entity_index , reqres.insert_id
                    """).bindparams(  # noqa: S608 - self._tablename is not untrusted
                    operation_id=operation_id
                )
                cur = connectable.execute(query)
        except SQLAlchemyError as error:
            msg = f"Unable to get the measurement results for operation {operation_id}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        parsed_results = []
        for row in cur:
            row_dict = json.loads(row[0])
            if "reason" in row_dict:
                parsed_results.append(InvalidMeasurementResult.model_validate(row_dict))
            else:
                parsed_results.append(ValidMeasurementResult.model_validate(row_dict))

        return parsed_results

    def _measurement_requests_cursor_to_pydantic(
        self, db_cursor: sqlalchemy.CursorResult[typing.Any]
    ) -> list[MeasurementRequest]:

        entries = {}
        measurement_results_for_entities = {}

        for entry in db_cursor:
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

            metadata = json.loads(metadata)

            # Parse the result - we will always need it
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

    def entity_identifiers_in_operation(self, operation_id: str) -> set[str]:
        try:
            with self.engine.begin() as connectable:
                query = sqlalchemy.text(f"""
                    SELECT DISTINCT(res.entity_id)
                    FROM (
                        SELECT *
                        FROM {self._tablename}_measurement_requests
                        WHERE operation_id = :operation_id
                    ) req
                    JOIN {self._tablename}_measurement_requests_results reqres ON reqres.request_uid = req.uid
                    JOIN {self._tablename}_measurement_results res ON reqres.result_uid = res.uid
                    """).bindparams(  # noqa: S608 - self._tablename is not untrusted
                    operation_id=operation_id
                )
                cur = connectable.execute(query)
        except SQLAlchemyError as error:
            msg = f"Unable to get the entity ids for operation {operation_id}"
            self.log.critical(f"{msg}. Error: {error}")
            raise SystemError(f"{msg}. Error: {error}") from error

        return {ident[0] for ident in cur}

    def complete_measurement_request_with_results_timeseries(
        self,
        operation_id: str,
        output_format: typing.Literal["target", "observed"],
        limit_to_properties: list[str] | None = None,
    ) -> "pd.DataFrame":
        import pandas as pd

        """
        Returns the complete timeseries of measurement requests and measurement results.

        Parameters:
        - operation_id (str): The ID of the operation to retrieve measurement requests and results for.
        - output_format (typing.Literal["target", "observed"]): The format of the output data.
        - limit_to_properties (typing.Optional[list[str]]): A list of properties to limit the output to.

        Returns:
        pd.DataFrame: The timeseries of measurement requests and results for the operation.
        """
        measurement_requests = self.measurement_requests_for_operation(operation_id)
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
        return df.set_index("request_index")
