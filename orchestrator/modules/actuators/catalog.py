# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import abc
import enum
import logging
from typing import Annotated

import pydantic

from orchestrator.schema.experiment import Experiment
from orchestrator.schema.reference import ExperimentReference


class ActuatorCatalogExtensionConf(pydantic.BaseModel):
    """Represents a dynamically loadable set of experiments for an actuator"""

    name: Annotated[
        str, pydantic.Field(description="The name of the catalog extension")
    ]
    location: Annotated[
        str, pydantic.Field(description="The location of the catalog extension")
    ]

    @property
    def catalogExtensionLocation(self) -> str:
        import os

        return os.path.join(self.location, self.name)


class ActuatorCatalogExtension(pydantic.BaseModel):
    """A list of experiments that can be added to a catalog

    TODO: This should be combined with ExperimentCatalog so they are the same class.
    Holding off on this as ExperimentCatalog has internal data-structures that need to change
    """

    experiments: Annotated[
        list[Experiment], pydantic.Field(description="A list of experiments")
    ]


class BaseCatalog(abc.ABC):

    @property
    @abc.abstractmethod
    def experiments(self) -> list[Experiment]:
        pass

    @property
    @abc.abstractmethod
    def experimentsMap(self) -> dict:
        pass

    @abc.abstractmethod
    def experimentForReference(self, reference: ExperimentReference) -> Experiment:
        pass


class ExperimentCatalog(BaseCatalog):
    """Base class for class that provide information on the available experiments"""

    def __init__(
        self, experiments: dict | None = None, catalogIdentifier: str = "UnnamedCatalog"
    ) -> None:
        """
        Parameters:
            experiments: A dictionary whose keys are experiment identifiers
                The values are orchestrator.model.data.Experiment instances

        :return: An ExperimentCatalog subclass
        """

        import os

        LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
        logging.basicConfig(level=LOGLEVEL)
        self.log = logging.getLogger("experiment-catalog")
        self._identifier = catalogIdentifier
        self._experiments = experiments if experiments is not None else {}

    def __str__(self) -> str:

        return f"Catalog {self._identifier} with {len(self._experiments)} experiments"

    @property
    def experiments(self) -> list[Experiment]:
        return list(self._experiments.values())

    @property
    def supported_experiments(self) -> list[Experiment]:
        return [e for e in self.experiments if not e.deprecated]

    @property
    def deprecated_experiments(self) -> list[Experiment]:
        return [e for e in self.experiments if e.deprecated]

    @property
    def identifier(self) -> str:

        return self._identifier

    @property
    def experimentsMap(self) -> dict[str, Experiment]:
        return {e.identifier: e for e in self.experiments}

    def experimentForReference(self, reference: ExperimentReference) -> Experiment:
        """Returns the experiment matching reference or None if there is no match

        Note: Here an experiment matches a reference if they have same actuator id and experiment id.
        In particular if the ExperimentReference is parameterized it does not affect the matching
        """

        experiments = self.experiments
        match = [
            e
            for e in experiments
            if e.reference.compareWithoutParameterization(reference)
        ]
        return None if len(match) == 0 else match[0]

    def addExperiment(self, experiment: Experiment) -> None:

        if self._experiments.get(experiment.identifier) is not None:
            self.log.warning(
                f"Experiment with identifier {experiment.identifier} already in receiver. Overwriting"
            )
            self.log.debug(
                f"New experiment {experiment.model_dump()}, existing experiment "
                f"{self._experiments.get(experiment.identifier).model_dump()}"
            )

        self._experiments[experiment.identifier] = experiment


class ExperimentNotInCatalogError(Exception):

    pass


class CatalogConfigurationRequirementEnum(enum.Enum):

    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    OPTIONAL = "optional"
