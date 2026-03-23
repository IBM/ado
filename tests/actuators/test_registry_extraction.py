# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for extracting base classes from actuator classes."""

import inspect

from orchestrator.modules.actuators.base import ActuatorBase
from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import ExperimentReference


def test_extract_base_class_from_undecorated_actuator() -> None:
    """Test that extraction works with undecorated classes (returns as-is)."""
    from orchestrator.modules.actuators.registry import _extract_base_actuator_class

    class TestActuator(ActuatorBase):  # noqa: ANN001, ANN202, ANN206
        identifier = "test_undecorated"

        def submit(
            self,
            entities: list[Entity],
            experimentReference: ExperimentReference,
            requesterid: str,
            requestIndex: int,
        ) -> list[str]:
            return []

        @classmethod
        def catalog(cls, actuator_configuration=None):  # noqa: ANN001, ANN206
            from orchestrator.modules.actuators.catalog import ExperimentCatalog

            return ExperimentCatalog(identifier=cls.identifier, experiments=[])

    # Extract from undecorated class (should return the same class)
    extracted_class = _extract_base_actuator_class(TestActuator)

    # Should return the same class
    assert extracted_class is TestActuator
    assert issubclass(extracted_class, ActuatorBase)
    assert extracted_class.identifier == "test_undecorated"


def test_extract_base_class_from_undecorated_actuator_in_codebase() -> None:
    """Test extraction from a real undecorated actuator in the codebase.

    Actuators are no longer decorated with @ray.remote; operators/setup.py
    applies the decorator when instantiating them. Extraction returns the
    class as-is for undecorated actuators.
    """
    from orchestrator.modules.actuators import custom_experiments
    from orchestrator.modules.actuators.registry import _extract_base_actuator_class

    actuator_class = custom_experiments.CustomExperiments

    # Extraction should return the same class for undecorated actuators
    extracted = _extract_base_actuator_class(actuator_class)
    assert extracted is actuator_class
    assert issubclass(extracted, ActuatorBase)
    assert extracted.identifier == "custom_experiments"
    assert hasattr(extracted, "catalog")
    assert inspect.ismethod(extracted.catalog) or callable(extracted.catalog)
    assert hasattr(extracted, "default_parameters")
    params = extracted.default_parameters()
    assert params is not None
