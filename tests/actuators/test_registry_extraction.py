# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for extracting base classes from Ray-decorated actuator classes."""

import inspect

import pytest

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


def test_extract_base_class_from_decorated_actuator() -> None:
    """Test extraction from a real decorated actuator in the codebase.

    This test verifies the extraction logic works with real ActorClass instances
    from actuators decorated with @ray.remote.
    """
    # Import a real decorated actuator from the codebase
    # CustomExperiments is decorated with @ray.remote in the source
    from orchestrator.modules.actuators import custom_experiments
    from orchestrator.modules.actuators.registry import _extract_base_actuator_class

    # Get the actuator which is decorated
    decorated_actuator = custom_experiments.CustomExperiments

    # Check if it's an ActorClass instance
    try:
        import ray.actor

        is_actor_class = isinstance(decorated_actuator, ray.actor.ActorClass)
    except ImportError:
        # Ray not available, skip this test
        pytest.skip("Ray not available")

    # If it's decorated, test extraction
    if is_actor_class:
        extracted = _extract_base_actuator_class(decorated_actuator)

        # The extracted class should NOT be an ActorClass
        assert not isinstance(
            extracted, ray.actor.ActorClass
        ), f"Extracted class should not be ActorClass, got {type(extracted)}"

        # It should be a type (class)
        assert isinstance(extracted, type), f"Expected a class, got {type(extracted)}"

        # It should be a subclass of ActuatorBase
        assert issubclass(
            extracted, ActuatorBase
        ), "Extracted class should be subclass of ActuatorBase"

        # It should have the identifier attribute
        assert hasattr(extracted, "identifier")
        assert extracted.identifier == "custom_experiments"

        # It should have the catalog classmethod
        assert hasattr(extracted, "catalog")
        assert inspect.ismethod(extracted.catalog) or callable(extracted.catalog)

        # It should have default_parameters classmethod
        assert hasattr(extracted, "default_parameters")

        # Should be able to call default_parameters (it's a classmethod)
        params = extracted.default_parameters()
        assert params is not None
    else:
        # Not decorated in this environment, extraction should return it as-is
        extracted = _extract_base_actuator_class(decorated_actuator)
        assert extracted is decorated_actuator
