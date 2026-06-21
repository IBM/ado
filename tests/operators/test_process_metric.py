# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for the process_metric function in the RayTune operator."""

import math
from unittest.mock import MagicMock

import pytest
from ado_ray_tune.operator import process_metric

from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.measurementspace import (
    MeasurementSpace,
    MeasurementSpaceConfiguration,
)
from orchestrator.schema.property import ConstitutivePropertyDescriptor
from orchestrator.schema.property_value import ConstitutivePropertyValue


def _make_trainable_params(
    metric_format: str = "target", failed_value: float = float("nan")
) -> MagicMock:
    """Return a minimal mock of OrchTrainableParameters.

    Only orchestrator_config and measurement_space are used by process_metric;
    other fields of OrchTrainableParameters require live Ray actors so are left as mocks.
    Tests that exercise the virtual-property path should override
    params.measurement_space with a real MeasurementSpace instance.
    """
    params = MagicMock()
    params.orchestrator_config.metric_format = metric_format
    params.orchestrator_config.failed_metric_value = failed_value
    params.measurement_space.observedProperties = []
    return params


def _measurement_space(experiment: Experiment) -> MeasurementSpace:
    """Build a real MeasurementSpace from a single experiment."""
    return MeasurementSpace(
        configuration=MeasurementSpaceConfiguration(experiments=[experiment])
    )


class TestProcessMetricDirectHit:
    """Metrics present directly in all_results are returned without virtual property lookup."""

    def test_direct_metric_returns_last_value(self, entity: Entity) -> None:
        """When the metric is in all_results, the last entry is returned."""
        result = process_metric(
            metric="mip_gaps",
            all_results={"mip_gaps": [0.1, 0.2, 0.3]},
            entity=entity,
            trainable_params=_make_trainable_params(),
        )
        assert result == 0.3

    def test_direct_metric_single_value(self, entity: Entity) -> None:
        """Single-entry list returns that value."""
        result = process_metric(
            metric="mip_gaps",
            all_results={"mip_gaps": [0.05]},
            entity=entity,
            trainable_params=_make_trainable_params(),
        )
        assert result == 0.05


class TestProcessMetricVirtualTargetFormat:
    """Virtual property computed from allResults using measurement space observed properties."""

    def test_virtual_metric_uses_allresults_not_entity(
        self, entity: Entity, experiment: Experiment
    ) -> None:
        """Virtual property is looked up from measurement_space and computed from all_results.

        The experiment fixture exposes "pka" as a target property.  With metric_format="target",
        all_results is keyed by the target property identifier "pka".
        mean([0.1, 0.2]) == 0.15.
        """
        params = _make_trainable_params(metric_format="target")
        params.measurement_space = _measurement_space(experiment)

        result = process_metric(
            metric="pka-mean",
            all_results={"pka": [[0.1, 0.2]]},
            entity=entity,
            trainable_params=params,
        )

        assert result == pytest.approx(0.15)

    def test_virtual_metric_observed_format_uses_observed_key(
        self, entity: Entity, experiment: Experiment
    ) -> None:
        """With metric_format='observed', the observed property identifier is used as key.

        The observed property identifier for target "pka" is "{experiment_id}-pka".
        mean([0.3, 0.3]) == 0.3.
        """
        params = _make_trainable_params(metric_format="observed")
        params.measurement_space = _measurement_space(experiment)

        obs_key = f"{experiment.identifier}-pka"
        result = process_metric(
            metric=f"{obs_key}-mean",
            all_results={obs_key: [[0.3, 0.3]]},
            entity=entity,
            trainable_params=params,
        )

        assert result == pytest.approx(0.3)

    def test_virtual_metric_all_none_returns_failed_value(
        self, entity: Entity, experiment: Experiment
    ) -> None:
        """When all measurements are None, aggregate returns None and failed_metric_value is used."""
        params = _make_trainable_params(failed_value=float("nan"))
        params.measurement_space = _measurement_space(experiment)

        result = process_metric(
            metric="pka-mean",
            all_results={"pka": [None, None, None]},
            entity=entity,
            trainable_params=params,
        )

        assert math.isnan(result)

    def test_virtual_metric_base_not_in_allresults_returns_failed_value(
        self, entity: Entity, experiment: Experiment
    ) -> None:
        """When the base property key is absent from all_results, failed_metric_value is returned."""
        params = _make_trainable_params(failed_value=-1.0)
        params.measurement_space = _measurement_space(experiment)

        result = process_metric(
            metric="pka-mean",
            all_results={},
            entity=entity,
            trainable_params=params,
        )

        assert result == -1.0

    def test_virtual_metric_properties_none_returns_failed_value(
        self, entity: Entity, experiment: Experiment
    ) -> None:
        """When no property in measurement_space matches the virtual identifier, failed_metric_value is returned."""
        params = _make_trainable_params(failed_value=-1.0)
        params.measurement_space = _measurement_space(experiment)

        # "nonexistent_prop" is not a target or observed property in the experiment
        result = process_metric(
            metric="nonexistent_prop-mean",
            all_results={"nonexistent_prop": [[0.1]]},
            entity=entity,
            trainable_params=params,
        )

        assert result == -1.0

    def test_not_virtual_property_returns_failed_value(self, entity: Entity) -> None:
        """When the metric identifier is not a valid virtual property, failed_metric_value is returned."""
        result = process_metric(
            metric="unknown_metric",
            all_results={},
            entity=entity,
            trainable_params=_make_trainable_params(failed_value=-999.0),
        )

        assert result == -999.0

    def test_ambiguous_virtual_properties_raises(
        self,
        entity: Entity,
        mock_parameterizable_experiment: Experiment,
        customParameterization: list[ConstitutivePropertyValue],
    ) -> None:
        """Two parameterizations of the same experiment both share the target property
        identifier, producing two matching virtual properties and raising ValueError.

        mock_parameterizable_experiment has target "measurable_one".  Parameterizing it
        two ways creates two observed properties with different identifiers but the same
        target.  from_observed_properties_matching_identifier then returns both, which is
        the ambiguity the code guards against.

        customParameterization uses test_opt1="C", test_opt2=-1.
        The second parameterization uses test_opt1="A", test_opt2=-5 — both differ from
        the experiment's defaults (B / -3) so the validator accepts them.
        """
        pe1 = ParameterizedExperiment(
            parameterization=customParameterization[:-1],
            **mock_parameterizable_experiment.model_dump(),
        )
        second_parameterization = [
            ConstitutivePropertyValue(
                value="A",
                property=ConstitutivePropertyDescriptor(identifier="test_opt1"),
            ),
            ConstitutivePropertyValue(
                value=-5,
                property=ConstitutivePropertyDescriptor(identifier="test_opt2"),
            ),
        ]
        pe2 = ParameterizedExperiment(
            parameterization=second_parameterization,
            **mock_parameterizable_experiment.model_dump(),
        )
        ms = MeasurementSpace(
            configuration=MeasurementSpaceConfiguration(experiments=[pe1, pe2])
        )
        params = _make_trainable_params()
        params.measurement_space = ms

        with pytest.raises(ValueError, match="Ambiguous"):
            process_metric(
                metric="measurable_one-mean",
                all_results={"measurable_one": [[0.1]]},
                entity=entity,
                trainable_params=params,
            )

    def test_measurement_space_used_not_entity(self, experiment: Experiment) -> None:
        """Lookup uses measurement_space, not entity observed properties.

        A bare entity (no observed properties) would cause the old entity-based lookup
        to return None and fall back to failed_metric_value.  With the measurement_space-
        based lookup the virtual property is found correctly and the result is computed.
        """
        bare_entity = Entity(
            identifier="bare-entity",
            generatorid="test",
            constitutive_property_values=(),
        )
        params = _make_trainable_params(metric_format="target")
        params.measurement_space = _measurement_space(experiment)

        result = process_metric(
            metric="pka-mean",
            all_results={"pka": [0.5, 0.5]},
            entity=bare_entity,
            trainable_params=params,
        )

        # mean([0.5, 0.5]) == 0.5; would be nan with entity-based lookup (bare entity
        # has no observed properties)
        assert result == pytest.approx(0.5)
