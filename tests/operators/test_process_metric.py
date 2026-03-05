# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for the process_metric function in the RayTune operator."""

import math
from unittest.mock import MagicMock

import pytest
from ado_ray_tune.operator import process_metric


def _make_trainable_params(
    metric_format: str = "target", failed_value: float = float("nan")
) -> MagicMock:
    """Return a minimal mock of OrchTrainableParameters."""
    params = MagicMock()
    params.orchestrator_config.metric_format = metric_format
    params.orchestrator_config.failed_metric_value = failed_value
    return params


def _make_entity(
    virtual_props: list | None = None, raises: Exception | None = None
) -> MagicMock:
    """Return a mock Entity whose virtualObservedPropertiesFromIdentifier returns virtual_props."""
    entity = MagicMock()
    entity.identifier = "test-entity"
    if raises is not None:
        entity.virtualObservedPropertiesFromIdentifier.side_effect = raises
    else:
        entity.virtualObservedPropertiesFromIdentifier.return_value = virtual_props
    return entity


def _make_virtual_prop(
    target_key: str = "mip_gaps",
    observed_key: str = "exp-mip_gaps",
    agg_value: float | None = 0.2,
) -> MagicMock:
    """Return a mock VirtualObservedProperty whose aggregate() returns agg_value."""
    vp = MagicMock()
    vp.baseObservedProperty.targetProperty.identifier = target_key
    vp.baseObservedProperty.identifier = observed_key
    vp.identifier = f"{target_key}-mean"
    agg_result = MagicMock()
    agg_result.value = agg_value
    vp.aggregate.return_value = agg_result
    return vp


class TestProcessMetricDirectHit:
    """Metrics present directly in all_results are returned without touching the entity."""

    def test_direct_metric_returns_last_value(self) -> None:
        """When the metric is in all_results, the last entry is returned."""
        result = process_metric(
            metric="mip_gaps",
            all_results={"mip_gaps": [0.1, 0.2, 0.3]},
            entity=_make_entity(),
            trainable_params=_make_trainable_params(),
        )
        assert result == 0.3

    def test_direct_metric_single_value(self) -> None:
        """Single-entry list returns that value."""
        result = process_metric(
            metric="mip_gaps",
            all_results={"mip_gaps": [0.05]},
            entity=_make_entity(),
            trainable_params=_make_trainable_params(),
        )
        assert result == 0.05


class TestProcessMetricVirtualTargetFormat:
    """Virtual property computed from allResults when metric_format='target'."""

    def test_virtual_metric_uses_allresults_not_entity(self) -> None:
        """Virtual property is computed from all_results, entity.valueForProperty is never called."""
        vp = _make_virtual_prop(target_key="mip_gaps", agg_value=0.15)
        entity = _make_entity(virtual_props=[vp])

        result = process_metric(
            metric="mip_gaps-mean",
            all_results={"mip_gaps": [[0.1, 0.2]]},
            entity=entity,
            trainable_params=_make_trainable_params(metric_format="target"),
        )

        assert result == 0.15
        # Confirms aggregate was called with the allResults values
        vp.aggregate.assert_called_once_with([[0.1, 0.2]])
        # Confirms entity.valueForProperty was NOT used
        entity.valueForProperty.assert_not_called()

    def test_virtual_metric_observed_format_uses_observed_key(self) -> None:
        """With metric_format='observed', the observed property identifier is used as key."""
        vp = _make_virtual_prop(
            target_key="mip_gaps", observed_key="exp-mip_gaps", agg_value=0.3
        )
        entity = _make_entity(virtual_props=[vp])

        result = process_metric(
            metric="mip_gaps-mean",
            all_results={"exp-mip_gaps": [[0.3, 0.3]]},
            entity=entity,
            trainable_params=_make_trainable_params(metric_format="observed"),
        )

        assert result == 0.3
        vp.aggregate.assert_called_once_with([[0.3, 0.3]])

    def test_virtual_metric_all_none_returns_failed_value(self) -> None:
        """When aggregate returns None value, failed_metric_value is returned (no crash)."""
        vp = _make_virtual_prop(target_key="mip_gaps", agg_value=None)
        entity = _make_entity(virtual_props=[vp])

        failed = float("nan")
        result = process_metric(
            metric="mip_gaps-mean",
            all_results={"mip_gaps": [[None, None, None]]},
            entity=entity,
            trainable_params=_make_trainable_params(failed_value=failed),
        )

        assert math.isnan(result)

    def test_virtual_metric_base_not_in_allresults_returns_failed_value(self) -> None:
        """When base property has no allResults entry, failed_metric_value is returned."""
        vp = _make_virtual_prop(target_key="mip_gaps")
        entity = _make_entity(virtual_props=[vp])

        result = process_metric(
            metric="mip_gaps-mean",
            all_results={},  # base property missing from results
            entity=entity,
            trainable_params=_make_trainable_params(failed_value=-1.0),
        )

        assert result == -1.0
        vp.aggregate.assert_not_called()

    def test_virtual_metric_properties_none_returns_failed_value(self) -> None:
        """When no observed property matches the virtual identifier, failed_metric_value is returned."""
        entity = _make_entity(virtual_props=None)

        result = process_metric(
            metric="mip_gaps-mean",
            all_results={"mip_gaps": [[0.1, 0.2]]},
            entity=entity,
            trainable_params=_make_trainable_params(failed_value=-1.0),
        )

        assert result == -1.0

    def test_not_virtual_property_returns_failed_value(self) -> None:
        """When the metric is not virtual, failed_metric_value is returned."""
        entity = _make_entity(raises=ValueError("not a virtual property"))

        result = process_metric(
            metric="unknown_metric",
            all_results={},
            entity=entity,
            trainable_params=_make_trainable_params(failed_value=-999.0),
        )

        assert result == -999.0

    def test_ambiguous_virtual_properties_raises(self) -> None:
        """Multiple matching virtual properties raises ValueError."""
        vp1 = _make_virtual_prop(target_key="mip_gaps")
        vp2 = _make_virtual_prop(target_key="mip_gaps_alt")
        entity = _make_entity(virtual_props=[vp1, vp2])

        with pytest.raises(ValueError, match="Ambiguous"):
            process_metric(
                metric="mip_gaps-mean",
                all_results={"mip_gaps": [[0.1]]},
                entity=entity,
                trainable_params=_make_trainable_params(),
            )
