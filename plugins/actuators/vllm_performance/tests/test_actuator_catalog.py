# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for the packaged vLLM experiment catalog."""

from ado_actuators.vllm_performance.actuator import VLLMPerformanceTest

from ado.schema.experiment import Experiment


def test_catalog_preserves_utf8_descriptions() -> None:
    """Load the packaged catalog and retain Unicode property descriptions."""
    catalog = VLLMPerformanceTest.catalog()
    identifiers = {experiment.identifier for experiment in catalog.experiments}
    assert {"vllm-bench-deployment", "vllm-bench-endpoint"} <= identifiers

    descriptions = [
        prop.metadata.get("description", "")
        for experiment in catalog.experiments
        for prop in experiment.optionalProperties
    ]
    assert any("\u201cauto\u201d" in description for description in descriptions)

    for experiment in catalog.experiments:
        assert Experiment.model_validate(experiment.model_dump()) == experiment
