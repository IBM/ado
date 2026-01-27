# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""
Pydantic models for benchmark results

This module defines shared data models for benchmark results that can be used
by both vLLM and GuideLLM benchmarks, ensuring consistent output format.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
    from orchestrator.schema.observed_property import ObservedPropertyValue


class BenchmarkResult(BaseModel):
    """
    Standardized benchmark results format

    This model represents the output format used by both vLLM and GuideLLM benchmarks,
    ensuring consistency across different benchmark tools.
    """

    # Basic metrics
    duration: float = 0.0
    completed: int = 0
    total_input_tokens: float = 0.0
    total_output_tokens: float = 0.0

    # Throughput metrics
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    total_token_throughput: float = 0.0

    # Time to First Token (TTFT) metrics - in milliseconds
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    std_ttft_ms: float = 0.0
    p25_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p75_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0

    # Time Per Output Token (TPOT) metrics - in milliseconds
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    std_tpot_ms: float = 0.0
    p25_tpot_ms: float = 0.0
    p50_tpot_ms: float = 0.0
    p75_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0

    # Inter-Token Latency (ITL) metrics - in milliseconds
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    std_itl_ms: float = 0.0
    p25_itl_ms: float = 0.0
    p50_itl_ms: float = 0.0
    p75_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0

    # Request Latency (E2E) metrics - in milliseconds
    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    std_e2el_ms: float = 0.0
    p25_e2el_ms: float = 0.0
    p50_e2el_ms: float = 0.0
    p75_e2el_ms: float = 0.0
    p99_e2el_ms: float = 0.0

    def to_measurements(
        self, experiment: "Experiment | ParameterizedExperiment"
    ) -> list["ObservedPropertyValue"]:
        """
        Convert BenchmarkResult to a list of ObservedPropertyValue instances.

        This method extracts the results for the experiment and returns them as PropertyValues.
        Only properties in the result that are listed by the experiment are returned.

        :param experiment: Experiment definition with observed properties
        :return: A list of ObservedPropertyValue instances
        """
        from orchestrator.schema.observed_property import ObservedPropertyValue
        from orchestrator.schema.property_value import ValueTypeEnum

        measured_values = []
        results_dict = self.model_dump()

        # Get observed properties
        observed = experiment.observedProperties
        for op in observed:
            # for every observed property
            target = op.targetProperty.identifier
            # get measured value
            value = results_dict.get(target)
            if value is None:
                # default non-measured property
                value = -1
            # Set the type
            if isinstance(value, str):
                value_type = ValueTypeEnum.STRING_VALUE_TYPE
            elif isinstance(value, bytes):
                value_type = ValueTypeEnum.BLOB_VALUE_TYPE
            elif isinstance(value, list):
                value_type = ValueTypeEnum.VECTOR_VALUE_TYPE
            else:
                value_type = ValueTypeEnum.NUMERIC_VALUE_TYPE
            # build property value
            property_value = ObservedPropertyValue(
                value=value,
                property=op,
                valueType=value_type,
            )
            measured_values.append(property_value)
        return measured_values
