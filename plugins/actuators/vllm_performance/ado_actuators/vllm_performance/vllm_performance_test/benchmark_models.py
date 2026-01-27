# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""
Pydantic models for benchmarks

This module defines shared data models for benchmark results and parameters that can be used
by both vLLM and GuideLLM benchmarks, ensuring consistent output format.
"""

from typing import TYPE_CHECKING, Annotated

import pydantic
from pydantic import AfterValidator

if TYPE_CHECKING:
    from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
    from orchestrator.schema.observed_property import ObservedPropertyValue


def none_if_negative(value: int | None) -> int | None:
    return value if value is not None and value > 0 else None


class BenchmarkParameters(pydantic.BaseModel):
    """Model for common benchmark parameters extracted from experiment values."""

    model: Annotated[str, pydantic.Field()]
    endpoint: Annotated[str | None, pydantic.Field()] = None
    request_rate: Annotated[
        int | None, pydantic.Field(), AfterValidator(none_if_negative)
    ] = None
    max_concurrency: Annotated[
        int | None, pydantic.Field(), AfterValidator(none_if_negative)
    ] = None
    num_prompts: Annotated[int, pydantic.Field(gt=0)] = 500
    number_input_tokens: Annotated[int | None, pydantic.Field()] = None
    max_output_tokens: Annotated[int | None, pydantic.Field()] = None
    burstiness: Annotated[float, pydantic.Field()] = 1.0
    dataset: Annotated[str | None, pydantic.Field()] = "random"


class BenchmarkResult(pydantic.BaseModel):
    """
    Standardized benchmark results format

    This model represents the output format used by both vLLM and GuideLLM benchmarks,
    ensuring consistency across different benchmark tools.
    """

    # Basic metrics
    duration: Annotated[float, pydantic.Field()] = 0.0
    completed: Annotated[int, pydantic.Field()] = 0
    total_input_tokens: Annotated[float, pydantic.Field()] = 0.0
    total_output_tokens: Annotated[float, pydantic.Field()] = 0.0

    # Throughput metrics
    request_throughput: Annotated[float, pydantic.Field()] = 0.0
    output_throughput: Annotated[float, pydantic.Field()] = 0.0
    total_token_throughput: Annotated[float, pydantic.Field()] = 0.0

    # Time to First Token (TTFT) metrics - in milliseconds
    mean_ttft_ms: Annotated[float, pydantic.Field()] = 0.0
    median_ttft_ms: Annotated[float, pydantic.Field()] = 0.0
    std_ttft_ms: Annotated[float, pydantic.Field()] = 0.0
    p25_ttft_ms: Annotated[float, pydantic.Field()] = 0.0
    p50_ttft_ms: Annotated[float, pydantic.Field()] = 0.0
    p75_ttft_ms: Annotated[float, pydantic.Field()] = 0.0
    p99_ttft_ms: Annotated[float, pydantic.Field()] = 0.0

    # Time Per Output Token (TPOT) metrics - in milliseconds
    mean_tpot_ms: Annotated[float, pydantic.Field()] = 0.0
    median_tpot_ms: Annotated[float, pydantic.Field()] = 0.0
    std_tpot_ms: Annotated[float, pydantic.Field()] = 0.0
    p25_tpot_ms: Annotated[float, pydantic.Field()] = 0.0
    p50_tpot_ms: Annotated[float, pydantic.Field()] = 0.0
    p75_tpot_ms: Annotated[float, pydantic.Field()] = 0.0
    p99_tpot_ms: Annotated[float, pydantic.Field()] = 0.0

    # Inter-Token Latency (ITL) metrics - in milliseconds
    mean_itl_ms: Annotated[float, pydantic.Field()] = 0.0
    median_itl_ms: Annotated[float, pydantic.Field()] = 0.0
    std_itl_ms: Annotated[float, pydantic.Field()] = 0.0
    p25_itl_ms: Annotated[float, pydantic.Field()] = 0.0
    p50_itl_ms: Annotated[float, pydantic.Field()] = 0.0
    p75_itl_ms: Annotated[float, pydantic.Field()] = 0.0
    p99_itl_ms: Annotated[float, pydantic.Field()] = 0.0

    # Request Latency (E2E) metrics - in milliseconds
    mean_e2el_ms: Annotated[float, pydantic.Field()] = 0.0
    median_e2el_ms: Annotated[float, pydantic.Field()] = 0.0
    std_e2el_ms: Annotated[float, pydantic.Field()] = 0.0
    p25_e2el_ms: Annotated[float, pydantic.Field()] = 0.0
    p50_e2el_ms: Annotated[float, pydantic.Field()] = 0.0
    p75_e2el_ms: Annotated[float, pydantic.Field()] = 0.0
    p99_e2el_ms: Annotated[float, pydantic.Field()] = 0.0

    def to_observed_property_values(
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
