# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""
Pydantic models for benchmark results

This module defines shared data models for benchmark results that can be used
by both vLLM and GuideLLM benchmarks, ensuring consistent output format.
"""

from typing import TYPE_CHECKING, Annotated, Any

import pydantic
from pydantic import model_validator

if TYPE_CHECKING:
    from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
    from orchestrator.schema.observed_property import ObservedPropertyValue


class BenchmarkParameters(pydantic.BaseModel):
    """Model for common benchmark parameters extracted from experiment values."""

    request_rate: Annotated[int | None, pydantic.Field()] = None
    max_concurrency: Annotated[int | None, pydantic.Field()] = None
    num_prompts: Annotated[int, pydantic.Field(gt=0)] = 500
    number_input_tokens: Annotated[int | None, pydantic.Field()] = None
    max_output_tokens: Annotated[int | None, pydantic.Field()] = None
    burstiness: Annotated[float, pydantic.Field()] = 1.0
    dataset: Annotated[str | None, pydantic.Field()] = "random"

    @model_validator(mode="before")
    @classmethod
    def validate_parameters(cls, data: Any) -> dict[str, Any]:
        """Validate and transform benchmark parameters."""
        if not isinstance(data, dict):
            return data

        # Convert request_rate: negative values become None (unlimited)
        if "request_rate" in data and data["request_rate"] is not None:
            rate = int(data["request_rate"])
            data["request_rate"] = None if rate < 0 else rate

        # Convert max_concurrency: negative values become None (unlimited)
        if "max_concurrency" in data and data["max_concurrency"] is not None:
            concurrency = int(data["max_concurrency"])
            data["max_concurrency"] = None if concurrency < 0 else concurrency

        # Convert num_prompts to int with default
        if "num_prompts" in data and data["num_prompts"] is not None:
            data["num_prompts"] = int(data["num_prompts"])

        # Convert token counts to int or None
        for field in ["number_input_tokens", "max_output_tokens"]:
            if field in data and data[field] is not None:
                data[field] = int(data[field])

        # Convert burstiness to float with default
        if "burstiness" in data and data["burstiness"] is not None:
            data["burstiness"] = float(data["burstiness"])

        # Ensure dataset has a value
        if "dataset" not in data or data["dataset"] is None:
            data["dataset"] = "random"

        return data


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
