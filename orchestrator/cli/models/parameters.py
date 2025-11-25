# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
from pathlib import Path

import pydantic

from orchestrator.cli.core.config import AdoConfiguration
from orchestrator.cli.models.types import (
    AdoCreateSupportedResourceTypes,
    AdoEditSupportedEditors,
    AdoGetSupportedOutputFormats,
    AdoGetSupportedResourceTypes,
    AdoShowEntitiesSupportedEntityTypes,
    AdoShowEntitiesSupportedOutputFormats,
    AdoShowEntitiesSupportedPropertyFormats,
    AdoShowRequestsSupportedOutputFormats,
    AdoShowResultsSupportedOutputFormats,
    AdoShowSummarySupportedOutputFormats,
)
from orchestrator.core import CoreResourceKinds
from orchestrator.core.operation.config import DiscoveryOperationEnum
from orchestrator.schema.virtual_property import PropertyAggregationMethodEnum


class AdoGetCommandParameters(pydantic.BaseModel):
    ado_configuration: AdoConfiguration
    exclude_default: bool
    exclude_fields: list[str] | None
    exclude_none: bool
    exclude_unset: bool
    field_selectors: list[dict[str, str]]
    from_sample_store: str | None
    from_operation: str | None
    from_space: str | None
    matching_point: pathlib.Path | None
    matching_space_id: str | None
    matching_space: pathlib.Path | None
    minimize_output: bool
    output_format: AdoGetSupportedOutputFormats
    resource_id: str | None
    resource_type: AdoGetSupportedResourceTypes
    show_deprecated: bool
    show_details: bool


class AdoCreateCommandParameters(pydantic.BaseModel):
    ado_configuration: AdoConfiguration
    dry_run: bool
    new_sample_store: bool
    override_values: list[dict[str, str | None]]
    resource_configuration_file: Path | None
    resource_type: AdoCreateSupportedResourceTypes
    use_default_sample_store: bool
    use_latest: list[CoreResourceKinds] | None
    with_resources: dict[CoreResourceKinds, pathlib.Path | str] | None


class AdoDeleteCommandParameters(pydantic.BaseModel):
    ado_configuration: AdoConfiguration
    delete_local_db: bool | None
    force: bool
    resource_id: str


class AdoDescribeCommandParameters(pydantic.BaseModel):
    actuator_id: str | None
    ado_configuration: AdoConfiguration
    resource_id: str | None
    resource_configuration: Path | None


class AdoEditCommandParameters(pydantic.BaseModel):
    ado_configuration: AdoConfiguration
    editor: AdoEditSupportedEditors
    resource_id: str


class AdoShowDetailsCommandParameters(pydantic.BaseModel):
    ado_configuration: AdoConfiguration
    resource_id: str


class AdoShowEntitiesCommandParameters(pydantic.BaseModel):
    ado_configuration: AdoConfiguration
    aggregation_method: PropertyAggregationMethodEnum | None
    entities_output_format: AdoShowEntitiesSupportedOutputFormats
    entities_property_format: AdoShowEntitiesSupportedPropertyFormats
    entities_type: AdoShowEntitiesSupportedEntityTypes
    properties: list[str] | None
    resource_configuration: Path | None
    resource_id: str | None


class AdoShowRelatedCommandParameters(pydantic.BaseModel):
    ado_configuration: AdoConfiguration
    resource_id: str


class AdoShowRequestsCommandParameters(pydantic.BaseModel):
    ado_configuration: AdoConfiguration
    hide_fields: list[str] | None
    output_format: AdoShowRequestsSupportedOutputFormats
    resource_id: str


class AdoShowResultsCommandParameters(pydantic.BaseModel):
    ado_configuration: AdoConfiguration
    hide_fields: list[str] | None
    output_format: AdoShowResultsSupportedOutputFormats
    resource_id: str


class AdoShowSummaryCommandParameters(pydantic.BaseModel):
    ado_configuration: AdoConfiguration
    columns_to_hide: list[str] | None
    include_properties: list[str] | None
    output_format: AdoShowSummarySupportedOutputFormats
    query: list[dict[str, str | None]] | None
    render_output: bool
    resource_ids: list[str]


class AdoTemplateCommandParameters(pydantic.BaseModel):
    actuator_identifier: str | None
    ado_configuration: AdoConfiguration
    from_experiments: list[dict[str, str | None]] | None
    include_schema: bool
    operator_name: str | None
    operator_type: DiscoveryOperationEnum | None
    output_path: Path | None
    parameters_only_schema: bool
    template_local_context: bool


class AdoUpgradeCommandParameters(pydantic.BaseModel):
    ado_configuration: AdoConfiguration
