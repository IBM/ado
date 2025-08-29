# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing

import pydantic
from pydantic import ConfigDict

import orchestrator.utilities.location
from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.modules.module import (
    ModuleConf,
    ModuleTypeEnum,
    load_module_class_or_function,
)


class SampleStoreModuleConf(ModuleConf):
    moduleType: ModuleTypeEnum = pydantic.Field(default=ModuleTypeEnum.SAMPLE_STORE)

    @pydantic.field_validator("moduleType", mode="before")
    @classmethod
    def convert_module_type_entity_source_to_sample_store(cls, value: str):
        from orchestrator.core.resources import (
            CoreResourceKinds,
            warn_deprecated_resource_model_in_use,
        )

        old_value = "entity_source"
        new_value = "sample_store"
        if value == old_value:
            warn_deprecated_resource_model_in_use(
                affected_resource=CoreResourceKinds.SAMPLESTORE,
                deprecated_from_ado_version="v0.9.6",
                removed_from_ado_version="v1.0.0",
            )
            value = new_value

        return value

    @pydantic.field_validator("moduleClass", mode="before")
    @classmethod
    def convert_module_class_entity_source_to_sample_store(cls, value: str):
        from orchestrator.core.resources import (
            CoreResourceKinds,
            warn_deprecated_resource_model_in_use,
        )

        value_mappings = {
            "CSVEntitySource": "CSVSampleStore",
            "SQLEntitySource": "SQLSampleStore",
        }
        if value in value_mappings:
            warn_deprecated_resource_model_in_use(
                affected_resource=CoreResourceKinds.SAMPLESTORE,
                deprecated_from_ado_version="v0.9.6",
                removed_from_ado_version="v1.0.0",
            )
            value = value_mappings[value]

        return value

    @pydantic.field_validator("moduleName", mode="before")
    @classmethod
    def convert_module_name_entity_source_to_sample_store(cls, value: str):
        from orchestrator.core.resources import (
            CoreResourceKinds,
            warn_deprecated_resource_model_in_use,
        )

        path_mappings = {
            "orchestrator.core.entitysource": "orchestrator.core.samplestore",
            "orchestrator.plugins.entitysources": "orchestrator.plugins.samplestores",
        }
        for old_path in path_mappings.keys():
            if old_path in value:
                warn_deprecated_resource_model_in_use(
                    affected_resource=CoreResourceKinds.SAMPLESTORE,
                    deprecated_from_ado_version="v0.9.6",
                    removed_from_ado_version="v1.0.0",
                )
                return value.replace(old_path, path_mappings[old_path])

        return value


class SampleStoreSpecification(pydantic.BaseModel):
    """Model representing a SampleStore"""

    model_config = ConfigDict(extra="forbid")

    module: SampleStoreModuleConf = pydantic.Field(
        description="The SampleStore module and class to use",
    )
    parameters: typing.Any = pydantic.Field(
        default={},
        description="SampleStore specific parameters that configure its behaviour ",
    )

    # Note: type is Any as using ResourceLocation causes the serialisation
    # to use ResourceLocation for some reason (pydantic 2.8)
    storageLocation: typing.Optional[typing.Any] = pydantic.Field(
        default=None,
        description="Defines where the SampleStore is stored. Must be compatible with module and "
        "be and an instance of ResourceLocation or a subclass "
        "Optional: if not provided the user of the class can later add it",
    )

    @pydantic.field_validator("storageLocation", mode="after")
    @classmethod
    def check_is_resource_location_subclass(cls, storageLocation: typing.Any):

        if storageLocation is not None:
            assert isinstance(
                storageLocation, orchestrator.utilities.location.ResourceLocation
            )

        return storageLocation

    @pydantic.field_validator("parameters", mode="after")
    @classmethod
    def check_parameters_valid_for_sample_store_module(
        cls, parameters: typing.Dict, context: pydantic.ValidationInfo
    ):
        module = load_module_class_or_function(context.data["module"])
        return module.validate_parameters(parameters=parameters)

    @pydantic.field_validator("storageLocation", mode="before")
    @classmethod
    def set_correct_resource_location_class_for_sample_store_module(
        cls, storageLocation: typing.Dict, context: pydantic.ValidationInfo
    ):
        # Only do this if storageLocation is not None
        # Note: The default is None, in which case if storageLocation is not explicitly give
        # this method  is not called
        # However if None is passed explicitly, which would happen on a load of a module which had the "none" default
        # this method will be called
        if storageLocation is not None:
            sample_store_class = load_module_class_or_function(context.data["module"])
            storageLocationClass = sample_store_class.storage_location_class()
            # 24/04/2025 AP:
            # We use a pydantic.RootModel to support storageLocationClass being
            # a Union of multiple classes. Pydantic will validate them and the
            # root element will be the pydantic.BaseModel that passed the validation.
            rm = pydantic.RootModel[storageLocationClass]
            return rm.model_validate(storageLocation).root
        return storageLocation


class SampleStoreReference(SampleStoreSpecification):

    model_config = ConfigDict(extra="forbid")

    identifier: typing.Optional[str] = pydantic.Field(
        default=None,
        description="The identifier of the sample store. "
        "Required if this information is not specified in the storageLocation",
    )


class SampleStoreConfiguration(pydantic.BaseModel):
    """Object for configuring creation of a SampleStore"""

    model_config = ConfigDict(extra="forbid")

    specification: SampleStoreSpecification = pydantic.Field(
        description="The specification of the sample store",
    )
    copyFrom: typing.List[SampleStoreReference] = pydantic.Field(
        default=[],
        description="List of additional sample stores whose data is used to initialise the main sample store",
    )
    metadata: ConfigurationMetadata = pydantic.Field(
        default=ConfigurationMetadata(),
        description="User defined metadata about the configuration. A set of keys and values. "
        "Two optional keys that are used by convention are name and description",
    )

    @pydantic.field_validator("specification")
    def check_sample_store_specification_class_is_active(cls, value):

        import orchestrator.core.samplestore.base

        moduleClass = orchestrator.modules.module.load_module_class_or_function(
            value.module
        )
        assert issubclass(
            moduleClass, orchestrator.core.samplestore.base.ActiveSampleStore
        ), f"SampleStore module {moduleClass} is not active"

        return value
