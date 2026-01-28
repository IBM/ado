# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

import orchestrator.utilities.location
from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.modules.module import (
    ModuleConf,
    ModuleTypeEnum,
    load_module_class_or_function,
)

if typing.TYPE_CHECKING:
    from orchestrator.utilities.location import (
        SQLiteStoreConfiguration,
        SQLStoreConfiguration,
    )


class SampleStoreModuleConf(ModuleConf):
    moduleType: Annotated[ModuleTypeEnum, pydantic.Field()] = (
        ModuleTypeEnum.SAMPLE_STORE
    )


class SampleStoreSpecification(pydantic.BaseModel):
    """Model representing a SampleStore"""

    model_config = ConfigDict(extra="forbid")

    module: Annotated[
        SampleStoreModuleConf,
        pydantic.Field(description="The SampleStore module and class to use"),
    ]
    parameters: Annotated[
        dict,
        pydantic.Field(
            default_factory=dict,
            description="SampleStore specific parameters that configure its behaviour ",
        ),
    ]

    # Note: type is Any as using ResourceLocation causes the serialisation
    # to use ResourceLocation for some reason (pydantic 2.8)
    storageLocation: Annotated[
        typing.Any | None,
        pydantic.Field(
            description="Defines where the SampleStore is stored. Must be compatible with module and "
            "be and an instance of ResourceLocation or a subclass "
            "Optional: if not provided the user of the class can later add it",
        ),
    ] = None

    @pydantic.field_validator("storageLocation", mode="after")
    @classmethod
    def check_is_resource_location_subclass(
        cls,
        storageLocation: typing.Any | None,  # noqa: ANN401
    ) -> typing.Any | None:  # noqa: ANN401
        if storageLocation is not None and not isinstance(
            storageLocation, orchestrator.utilities.location.ResourceLocation
        ):
            raise ValueError(
                "The storageLocation field must be a ResourceLocation subclass"
            )

        return storageLocation

    @pydantic.field_validator("parameters", mode="after")
    @classmethod
    def check_parameters_valid_for_sample_store_module(
        cls, parameters: dict, context: pydantic.ValidationInfo
    ) -> dict:
        module = load_module_class_or_function(context.data["module"])
        validated_parameters = module.validate_parameters(parameters=parameters)
        # Convert Pydantic model back to dict for serialization
        if isinstance(validated_parameters, pydantic.BaseModel):
            return validated_parameters.model_dump()
        return validated_parameters

    @pydantic.field_validator("storageLocation", mode="before")
    @classmethod
    def set_correct_resource_location_class_for_sample_store_module(
        cls, storageLocation: dict, context: pydantic.ValidationInfo
    ) -> "SQLiteStoreConfiguration | SQLStoreConfiguration | None":
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

    identifier: Annotated[
        str | None,
        pydantic.Field(
            description="The identifier of the sample store. "
            "Required if this information is not specified in the storageLocation",
        ),
    ] = None


class SampleStoreConfiguration(pydantic.BaseModel):
    """Object for configuring creation of a SampleStore"""

    model_config = ConfigDict(extra="forbid")

    specification: Annotated[
        SampleStoreSpecification,
        pydantic.Field(description="The specification of the sample store"),
    ]
    copyFrom: Annotated[
        list[SampleStoreReference],
        pydantic.Field(
            default_factory=list,
            description="List of additional sample stores whose data is used to initialise the main sample store",
        ),
    ]
    metadata: Annotated[
        ConfigurationMetadata,
        pydantic.Field(
            description="Metadata about the configuration including optional name, description, "
            "labels for filtering, and any additional custom fields"
        ),
    ] = ConfigurationMetadata()

    @pydantic.field_validator("specification")
    def check_sample_store_specification_class_is_active(
        cls, value: SampleStoreSpecification
    ) -> SampleStoreSpecification:
        import orchestrator.core.samplestore.base

        moduleClass = orchestrator.modules.module.load_module_class_or_function(
            value.module
        )

        if not issubclass(
            moduleClass, orchestrator.core.samplestore.base.ActiveSampleStore
        ):
            raise ValueError(
                f"SampleStore module {moduleClass} is not an ActiveSampleStore"
            )

        return value
