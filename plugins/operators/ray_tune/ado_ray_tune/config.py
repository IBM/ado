# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pydantic
import ray
from pydantic import ConfigDict

from ado_ray_tune.samplers import LhuSampler
from orchestrator.modules.module import (
    ModuleConf,
    ModuleTypeEnum,
    load_module_class_or_function,
)


class RayTuneOrchestratorConfiguration(pydantic.BaseModel):
    """Model for specific orchestrator options related to ray tune"""

    single_measurement_per_property: bool = pydantic.Field(
        default=True,
        description="Indicate that each property (experiment) "
        "should only be executed once",
    )
    failed_metric_value: float = pydantic.Field(
        default=float("nan"),
        description="Assign this value as the metric value for points if the measurements fail",
    )
    result_dump: str = pydantic.Field(
        default="none",
        deprecated=True,
        description="Location to store the result of ray.tune() (Not Used)",
    )

    model_config = ConfigDict(extra="forbid")


class OrchSearchAlgorithm(pydantic.BaseModel):
    name: str = pydantic.Field(description="The name of the search alg")
    params: dict = pydantic.Field(
        default={}, description="The params of the search alg"
    )

    @pydantic.model_validator(mode="after")
    def map_nevergrad_optimizer_name_to_type(self):

        if self.name != "nevergrad":
            return self

        # nevergrad wrapper requires passing the class of the optimizer in the "optimizer" param
        # here we have to switch from string to class
        # Note: The NevergradSearch interface types optimizer as optional, but it's not
        # We let Nevergrad handle this
        if optimizer := self.params.get("optimizer"):
            import nevergrad

            self.params["optimizer"] = nevergrad.optimizers.registry[optimizer]

        return self


class OrchStopperAlgorithm(pydantic.BaseModel):
    name: str = pydantic.Field(description="The name of the stopper")
    positionalParams: list = pydantic.Field(
        default=[], description="The positional params of the stopper"
    )
    keywordParams: dict = pydantic.Field(
        default={}, description="The keyword params of the stopper"
    )


class OrchTuneConfig(pydantic.BaseModel):
    """Model for options that will initialize a ray.tune.TuneConfig instance

    The aim of this class is to translate fields between the orchestrator RayTune agent's
    config and RayTune "TuneConfig"

    We need this as the values of certain TuneConfig fields e.g. search_alg, are complex objects which need to
    be instantiated, and we can't do this in our YAML config file.
    This class replaces the values of these fields with simple pydantic models that provide the information required
    for instantiation of the TuneConfig object.
    Then instances of this class can use this information to create the object and use it to init the actual TuneConfig

    Any keyword args passed to this classes init will become a field and be used to create the TuneConfig
    """

    # The following fields are required
    mode: str = pydantic.Field(default="min")
    metric: str = pydantic.Field(description="The metric to optimize")
    max_concurrent_trials: int = pydantic.Field(
        default=1,
        description="The maximum number of trials to have running at a time. Default 1",
    )

    # Here are the special fields that are used to create the inputs for TuneConfig
    search_alg: OrchSearchAlgorithm
    model_config = ConfigDict(extra="allow")

    def rayTuneConfig(self):
        # Get all values passed
        tune_options = self.model_dump()

        # To make the API of 'BasicVariantGenerator' compatible to all others
        if (
            hasattr(self, "max_concurrent_trials")
            and self.search_alg.name == "variant_generator"
        ):
            self.search_alg.params["max_concurrent"] = self.max_concurrent_trials
            del self.max_concurrent_trials  # suppress warning
        # For the special fields we need to deal with like search_alg do it and replace them
        # TODO: Do we need to pass "mode" and "metric" to the search algorithm?
        # Since we are also passing them to TuneConfig three lines below this.
        if self.search_alg.name == "lhu_sampler":
            search_alg = LhuSampler(
                mode=self.mode, metric=self.metric, **self.search_alg.params
            )
        else:
            search_alg = ray.tune.search.create_searcher(
                self.search_alg.name,
                mode=self.mode,
                metric=self.metric,
                **self.search_alg.params,
            )
        tune_options["search_alg"] = search_alg
        return ray.tune.TuneConfig(**tune_options)


class OrchRunConfig(pydantic.BaseModel):
    """Model for options that will initialize a ray.tune.RunConfig instance

    The aim of this class is to translate fields between the orchestrator RayTune agent's
    config and Ray Air "RunConfig"

    We need this as the values of certain RuneConfig fields e.g. stop, are complex objects which need to be
    instantiated, and we can't do this in our YAML config file.
    This class replaces the values of these fields with simple pydantic models that provide the information required
    for instantiation of the TuneConfig object.
    Then instances of this class can use this information to create the object and use it to init the actual TuneConfig

    Any keyword args passed to this classes init will become a field and be used to create the TuneConfig
    """

    # Here are the special fields that are used to create the inputs for RayConfig
    stop: list[OrchStopperAlgorithm] | None = pydantic.Field(
        default=None,
        description="A list of stopper(s) to use. If more than one will be combined with CombinedStopper",
    )
    model_config = ConfigDict(extra="allow")

    def rayRuntimeConfig(self) -> ray.tune.RunConfig:

        # Get all values passed
        run_options = self.model_dump()

        # Create the stoppers
        if self.stop is not None and len(self.stop) > 0:
            stoppers = []
            for stopperConf in self.stop:
                if stopperConf.name in [
                    "SimpleStopper",
                    "GrowthStopper",
                    "MaxSamplesStopper",
                    "InformationGainStopper",
                ]:
                    module_name = "ado_ray_tune.stoppers"
                else:
                    module_name = "ray.tune.stopper"

                module_conf = ModuleConf(
                    moduleType=ModuleTypeEnum.GENERIC,
                    moduleName=module_name,
                    moduleClass=stopperConf.name,
                )

                stopper_class = load_module_class_or_function(module_conf)

                if stopperConf.name in [
                    "SimpleStopper",
                    "GrowthStopper",
                    "MaxSamplesStopper",
                    "InformationGainStopper",
                ]:
                    # There is some problem passing the in-build stoppers params via init
                    stopper = stopper_class()
                    stopper.set_config(
                        *stopperConf.positionalParams, **stopperConf.keywordParams
                    )
                else:
                    stopper = stopper_class(
                        *stopperConf.positionalParams, **stopperConf.keywordParams
                    )

                stoppers.append(stopper)

            if len(stoppers) > 1:
                stopper = ray.tune.stopper.CombinedStopper(*stoppers)
            else:
                stopper = stoppers[0]

            run_options["stop"] = stopper

        return ray.tune.RunConfig(
            failure_config=ray.tune.FailureConfig(max_failures=0, fail_fast=True),
            **run_options,
        )


class RayTuneConfiguration(pydantic.BaseModel):
    """Model for options related to using ray tune"""

    tuneConfig: OrchTuneConfig = pydantic.Field(
        description="ray tune configuration options"
    )
    # This is a ray.tune.config.RunConfig object which is also pydantic model
    # However pydantic is throwing "pydantic.errors.ConfigError: field "callbacks"
    # not yet prepared so type is still a ForwardRef, you might need to call RunConfig.update_forward_refs()." error
    # When it is explicitly typed.
    # To get around this were are using Any and then converting any dicts to RunConfig in a validator
    runtimeConfig: OrchRunConfig | None = pydantic.Field(
        default=OrchRunConfig(), description="ray tune runtime options"
    )
    orchestratorConfig: RayTuneOrchestratorConfiguration = pydantic.Field(
        default=RayTuneOrchestratorConfiguration(), description="orchestrator options"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @pydantic.field_validator("runtimeConfig")
    def validate_runtime_config(cls, value):
        # Check we can create the runtime config
        _ = value.rayRuntimeConfig()

        return value

    @pydantic.field_validator("tuneConfig")
    def validate_tune_config(cls, value):
        # Check we can create the tune config
        _ = value.rayTuneConfig()

        return value
