# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import typing
from typing import Annotated

import pydantic
import ray
from pydantic import ConfigDict

from orchestrator.modules.module import (
    ModuleConf,
    ModuleTypeEnum,
    load_module_class_or_function,
)

from .samplers import LhuSampler


def create_optuna_ray_tune_config(
    metric: str | list, mode: str | list, parameters: dict, tune_options: dict
) -> ray.tune.TuneConfig:
    try:
        from ray.tune.search.optuna import OptunaSearch
    except ImportError as error:
        raise ImportError(
            "Optuna must be installed! Run `pip install optuna`."
        ) from error

    # Params dict is already preprocessed (sampler class instantiated) by model validator
    search_alg = OptunaSearch(metric=metric, mode=mode, **parameters)
    tune_options["search_alg"] = search_alg
    tune_options["metric"] = metric[0] if isinstance(metric, list) else metric
    tune_options["mode"] = mode[0] if isinstance(mode, list) else mode
    return ray.tune.TuneConfig(**tune_options)


def create_lhu_ray_tune_config(
    metric: str,
    mode: str,
    parameters: dict,
    tune_options: dict,
) -> ray.tune.TuneConfig:
    search_alg = LhuSampler(mode=mode, metric=metric, **parameters)

    tune_options["search_alg"] = search_alg
    tune_options["metric"] = metric
    tune_options["mode"] = mode
    return ray.tune.TuneConfig(**tune_options)


def create_general_ray_tune_config(
    name: str,
    metric: str,
    mode: str,
    parameters: dict,
    tune_options: dict,
) -> ray.tune.TuneConfig:

    search_alg = ray.tune.search.create_searcher(
        name,
        mode=mode,
        metric=metric,
        **parameters,
    )

    tune_options["search_alg"] = search_alg
    tune_options["metric"] = metric
    tune_options["mode"] = mode
    return ray.tune.TuneConfig(**tune_options)


class RayTuneOrchestratorConfiguration(pydantic.BaseModel):
    """Model for specific orchestrator options related to ray tune"""

    metric_format: Annotated[
        typing.Literal["target", "observed"],
        pydantic.Field(
            description="Format for metric identifiers: 'target' (use target property identifiers) "
            "or 'observed' (use observed property identifiers)",
        ),
    ] = "target"
    single_measurement_per_property: Annotated[
        bool,
        pydantic.Field(
            description="Indicate that each property (experiment) "
            "should only be executed once",
        ),
    ] = True
    failed_metric_value: Annotated[
        float,
        pydantic.Field(
            description="Assign this value as the metric value for points if the measurements fail",
        ),
    ] = float("nan")
    result_dump: Annotated[
        str,
        pydantic.Field(
            deprecated=True,
            description="Location to store the result of ray.tune() (Not Used)",
        ),
    ] = "none"

    model_config = ConfigDict(extra="forbid")


class OrchSearchAlgorithm(pydantic.BaseModel):
    name: Annotated[str, pydantic.Field(description="The name of the search alg")]
    params: Annotated[
        dict,
        pydantic.Field(
            default_factory=dict, description="The params of the search alg"
        ),
    ]

    @pydantic.model_validator(mode="after")
    def map_optuna_sampler_name_to_instance(self) -> "OrchSearchAlgorithm":

        if self.name.lower() != "optuna":
            return self

        sampler_parameters = self.params.get("sampler_parameters")
        if not (optuna_sampler := self.params.get("sampler")):
            if sampler_parameters:
                raise ValueError(
                    "Optuna sampler parameters specified but no sampler specified"
                )
            return self

        try:
            import optuna.samplers

            sampler_cls = getattr(optuna.samplers, optuna_sampler)
        except (ImportError, AttributeError) as ex:
            raise ImportError(
                f"Optuna sampler '{optuna_sampler}' not found in optuna.samplers. Original error: {ex}"
            ) from ex
        # instantiate the sampler with any provided parameters
        sampler_instance = (
            sampler_cls(**sampler_parameters) if sampler_parameters else sampler_cls()
        )
        self.params["sampler"] = sampler_instance
        # delete sampler_parameters
        self.params.pop("sampler_parameters", None)

        return self

    @pydantic.model_validator(mode="after")
    def map_nevergrad_optimizer_name_to_type(self) -> "OrchSearchAlgorithm":

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
    name: Annotated[str, pydantic.Field(description="The name of the stopper")]
    positionalParams: Annotated[
        list,
        pydantic.Field(
            default_factory=list, description="The positional params of the stopper"
        ),
    ]
    keywordParams: Annotated[
        dict,
        pydantic.Field(
            default_factory=dict, description="The keyword params of the stopper"
        ),
    ]


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
    mode: Annotated[
        str | list[str], pydantic.Field(description="Mode(s) to use for optimization")
    ] = "min"
    metric: Annotated[
        str | list[str],
        pydantic.Field(description="Metric(s) to optimize (str or list[str])"),
    ]
    max_concurrent_trials: Annotated[
        int,
        pydantic.Field(
            description="The maximum number of trials to have running at a time. Default 1",
        ),
    ] = 1

    # Here are the special fields that are used to create the inputs for TuneConfig
    search_alg: Annotated[OrchSearchAlgorithm, pydantic.Field()]
    model_config = ConfigDict(extra="allow")

    def rayTuneConfig(self) -> ray.tune.TuneConfig:
        tune_options = self.model_dump()
        if self.search_alg.name.lower() == "optuna":
            return create_optuna_ray_tune_config(
                metric=self.metric,
                mode=self.mode,
                parameters=self.search_alg.params,
                tune_options=tune_options,
            )

        if isinstance(self.metric, list) or isinstance(self.mode, list):
            raise Exception(
                f"Multi-objective optimization with {self.search_alg.name} is not supported in ado_ray_tune."
            )
        if self.search_alg.name == "lhu_sampler":
            return create_lhu_ray_tune_config(
                mode=self.mode,
                metric=self.metric,
                tune_options=tune_options,
                parameters=self.search_alg.params,
            )
        return create_general_ray_tune_config(
            self.search_alg.name,
            mode=self.mode,
            metric=self.metric,
            tune_options=tune_options,
            parameters=self.search_alg.params,
        )


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
    stop: Annotated[
        list[OrchStopperAlgorithm] | None,
        pydantic.Field(
            description="A list of stopper(s) to use. If more than one will be combined with CombinedStopper",
        ),
    ] = None
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
                    "BayesianMetricDifferenceStopper",
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
                    "BayesianMetricDifferenceStopper",
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

    tuneConfig: Annotated[
        OrchTuneConfig, pydantic.Field(description="ray tune configuration options")
    ]
    # This is a ray.tune.config.RunConfig object which is also pydantic model
    # However pydantic is throwing "pydantic.errors.ConfigError: field "callbacks"
    # not yet prepared so type is still a ForwardRef, you might need to call RunConfig.update_forward_refs()." error
    # When it is explicitly typed.
    # To get around this were are using Any and then converting any dicts to RunConfig in a validator
    runtimeConfig: Annotated[
        OrchRunConfig | None, pydantic.Field(description="ray tune runtime options")
    ] = OrchRunConfig()
    orchestratorConfig: Annotated[
        RayTuneOrchestratorConfiguration,
        pydantic.Field(description="orchestrator options"),
    ] = RayTuneOrchestratorConfiguration()
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @pydantic.field_validator("runtimeConfig")
    def validate_runtime_config(cls, value: OrchRunConfig) -> OrchRunConfig:
        # Check we can create the runtime config
        _ = value.rayRuntimeConfig()

        return value

    @pydantic.field_validator("tuneConfig")
    def validate_tune_config(cls, value: OrchTuneConfig) -> OrchTuneConfig:
        # If BOTH metric/mode are lists, ensure they have the same length
        if (
            isinstance(value.metric, list)
            and isinstance(value.mode, list)
            and len(value.metric) != len(value.mode)
        ):
            raise ValueError(
                "If metric and mode are both lists, they must have the same length."
            )
        # Check we can create the tune config
        _ = value.rayTuneConfig()
        return value
