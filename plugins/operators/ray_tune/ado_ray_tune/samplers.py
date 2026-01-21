# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import copy
import logging

import numpy as np
import ray
import ray.tune
from ray.tune.search.variant_generator import parse_spec_vars

from orchestrator.schema.entityspace import EntitySpaceRepresentation


class LhuSampler(ray.tune.search.Searcher):
    def __init__(
        self,
        space: dict | list[dict] | None = None,
        metric: str | None = None,
        mode: str | None = None,
        points_to_evaluate: list[dict] | None = None,
        entity_space: EntitySpaceRepresentation | None = None,
    ) -> None:

        if mode and mode not in {"max", "min"}:
            raise ValueError(f"mode must be either max or min (was {mode})")

        super().__init__(
            metric=metric,
            mode=mode,
        )

        self.logger = logging.getLogger(__name__)
        self._entity_space = entity_space

        if isinstance(space, dict) and space:
            _resolved_vars, domain_vars, grid_vars = parse_spec_vars(space)
            if domain_vars or grid_vars:
                self.logger.warning(
                    ray.tune.search.UNRESOLVED_SEARCH_SPACE.format(
                        par="space", cls=type(self)
                    )
                )
                space = self.convert_search_space(space)

        self._space = space

        self._points_to_evaluate = copy.deepcopy(points_to_evaluate)

        # self._parameters = []
        # self._live_trial_mapping = {}
        self._sampler = None
        self._l_bounds = None
        self._u_bounds = None
        self._num_samples = 0
        self._samples_generated_so_far = 0
        self._suggestions = []

        if self._space:
            self._setup_experiment()

    def _setup_experiment(self) -> None:
        from .doe import LatinHypercubeSampler

        self._num_samples = 0
        self._samples_generated_so_far = 0
        self._sampler = LatinHypercubeSampler(self._space)
        self._samples_generated_so_far = 0
        self._suggestions = []
        self._generate_new_samples()

    def _generate_new_samples(self) -> None:
        self._suggestions.extend(
            self._sampler.generate_new_categorical_samples(n_factor=4)
        )
        self._samples_generated_so_far = len(self._suggestions)

    def set_entity_space(self, entity_space: EntitySpaceRepresentation) -> None:
        """LHC samplers requires the orchestrators ExplicitEntitySpaceRepresentation

        This is because the ray tune search space does not give sufficient information
        for the LHC sampler to work with dimensions that are not categorical"""

        self._entity_space = entity_space

    def set_search_properties(
        self,
        metric: str | None,
        mode: str | None,
        config: dict,
        **spec: dict,
    ) -> bool:
        space = self.convert_search_space(config)
        self._space = space
        if metric:
            self._metric = metric
        if mode:
            self._mode = mode

        self._setup_experiment()
        return True

    def suggest(self, trial_id: str) -> dict | None:
        if not self._space:
            raise RuntimeError(
                ray.tune.search.UNDEFINED_SEARCH_SPACE.format(
                    cls=self.__class__.__name__, space="space"
                )
            )

        if not self._metric or not self._mode:
            raise RuntimeError(
                ray.tune.search.UNDEFINED_METRIC_MODE.format(
                    cls=self.__class__.__name__, metric=self._metric, mode=self._mode
                )
            )

        if self._points_to_evaluate:
            config = self._points_to_evaluate.pop(0)
        else:
            if self._num_samples >= self._samples_generated_so_far:
                self._generate_new_samples()
            # config = self._suggestions.pop(0)
            config = self._suggestions[self._num_samples]
            self._num_samples += 1

        return config

    def convert_search_space(self, spec: dict) -> dict:

        import math

        import ray.tune.search.sample

        space = {}
        if not self._entity_space:
            for k, v in spec.items():
                if not isinstance(spec[k], ray.tune.search.sample.Categorical):
                    raise NotImplementedError(
                        "lhu_sampler cannot handle non-categorical search dimensions unless provided with ExplicitEntitySpace object"
                    )
                space[k] = v.categories
        else:
            for cp in self._entity_space.constitutiveProperties:
                if cp.propertyDomain.size != math.inf:
                    if values := cp.propertyDomain.values:
                        space[cp.identifier] = values
                    else:
                        space[cp.identifier] = np.arange(
                            start=min(cp.propertyDomain.domainRange),
                            stop=max(cp.propertyDomain.domainRange),
                            step=cp.propertyDomain.interval,
                        )
                else:
                    raise NotImplementedError(
                        "lhu_sampler can only be used with discrete bounded dimensions"
                    )

        return space

    def on_trial_complete(
        self, trial_id: str, result: dict | None = None, error: bool = False
    ) -> None:
        # we don't care??
        return None
