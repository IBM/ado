# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import time
import typing
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

import anyio
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from ado.core.discoveryspace.samplers import BaseSampler
from ado.utilities.logging import configure_logging
from trim.missing_target import record_unmeasured_entity
from trim.samplers.no_priors_utils import (
    get_index_list_van_der_corput,
    get_list_of_entities_from_df_and_space,
    get_source_and_target,
)
from trim.trim_pydantic import (
    MissingTargetMeasurementMode,
    TrimSamplerParametersInternal,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ado.core.discoveryspace.space import DiscoverySpace, Entity
    from ado.modules.operators.discovery_space_manager import (
        DiscoverySpaceManager,
    )

from ado.utilities.pandas import sort_rows_by_column_names
from trim.utils.exceptions import InsufficientDataError
from trim.utils.logging_utils import (
    log_after_first_holdout_creation,
    log_after_split_common_and_diff,
    log_before_first_holdout_update,
    save_source_train_holdout_dfs,
    training_guardrail,
)
from trim.utils.miscellaneous import delete_dir
from trim.utils.order import get_feature_importance_order
from trim.utils.rowsring import RowsRing
from trim.utils.split_common_and_diff import (
    split_common_and_diff,
)
from trim.utils.stopping_criterion import stopping_bool_from_ratios


# NOTE: to repeat the operation on the same space you can delete the operation
# but first make sure that the output of this operation is not used by another operation
class TrimSampleSelector(BaseSampler):
    @classmethod
    def samplerCompatibleWithDiscoverySpaceRemote(
        cls,
        remoteDiscoverySpace: DiscoverySpaceManager,  # type: ignore[name-defined]
    ) -> bool:
        # do you want to return False if no point has been measured?
        return True

    def _setup_debug_directory_sync(self) -> None:
        """Synchronously setup debug directory if debug logging is enabled."""
        if self.log.isEnabledFor(logging.DEBUG):
            debug_dir = pathlib.Path(self.params.debugDirectory).expanduser().resolve()
            self.log.debug(
                f"Creating a folder to save intermediate files:\n{debug_dir}\n\n"
            )
            debug_dir.mkdir(parents=True, exist_ok=True)

    async def _setup_debug_directory_async(self) -> None:
        """Asynchronously setup debug directory if debug logging is enabled."""
        if self.log.isEnabledFor(logging.DEBUG):
            debug_dir = await anyio.Path(self.params.debugDirectory).expanduser()
            debug_dir = await debug_dir.resolve()
            self.log.debug(
                f"Creating a folder to save intermediate files:\n{debug_dir}\n\n"
            )
            await debug_dir.mkdir(parents=True, exist_ok=True)

    def handle_unmeasured_targetOutputs_from_no_priors(
        self,
        discoverySpace: DiscoverySpace,
        train_target_cols: list[str],
    ) -> int:
        """Scan the no-priors operation and handle entities that did not produce the targetOutput.

        Counts every result that either failed (InvalidMeasurementResult) or succeeded
        without measuring the targetOutput. When mode is InjectDefaultValue, also
        populates self.injected_defaults with one default row per such entity.
        No-op when mode is Error.

        Args:
            discoverySpace: The discovery space being characterised.
            train_target_cols: Column names (constitutive properties + targetOutput)
                used to initialise the injected_defaults DataFrame.

        Returns:
            The number of entities that did not measure their targetOutput.
        """
        from ado.schema.result import InvalidMeasurementResult, ValidMeasurementResult

        self.injected_defaults = pd.DataFrame(columns=train_target_cols)

        if (
            self.params.missingTargetMeasurements.mode
            == MissingTargetMeasurementMode.Error
        ):
            return 0

        exp_refs = set(discoverySpace.measurementSpace.experimentReferences)
        prior_results = discoverySpace.measurement_results_for_operation(
            self.params.noPriorsOperationId
        )
        seen_entity_ids: set[str] = set()
        total_unmeasured = 0
        for result in prior_results:
            if isinstance(result, InvalidMeasurementResult):
                if result.experimentReference not in exp_refs:
                    continue
            elif isinstance(result, ValidMeasurementResult):
                # Only inject when the targetOutput was not measured
                measured_targets = {
                    m.property.targetProperty.identifier for m in result.measurements
                }
                if self.params.targetOutput in measured_targets:
                    continue
            else:
                continue

            total_unmeasured += 1
            if (
                self.params.missingTargetMeasurements.mode
                != MissingTargetMeasurementMode.InjectDefaultValue
            ):
                continue

            entity_id = result.entityIdentifier
            if entity_id in seen_entity_ids:
                continue
            seen_entity_ids.add(entity_id)
            entities = discoverySpace.sample_store.get_entities(
                identifiers=entity_id, require_measurements=False
            )
            if not entities:
                self.log.warning(
                    f"No entity found in store for identifier {entity_id!r}; skipping default injection."
                )
                continue
            entity = entities[0]
            row = {
                cpv.property.identifier: cpv.value
                for cpv in entity.constitutive_property_values
            }
            row[self.params.targetOutput] = (
                self.params.missingTargetMeasurements.defaultValue
            )
            self.injected_defaults = pd.concat(
                [self.injected_defaults, pd.DataFrame([row])],
                ignore_index=True,
            )
            self.log.info(
                f"Pre-scan: injecting default for entity {entity_id!r} "
                f"from no-priors operation {self.params.noPriorsOperationId!r}."
            )
        self.log.info(
            f"Pre-scan complete: discovered {total_unmeasured} entities that did not measure "
            f"the targetOutput from no-priors operation {self.params.noPriorsOperationId!r}."
        )

        return total_unmeasured

    def _core_iterator_logic(
        self,
        discoverySpace: DiscoverySpace,
        list_of_entities: list[Entity],
    ) -> typing.Generator[list[Entity], None, None]:
        """
        Core iterator logic shared between sync and async implementations.
        This is a synchronous generator that yields entities based on the TRIM algorithm.
        Each iteration yields exactly one entity wrapped in a list.
        This is because ado calls this in an async way and we want to know that the
        entity we've yielded in the previous iteration is now measured.
        We cannot yield multiple entities because we might not know which of the
        entities got measured unless we do more checks.
        """
        expected_paths = {
            "autoGluonArgs": self.params.outputDirectory,
            "finalModelAutoGluonArgs": (self.params.outputDirectory or "")
            + "_finalized",
        }
        for arg_name, args_obj in (
            ("autoGluonArgs", self.params.autoGluonArgs),
            ("finalModelAutoGluonArgs", self.params.finalModelAutoGluonArgs),
        ):
            actual = args_obj.tabularPredictorArgs.get("path")
            expected = expected_paths[arg_name]
            if actual != expected:
                raise ValueError(
                    f"{arg_name}.tabularPredictorArgs['path'] is {actual!r} but expected {expected!r}. "
                    "TRIM expects operator.py to inject the correct path from outputDirectory."
                )

        numberEntities = len(list_of_entities)

        train_cols = [
            cp.identifier for cp in discoverySpace.entitySpace.constitutiveProperties
        ]
        train_target_cols = [*train_cols, self.params.targetOutput]

        total_unmeasured = self.handle_unmeasured_targetOutputs_from_no_priors(
            discoverySpace, train_target_cols
        )
        total_measured = 0

        initial_source_df, _target_df = (
            self.get_source_and_target_with_injected_defaults(
                discoverySpace,
                self.params.targetOutput,
            )
        )

        if self.log.isEnabledFor(logging.DEBUG):
            initial_source_df.to_csv(
                os.path.join(self.params.debugDirectory, "initial_source_df.csv")
            )

        msg = (
            f"There are {numberEntities} entities of which TRIM will measure up "
            f"to {self.params.numberEntitiesIterativeModeling}.\n"
            f"These entities have been ordered using {len(initial_source_df)} measurements from the discovery space."
        )
        if (
            self.params.missingTargetMeasurements.mode
            == MissingTargetMeasurementMode.InjectDefaultValue
        ):
            msg += f" There are {total_unmeasured} entities with injected targetOutput={self.params.missingTargetMeasurements.defaultValue}"
        elif (
            self.params.missingTargetMeasurements.mode
            == MissingTargetMeasurementMode.Skip
        ):
            pass

        self.log.info(msg)

        if numberEntities < self.params.numberEntitiesIterativeModeling:
            self.log.warning(
                f"TRIM is configured to measure {self.params.numberEntitiesIterativeModeling} "
                f"entities but there are only {numberEntities} in the space"
            )

        self.log.info(
            f"Training columns are {train_cols},\nThe dependent variable (target Output) is {train_target_cols[-1]}"
        )

        if not self.params.outputDirectory:
            self.log.warning("outputDirectory is empty; defaulting to 'trim_models'.")
            self.params.outputDirectory = "trim_models"

        ############################################################################################################
        ######################################### MAIN LOOP STARTS #################################################
        ############################################################################################################

        metric_dict = {}
        comparison_indices = []
        previous_holdout_df = pd.DataFrame({})
        # Ring-like data structures
        yielded_entities = deque(maxlen=self.params.holdoutSize)
        yielded_rows = RowsRing(
            maxlen=(self.params.holdoutSize or self.params.iterationSize)
        )

        previous_source_df = initial_source_df

        # numberEntitiesIterativeModeling +1 is the exact count RandomWalk will attempt to draw.
        # When TRIM exhausts all entities it could have yielded AND random_walk asks for more then,
        # TRIM finalizes the model and does not yield any more entities.
        for i, entity in enumerate(list_of_entities):
            self.log.info(f"Yielding entity at index {i}: {entity}")
            yield [entity]

            # VV: it's safe to get data out of the sample store, if we're here the RandomWalk
            # operation attempted the Entity we yielded above and it's now asking for one more.
            # First things first, check whether RandomWalk managed to measure the targetOutput for the
            # Entity or not.
            current_source_df, _current_batch_size_target_df = (
                self.get_source_and_target_with_injected_defaults(
                    discoverySpace,
                    self.params.targetOutput,
                )
            )

            compare_to_previous_source_df, one_additional_row = split_common_and_diff(
                longer_df_from_which_you_subtract=current_source_df,
                shorter_df_that_you_subtract=previous_source_df,
            )
            self.log.info(
                f"Entity {entity.identifier} added {len(one_additional_row)} rows"
            )

            if len(one_additional_row) == 0:
                total_unmeasured += 1
                record_unmeasured_entity(
                    entity_identifier=entity.identifier,
                    missing_target_measurements=self.params.missingTargetMeasurements,
                    total_unmeasured=total_unmeasured,
                    target_output=self.params.targetOutput,
                    additional_info="",
                    logger=self.log,
                )

                if (
                    self.params.missingTargetMeasurements.mode
                    == MissingTargetMeasurementMode.InjectDefaultValue
                ):
                    row = {
                        cpv.property.identifier: cpv.value
                        for cpv in entity.constitutive_property_values
                    }
                    self.log.info(
                        f"Injecting default {row} for entity {entity.identifier}"
                    )
                    row[self.params.targetOutput] = (
                        self.params.missingTargetMeasurements.defaultValue
                    )
                    self.injected_defaults = pd.concat(
                        [self.injected_defaults, pd.DataFrame([row])],
                        ignore_index=True,
                    )
                    # VV: This is a yielded row, just a synthetic one not one we actually measured
                    yielded_rows += one_additional_row
                    # VV: Since inject_defaults was updated we need to remember this row so that the next time
                    # we check whether an Entity measured its targetOutput that's the only candidate for a new
                    # row in the current_source_df dataframe
                    current_source_df, _current_batch_size_target_df = (
                        self.get_source_and_target_with_injected_defaults(
                            discoverySpace,
                            self.params.targetOutput,
                        )
                    )

                # Advance the baseline so the next iteration's diff stays at 1 row.
                previous_source_df = current_source_df

                continue
            elif len(one_additional_row) == 1:
                total_measured += 1
                yielded_entities.append(entity)
                yielded_rows += one_additional_row
                self.log.info(
                    f"Measured {entity.identifier} in\n{one_additional_row.to_string()}"
                )
            else:
                self.log.error(
                    f"Entity {entity.identifier}: expected 1 new row in source_df but got {len(one_additional_row)}. "
                    f"previous_source_df={len(previous_source_df)} rows, "
                    f"current_source_df={len(current_source_df)} rows. "
                    f"Extra rows:\n{one_additional_row.to_string()}"
                )
                msg = (
                    f"This is a bug in TRIM. "
                    f"Unexpected additional rows for {entity.identifier}: {one_additional_row.to_string()}"
                )
                self.log.error(msg)
                raise ValueError(msg)

            # TODO: the first holdout set can also be obtained from the source space
            # atm we sample new points from the target and put these into the holdout
            # we can instead look at the source at iter=0 and select within this set the best
            # source and holdout df, the rationale here would be selecting the holdout set first
            # to prioritize representativeness in the OOS set, and put the remaining points in
            # the test set
            if total_measured < self.params.iterationSize:
                log_after_split_common_and_diff(
                    total_measured,
                    compare_to_previous_source_df,
                    previous_source_df,
                    one_additional_row,
                    directory=self.params.debugDirectory,
                )
                previous_source_df = current_source_df
                continue
            elif (
                total_measured == self.params.iterationSize
            ):  # at this point we build the first model
                self.log.info(
                    f"First model: initial_source_df={len(initial_source_df)} rows, "
                    f"current_source_df={len(current_source_df)} rows, "
                    f"expected holdout size={self.params.holdoutSize}"
                )
                train_df, current_holdout_df = split_common_and_diff(
                    longer_df_from_which_you_subtract=current_source_df,
                    shorter_df_that_you_subtract=initial_source_df,
                )

                previous_holdout_df = current_holdout_df

                log_after_first_holdout_creation(
                    current_holdout_df,
                    yielded_rows,
                    iter_index=total_measured,
                    params=self.params,
                )
            else:  # i > self.params.iterationSize
                train_df, one_additional_row = split_common_and_diff(
                    longer_df_from_which_you_subtract=current_source_df,
                    shorter_df_that_you_subtract=previous_source_df,
                )

                log_before_first_holdout_update(
                    one_additional_row,
                    current_source_df,
                    previous_source_df,
                    iter_index=total_measured,
                    debugDirectory=self.params.debugDirectory,
                    batchsize=1,
                )

                current_holdout_df = pd.DataFrame(yielded_rows.df)

                if current_holdout_df.equals(previous_holdout_df):
                    self.log.warning("Holdout dataframe is not changing!")

            # we rename appropriately
            previous_source_df = current_source_df
            previous_holdout_df = current_holdout_df
            if self.log.isEnabledFor(logging.DEBUG):
                save_source_train_holdout_dfs(
                    current_source_df=current_source_df,
                    train_df=train_df,
                    current_holdout_df=current_holdout_df,
                    iter=total_measured,
                    directory=self.params.debugDirectory,
                )

            ##############  MODEL BUILDING AND EVALUATION  #####################
            self.log.info(
                f"Building and evaluating a predictive model "
                f"that includes 1 more entity "
                f"in the training set:\n {entity}"
            )
            # ensures we only train on rows where the target is measured
            # TODO: monitor if this is needed
            train_df = training_guardrail(
                train_df, targetOutput=self.params.targetOutput
            )

            train_data = TabularDataset(train_df)
            holdout_data = TabularDataset(current_holdout_df)

            if len(train_data) < 2:
                self.log.info(
                    f"Skipping model building because there are only {len(train_data)} < 2 train_data rows"
                )
                continue

            if len(holdout_data) < 2:
                self.log.info(
                    f"Skipping model building because there are only {len(holdout_data)} < 2 holdout_data rows"
                )
                continue

            # NOTE: assigning more weight to target space points does NOT generally improve performance
            predictor = TabularPredictor(
                label=self.params.targetOutput,
                **self.params.autoGluonArgs.tabularPredictorArgs,
            )

            self.log.info(
                f"Fitting AutoGluon TabularPredictor, iteration {total_measured}..."
            )
            predictor.fit(train_data=train_data, **self.params.autoGluonArgs.fitArgs)

            # metric metric used in training
            training_metric = getattr(predictor, "eval_metric", None)
            lb = predictor.leaderboard(silent=True)
            if lb is not None and not lb.empty:
                best_row = lb.iloc[0]
                best_model_name = best_row.get("model", None)
                best_score_val = best_row.get("score_val", None)
            else:
                best_model_name, best_score_val = None, None

            metric_dict[total_measured] = {
                "metric": training_metric,
                "best_model": best_model_name,
                "best_score_val": best_score_val,
                "holdout_score": predictor.evaluate(holdout_data, silent=True)[
                    predictor.eval_metric.name
                ],
            }

            self.log.info(
                f"[Batch under consideration: {total_measured}] Training metric: {training_metric};\n"
                f"Best model: {best_model_name}; score_val: {best_score_val:.2f}; holdout_score: {metric_dict[total_measured]['holdout_score']:.2f}",
            )

            # Capture model path and delete the folder
            if not self.log.isEnabledFor(logging.DEBUG):
                model_dir = getattr(predictor, "path", None)
                self.log.info(f"AutoGluon model directory: {model_dir}")
                del predictor
                delete_dir(model_dir=model_dir)

            should_stop = 0

            # for the first 2*iterationSize we do not have enough data to compare
            # i need to go up to self.params.iterationSize * 3
            # if I want that I have one iteration size of models already measured:
            # i<iter_size: no models
            # itersize =< i< itersize *2 : 1st iter of models
            # itersize*2 =< i< itersize *3 : 2nd iter of models
            if (
                total_measured < self.params.iterationSize * 3 - 1
                or not self.params.stoppingCriterion.enabled
            ):
                yielded_entities.append(entity)
                continue

            # NOTE: at the moment comparison does NOT happen at every params.iterationSize steps
            # instead, it happens at every batchsize=1 step, in a rolling fashion,
            else:
                comparison_indices.append(total_measured)
                # NOTE: if batchsize==iterationSize will compare just two models,
                # one model from prev_iter_list_range, whose len would be 1, and
                # one model from this_iter_list_range, whose len would be 1
                _prev_iter_list_range = list(
                    range(
                        total_measured
                        - self.params.iterationSize * 2
                        + 1,  # this index might be included
                        total_measured
                        - self.params.iterationSize
                        + 1,  # this index cannot be included
                    )
                )
                _this_iter_list_range = list(
                    range(
                        total_measured - self.params.iterationSize + 1,
                        total_measured
                        + 1,  # this index cannot be included, but i can be included (this is desired)
                    )
                )
                # I filter these to keep only points that I know correspond to models
                prev_iter_list_range = [
                    i for i in _prev_iter_list_range if i in range(numberEntities)
                ]
                this_iter_list_range = [
                    i for i in _this_iter_list_range if i in range(numberEntities)
                ]

                self.log.info(
                    f"Since iterationSize is {self.params.iterationSize}, "
                    f"We now compare models at the following batch indices\n{prev_iter_list_range}\nand\n{this_iter_list_range}"
                )

                scores_previous_iteration = [
                    float(metric_dict[el]["holdout_score"])
                    for el in prev_iter_list_range
                ]
                scores_this_iteration = [
                    float(metric_dict[el]["holdout_score"])
                    for el in this_iter_list_range
                ]

                self.log.info(
                    f"Scores that correspond to these i-ranges are:\n{scores_previous_iteration}\nand\n{scores_this_iteration}"
                )

                try:
                    mean_ratio = (
                        np.array(scores_this_iteration).mean()
                        / np.array(scores_previous_iteration).mean()
                    )
                    if (
                        np.array(scores_previous_iteration).std()
                        * np.array(scores_this_iteration).std()
                        == 0
                    ):
                        self.log.info(
                            "Product of standard deviation of the scores across batches is 0."
                            "Setting the ratio to 0"
                        )
                        std_ratio = 0

                    else:
                        std_ratio = (
                            np.array(scores_this_iteration).std()
                            / np.array(scores_previous_iteration).std()
                        )
                except Exception as e:
                    self.log.warning(
                        f"Exception occurred: {e}, should stop will be true."
                    )
                    mean_ratio = 1
                    std_ratio = 1
                self.log.info(
                    f"Testing stopping criterion after measuring {total_measured} points, "
                    "mean_ratio={mean_ratio} and std_ratio={std_ratio}"
                )
                should_stop = stopping_bool_from_ratios(
                    mean_ratio=mean_ratio,
                    std_ratio=std_ratio,
                    mean_ratio_threshold=self.params.stoppingCriterion.meanThreshold,
                    std_ratio_threshold=self.params.stoppingCriterion.stdThreshold,
                )

            if should_stop:
                self.log.info(
                    f"Stopping criteria hit after measuring {total_measured} entities.\n"
                    f"On a iteration of batch size {self.params.iterationSize}.\n"
                    "Performance of the model on the holdout set is estimated as:"
                    f"Mean performance of the model on the holdout set over the last iteration: {np.array(scores_this_iteration).mean()}"
                    f"Standard deviation of the performance of the model on the holdout set over the last iteration: {np.array(scores_this_iteration).std()}"
                )
                _predictor = self.finalize_model(
                    discoverySpace=discoverySpace,
                    stopping_criteria_satisfied=True,
                )
                break

        self.log.info(
            f"Finished yielding {len(yielded_entities)} entities. Finalizing model but this "
            "model does not actually satisfy the stopping criteria"
        )
        self.finalize_model(discoverySpace, stopping_criteria_satisfied=False)

    async def remoteEntityIterator(
        self,
        remoteDiscoverySpace: DiscoverySpaceManager,
        batchsize: int = 1,  # type: ignore[name-defined]
    ) -> typing.AsyncGenerator[list[Entity], None]:
        """Returns a remoteEntityIterator that returns entities in order"""

        self.log.debug(f"Trim starts with parameters:\n{self.params}\n\n")

        await self._setup_debug_directory_async()

        discoverySpace = await remoteDiscoverySpace.discoverySpace.remote()
        list_of_entities, _df_ordered_to_sample = (
            self.entities_for_iterative_modeling_from_discovery_space(
                discoverySpace=discoverySpace
            )
        )

        async def async_wrapper() -> typing.AsyncGenerator[list[Entity], None]:
            await asyncio.sleep(0.001)
            for entity in self._core_iterator_logic(discoverySpace, list_of_entities):
                yield entity
                await asyncio.sleep(0.001)  # Allow other async tasks to run

        return async_wrapper()

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize: int = 1
    ) -> typing.Generator[list[Entity], None, None]:
        """Returns an entityIterator that returns entities in order"""

        self.log.debug(f"Trim starts with parameters:\n{self.params}\n\n")

        self._setup_debug_directory_sync()

        list_of_entities, _df_ordered_to_sample = (
            self.entities_for_iterative_modeling_from_discovery_space(
                discoverySpace=discoverySpace
            )
        )

        return self._core_iterator_logic(discoverySpace, list_of_entities)

    def finalize_model(
        self, discoverySpace: DiscoverySpace, stopping_criteria_satisfied: bool
    ) -> TabularPredictor:
        """
        Train a final predictive model on all sampled source space data.

        Args:
            discoverySpace: The discovery space containing the entities
            stopping_criteria_satisfied: Whether the model that is about to be stored
                has met the stopping criteria.

        Returns:
            TabularPredictor: The trained AutoGluon predictor on full source data
        """
        # FIT ON FULL SOURCE SPACE DATA
        source_df, target_df = self.get_source_and_target_with_injected_defaults(
            discoverySpace,
            self.params.targetOutput,
        )

        # TODO: check why len(source_df) is minor than max(i) of the iterative modeling phase
        self.log.info(
            f"Finalizing the predictive model:"
            f"Fitting AutoGluon TabularPredictor on full Source Space data of {len(source_df)} rows."
            f"Model will be saved in: {self.params.finalModelAutoGluonArgs.tabularPredictorArgs['path']}"
        )

        train_cols = [
            cp.identifier for cp in discoverySpace.entitySpace.constitutiveProperties
        ]
        train_target_cols = [*train_cols, self.params.targetOutput]

        train_df = source_df[train_target_cols]
        # think about replicating here the guardrail about NaN in target
        if self.log.isEnabledFor(logging.DEBUG):
            train_df.to_csv(
                os.path.join(
                    self.params.debugDirectory,
                    "final_model_training_data.csv",
                ),
                index=False,
            )

        train_data = TabularDataset(train_df)
        # Now, train a model on new_source_df and get performance
        predictor = TabularPredictor(
            label=self.params.targetOutput,
            **self.params.finalModelAutoGluonArgs.tabularPredictorArgs,
        )

        start_time = time.time()
        predictor.fit(
            train_data=train_data, **self.params.finalModelAutoGluonArgs.fitArgs
        )
        elapsed_time_for_training = time.time() - start_time

        final_lb = predictor.leaderboard(silent=True)
        final_model_metric = (
            final_lb.iloc[0].get("score_val", None)
            if final_lb is not None and not final_lb.empty
            else None
        )
        training_metric = getattr(predictor, "eval_metric", None)
        self.log.info(
            f"Model finalized using as training set all sampled points, of cardinality {len(train_data)}.\n"
            f"Final model {training_metric}={final_model_metric}."
            f"Saving predicted model to: {self.params.finalModelAutoGluonArgs.tabularPredictorArgs['path']}."
        )

        target_predictions = predictor.predict(pd.DataFrame(target_df[train_cols]))
        target_df_with_predictions = target_df.copy()
        target_df_with_predictions[self.params.targetOutput] = target_predictions
        self.log.info(f"Generated predictions for {len(target_df)} target data points.")

        source_df_marked = source_df.copy()
        source_df_marked["is_predicted"] = False
        target_df_with_predictions["is_predicted"] = True

        combined_df = pd.concat(
            [source_df_marked, target_df_with_predictions], ignore_index=True
        )

        combined_df_path = os.path.join(predictor.path, "combined_predictions.csv")
        combined_df.to_csv(combined_df_path, index=False)
        self.log.info(f"Saved combined predictions to: {combined_df_path}")

        if final_lb is not None and not final_lb.empty:
            leaderboard_path = os.path.join(predictor.path, "model_leaderboard.csv")
            final_lb.to_csv(leaderboard_path, index=False)
            self.log.info(f"Saved model leaderboard to: {leaderboard_path}")

        model_card = {
            "train_fraction_wrt_space": len(source_df)
            / (len(source_df) + len(target_df)),
            "size_byte": predictor.disk_usage(),
            "elapsed_time": elapsed_time_for_training,
            "timestamp": datetime.now().isoformat(),
            "training_metric": str(training_metric) if training_metric else None,
            "final_model_metric": final_model_metric,
            "num_train_samples": len(source_df),
            "target_output": self.params.targetOutput,
            "stopping_criteria_satisfied": stopping_criteria_satisfied,
        }

        model_card_path = os.path.join(predictor.path, "model_card.json")
        with open(model_card_path, "w") as f:
            json.dump(model_card, f, indent=2)
        self.log.info(f"Saved model card to: {model_card_path}")

        return predictor

    def entities_for_iterative_modeling_from_discovery_space(
        self,
        discoverySpace: DiscoverySpace,
    ) -> tuple[list, pd.DataFrame]:
        """
        Generate an ordered list of entities for iterative modeling from a discovery space.

        Steps:
        - Validate source data (distinct target values, minimum sampling budget).
        - Compute feature importance and reorder source-target merged dataframe.
        - Determine sampling order using nearest-neighbor strategy.
        - Return ordered entities and the corresponding dataframe.

        Parameters
        ----------
        discoverySpace : DiscoverySpace
            The discovery space containing entities and measured data.

        Returns
        -------
        tuple
            (list_of_entities, df_ordered_to_sample)

        Raises
        ------
        InsufficientDataError
            If data is insufficient for modeling.
        ValueError
            If validation checks fail.
        """

        source_df, target_df = self.get_source_and_target_with_injected_defaults(
            discoverySpace, self.params.targetOutput
        )

        if self.log.isEnabledFor(logging.DEBUG):
            source_df.to_csv(
                os.path.join(self.params.debugDirectory, "Initial_source_space.csv")
            )

        distinct_count = source_df[self.params.targetOutput].nunique(dropna=False)
        if distinct_count == 1:
            unique_val = source_df[self.params.targetOutput].unique()[0]
            msg = (
                f"Target output '{self.params.targetOutput}' has only a single distinct value: {unique_val}. "
                "This is insufficient for downstream processing."
            )
            self.log.error(msg)
            raise InsufficientDataError(msg)

        if len(source_df) < self.params.samplingBudget.minPoints:
            info_str = """This may happen because it may be that the target variable cannot be measured for all
            the entities in the space. For example a recommender could be unable to recommend the target variables
            for some entities"""
            missing_points = self.params.samplingBudget.minPoints - len(source_df)
            self.log.error(
                f"Insufficient data: need {self.params.samplingBudget.minPoints}, but only {len(source_df)} available. "
                f"Consider adding {missing_points} more points or adjusting the budget."
            )
            self.log.info(info_str)
            if len(source_df) > 10:
                self.log.warning(
                    "Attempting iterative modelling with 10 source space points"
                )
            else:
                raise InsufficientDataError(
                    f"Insufficient data: need {self.params.samplingBudget.minPoints}, but only {len(source_df)} available. "
                )

        # Compute feature importance and order
        ordered_features, _importance_dict = get_feature_importance_order(
            source_df=source_df,
            target_output=self.params.targetOutput,
            min_measured_entities=self.params.samplingBudget.minPoints,
            autoGluonArgs=self.params.autoGluonArgs,
        )

        merged_df = source_df.merge(target_df, how="outer")

        if self.log.isEnabledFor(logging.DEBUG):
            merged_df.to_csv(
                os.path.join(self.params.debugDirectory, "initial_debug_merged.csv")
            )
            source_df.to_csv(
                os.path.join(self.params.debugDirectory, "initial_debug_source.csv")
            )
            target_df.to_csv(
                os.path.join(self.params.debugDirectory, "initial_debug_target.csv")
            )

        # Check that rows with NaNs in train_target_cols equal len(target_df)
        nan_rows_count = merged_df[[self.params.targetOutput]].isna().any(axis=1).sum()
        if nan_rows_count != len(target_df):
            msg = (
                f"Validation failed: Expected {len(target_df)} rows with NaNs in {self.params.targetOutput}, "
                f"but found {nan_rows_count}."
            )
            self.log.error(msg)
            raise ValueError(msg)

        # Order merged dataframe by source space feature importance
        merged_df_ordered_by_source_importance = sort_rows_by_column_names(
            merged_df, ordered_features
        )

        # Sampled indices: rows where targetOutput is NOT NaN
        sampled_indices = merged_df_ordered_by_source_importance[
            merged_df_ordered_by_source_importance[self.params.targetOutput].notna()
        ].index.tolist()

        # Compute index order for sampling.
        # get_index_list_van_der_corput seeds its output with sampled_indices and
        # then appends new indices until it reaches tot_points_to_sample.  We
        # therefore ask for len(sampled_indices) + len(target_df) so that after
        # filtering out the already-measured positions exactly len(target_df)
        # new indices remain.
        idx_order = get_index_list_van_der_corput(
            len(merged_df_ordered_by_source_importance),
            len(sampled_indices) + len(target_df),
            sampled_indices=sampled_indices,
        )

        # Filter out sampled indices while maintaining order
        idx_order_filtered = [i for i in idx_order if i not in sampled_indices]

        # Final dataframe to sample
        df_ordered_to_sample = merged_df_ordered_by_source_importance.iloc[
            idx_order_filtered
        ]

        list_of_entities_identifiers = df_ordered_to_sample["identifier"]
        list_of_entities = get_list_of_entities_from_df_and_space(
            df=df_ordered_to_sample, space=discoverySpace
        )

        if self.log.isEnabledFor(logging.DEBUG):
            ordered_df_path_and_name = os.path.join(
                self.params.debugDirectory, "df_ordered_to_sample_with_id.csv"
            )
            ordered_data_log_string = f"DataFrame successfully ordered, saving it now to {ordered_df_path_and_name}"
            self.log.info(ordered_data_log_string)
            self.log.info(
                f"Ordered list of inferred entities identifiers is:\n{list_of_entities_identifiers}\n"
                "Proceeding to sample entities in this order.\n"
                f"Valid entities are built and validated using the dataframe contained in {ordered_df_path_and_name}"
            )
            df_ordered_to_sample.to_csv(ordered_df_path_and_name)

        return list_of_entities, df_ordered_to_sample

    @classmethod
    def parameters_model(cls) -> type[BaseModel] | None:
        return TrimSamplerParametersInternal

    def get_source_and_target_with_injected_defaults(
        self,
        discovery_space: DiscoverySpace | str,
        log_string: str = "",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        source_df, target_df = get_source_and_target(
            discoverySpace=discovery_space,
            targetOutput=self.params.targetOutput,
            log_string=log_string,
        )

        if self.injected_defaults is not None and len(self.injected_defaults) > 0:
            source_df = pd.concat(
                [source_df, self.injected_defaults],
                ignore_index=True,
            )

        return source_df, target_df

    def __init__(self, parameters: TrimSamplerParametersInternal) -> None:
        self.params = parameters

        configure_logging()

        # VV: When self.params.missingTargetMeasurements.mode==InjectDefaultValues
        # The sampler will keep track of the injected values in this dataframe.
        # It's used in get_source_and_target_with_injected_defaults() to concatenate the
        # source_df with self.injected_defaults
        self.injected_defaults: pd.DataFrame | None = None

        self.log = logging.getLogger(__name__)
