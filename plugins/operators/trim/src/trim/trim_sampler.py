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
from datetime import datetime
from typing import TYPE_CHECKING

import anyio
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from ado.core.discoveryspace.samplers import BaseSampler
from trim.samplers.missing_target_utils import (
    entity_hit_in_source,
    entity_measured_target,
    entity_row_in_source,
    record_missing_and_check_budget,
)
from trim.samplers.no_priors_parameters import MissingTargetMode
from trim.samplers.no_priors_utils import (
    get_index_list_van_der_corput,
    get_list_of_entities_from_df_and_space,
    get_source_and_target,
)
from trim.trim_pydantic import TrimParameters

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
    log_unable_to_proceed_with_iterative_modeling_and_raise_error,
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

logger_trim_sampler = logging.getLogger(__name__)


def _make_default_row(
    entity: Entity,
    target_output: str,
    default_value: float,
) -> pd.DataFrame:
    """Build a one-row DataFrame representing a defaulted measurement for an entity.

    The row contains the entity identifier, all constitutive property values,
    and the target output set to ``default_value``. Its column layout matches
    the ``source_df`` produced by :func:`get_source_and_target`.

    Args:
        entity: The entity that did not produce a target measurement.
        target_output: Identifier of the target property column.
        default_value: Value to inject for the target property.

    Returns:
        A single-row :class:`pandas.DataFrame` with columns
        ``['identifier', <cp_ids...>, target_output]``.
    """
    row: dict = {"identifier": entity.identifier}
    for cpv in entity.constitutive_property_values:
        row[cpv.property.identifier] = cpv.value
    row[target_output] = default_value
    return pd.DataFrame([row])


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
        if logger_trim_sampler.isEnabledFor(logging.DEBUG):
            debug_dir = pathlib.Path(self.params.debugDirectory).expanduser().resolve()
            logger_trim_sampler.debug(
                f"Creating a folder to save intermediate files:\n{debug_dir}\n\n"
            )
            debug_dir.mkdir(parents=True, exist_ok=True)

    async def _setup_debug_directory_async(self) -> None:
        """Asynchronously setup debug directory if debug logging is enabled."""
        if logger_trim_sampler.isEnabledFor(logging.DEBUG):
            debug_dir = await anyio.Path(self.params.debugDirectory).expanduser()
            debug_dir = await debug_dir.resolve()
            logger_trim_sampler.debug(
                f"Creating a folder to save intermediate files:\n{debug_dir}\n\n"
            )
            await debug_dir.mkdir(parents=True, exist_ok=True)

    def _handle_missing_target_row(
        self,
        one_additional_row: pd.DataFrame,
        entity: Entity,
        discoverySpace: DiscoverySpace,
        additional_info: str,
    ) -> tuple[pd.DataFrame, bool]:
        """Apply the missing-target policy for an entity not found in ``current_source_df``.

        Raises on ``RaiseError``, injects a default row and returns ``skip=False``
        on ``InjectDefaultValue``, or returns ``skip=True`` on ``Skip``.

        Args:
            one_additional_row: Row DataFrame for the entity (may be empty).
            entity: The entity that did not produce a target measurement.
            discoverySpace: The active discovery space.
            additional_info: Context string appended to error messages.

        Returns:
            ``(one_additional_row, skip)`` — ``skip=True`` means the entity
            should be dropped from the current iteration.
        """
        if entity_hit_in_source(entity, self.current_source_df):
            return one_additional_row, False

        mode = self.params.missingTargetVariables.mode
        entity_id = entity.identifier  # always set after check_identifier validator

        if mode == MissingTargetMode.RaiseError:
            log_unable_to_proceed_with_iterative_modeling_and_raise_error(
                discoverySpace=discoverySpace,
                target_output=self.params.targetOutput,
                additional_info=additional_info,
            )

        self._missing_count = record_missing_and_check_budget(
            params=self.params,
            entity_id=entity_id,  # type: ignore[arg-type]
            missing_count=self._missing_count,
            discoverySpace=discoverySpace,
            additional_info=additional_info,
        )

        if mode == MissingTargetMode.InjectDefaultValue:
            logger_trim_sampler.info(
                f"Entity '{entity_id}' did not produce a measurement for "
                f"target variable '{self.params.targetOutput}'. "
                f"Injecting default value {self.params.missingTargetVariables.defaultValue}."
            )
            one_additional_row = _make_default_row(
                entity,
                self.params.targetOutput,
                self.params.missingTargetVariables.defaultValue,  # type: ignore[arg-type]
            )
            self.current_source_df = pd.concat(
                [self.current_source_df, one_additional_row], ignore_index=True
            )
            return one_additional_row, False

        # MissingTargetMode.Skip
        logger_trim_sampler.info(
            f"Entity '{entity_id}' did not produce a measurement for "
            f"target variable '{self.params.targetOutput}'. Skipping entity."
        )
        return one_additional_row, True

    def _did_entity_measure_target_output(
        self,
        entity: Entity,
        discoverySpace: DiscoverySpace,
        additional_info: str,
    ) -> tuple[pd.DataFrame, bool]:
        """Check whether a just-yielded entity produced a target measurement.

        Appends the new row to ``current_source_df`` on a hit. On a miss,
        delegates to ``_handle_missing_target_row`` to apply the configured
        missing-target policy.

        Args:
            entity: The entity that was just yielded and measured.
            discoverySpace: The active discovery space.
            additional_info: Context string forwarded to the error helper.

        Returns:
            ``(one_additional_row, skip)`` — ``skip=True`` means the entity
            produced no usable measurement and should be dropped.
        """
        hit, series = entity_measured_target(
            entity, discoverySpace, self.params.targetOutput
        )

        if hit:
            one_additional_row = series.to_frame().T.reset_index(drop=True)
            self.current_source_df = pd.concat(
                [self.current_source_df, one_additional_row], ignore_index=True
            )
            return one_additional_row, False

        # Target missing — pass the current source into _handle_missing_target_row
        # so it can apply RaiseError / InjectDefaultValue / Skip without
        # re-fetching the full DataFrame from the store.
        one_additional_row = entity_row_in_source(entity, self.current_source_df)
        return self._handle_missing_target_row(
            one_additional_row=one_additional_row,
            entity=entity,
            discoverySpace=discoverySpace,
            additional_info=additional_info,
        )

    def _no_target_entities_from_no_priors(
        self, discoverySpace: DiscoverySpace
    ) -> list[Entity]:
        """Return entities from the no-priors operation that produced no target measurement.

        Uses ``params.no_priors_operation_id`` to query the measurement requests
        recorded for the no-priors phase, then filters to those whose entity has
        no valid measurement for the target output.

        Returns an empty list when ``no_priors_operation_id`` is ``None`` (i.e.
        no no-priors phase was run).

        Args:
            discoverySpace: The active discovery space.

        Returns:
            List of Entity objects that did not produce a target measurement.
        """
        op_id = self.params.no_priors_operation_id
        if op_id is None:
            return []

        requests = discoverySpace.measurement_requests_for_operation(op_id)
        no_target = [
            entity
            for req in requests
            for entity in req.entities
            if not entity_measured_target(
                entity, discoverySpace, self.params.targetOutput
            )[0]
        ]
        if no_target:
            logger_trim_sampler.warning(
                f"TrimSampleSelector: {len(no_target)} entities from no-priors phase "
                "produced no target measurement."
            )
        return no_target

    def _handle_new_measured_entity(
        self,
        entity: Entity,
        discoverySpace: DiscoverySpace,
        row_entity: pd.DataFrame,
    ) -> bool:
        """Update internal state for a newly measured (or default-injected) entity.

        Builds the holdout, trains a model, and evaluates the stopping criterion
        once enough data has been collected.

        Args:
            entity: The entity whose measurement was just confirmed.
            discoverySpace: The active discovery space.
            row_entity: The measured (or synthetic) row for this entity.

        Returns:
            ``True`` if the stopping criterion was triggered, ``False`` otherwise.
        """
        self.train_df = self.current_source_df
        self.kept_count += 1
        self.yielded_rows += row_entity

        # TODO: the first holdout set can also be obtained from the source space
        # atm we sample new points from the target and put these into the holdout
        # we can instead look at the source at iter=0 and select within this set the best
        # source and holdout df, the rationale here would be selecting the holdout set first
        # to prioritize representativeness in the OOS set, and put the remaining points in
        # the test set
        if self.kept_count < self.params.iterationSize:
            return False
        if (
            self.kept_count == self.params.iterationSize
        ):  # at this point we build the first model
            self.train_df, current_holdout_df = split_common_and_diff(
                longer_df_from_which_you_subtract=self.current_source_df,
                shorter_df_that_you_subtract=self.initial_source_df,
            )

            self.last_holdout_df = current_holdout_df

            log_after_first_holdout_creation(
                current_holdout_df,
                self.yielded_rows,
                iter_index=self.kept_count,
                params=self.params,
            )
            return False
        # kept_count > self.params.iterationSize
        row_entity, skip = self._handle_missing_target_row(
            one_additional_row=row_entity,
            entity=entity,
            discoverySpace=discoverySpace,
            additional_info=f"Detected during Iterative Modeling, when the training DataFrame size is {len(self.train_df)}.",
        )
        if skip:
            return False

        current_holdout_df = pd.DataFrame(self.yielded_rows.df)

        if current_holdout_df.equals(self.last_holdout_df):
            logger_trim_sampler.warning("Holdout dataframe is not changing!")

        # we rename appropriately
        self.last_holdout_df = current_holdout_df
        if logger_trim_sampler.isEnabledFor(logging.DEBUG):
            save_source_train_holdout_dfs(
                current_source_df=self.current_source_df,
                train_df=self.train_df,
                current_holdout_df=current_holdout_df,
                iter=self.kept_count,
                directory=self.params.debugDirectory,
            )

        ##############  MODEL BUILDING AND EVALUATION  #####################

        logger_trim_sampler.info(
            "Building and evaluating a predictive model that includes 1 more entity "
            "in the training set:\n {entity}"
        )
        # ensures we only train on rows where the target is measured
        # TODO: monitor if this is needed
        self.train_df = training_guardrail(
            self.train_df, targetOutput=self.params.targetOutput
        )

        train_data = TabularDataset(self.train_df)
        holdout_data = TabularDataset(current_holdout_df)

        # NOTE: assigning more weight to target space points does NOT generally improve performance
        predictor = TabularPredictor(
            label=self.params.targetOutput,
            **self.params.autoGluonArgs.tabularPredictorArgs,
        )

        logger_trim_sampler.info(
            f"Fitting AutoGluon TabularPredictor, iteration {self.kept_count}..."
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

        self.metric_dict[self.kept_count] = {
            "metric": training_metric,
            "best_model": best_model_name,
            "best_score_val": best_score_val,
            "holdout_score": predictor.evaluate(holdout_data, silent=True)[
                predictor.eval_metric.name
            ],
        }

        logger_trim_sampler.info(
            f"[Batch under consideration: {self.kept_count}] Training metric: {training_metric};\n"
            f"Best model: {best_model_name}; score_val: {best_score_val:.2f}; holdout_score: {self.metric_dict[self.kept_count]['holdout_score']:.2f}",
        )

        # Capture model path and delete the folder
        if not logger_trim_sampler.isEnabledFor(logging.DEBUG):
            model_dir = getattr(predictor, "path", None)
            logger_trim_sampler.info(f"AutoGluon model directory: {model_dir}")
            del predictor
            delete_dir(model_dir=model_dir)

        should_stop = 0

        # for the first 2*iterationSize we do not have enough data to compare
        # kept_count < iter_size: no models
        # iter_size <= kept_count < iter_size*2 : 1st iter of models
        # iter_size*2 <= kept_count < iter_size*3 : 2nd iter of models
        if (
            self.kept_count < self.params.iterationSize * 3 - 1
            or not self.params.stoppingCriterion.enabled
        ):
            return False

        # NOTE: at the moment comparison does NOT happen at every params.iterationSize steps
        # instead, it happens at every batchsize=1 step, in a rolling fashion,
        # NOTE: if batchsize==iterationSize will compare just two models,
        # one model from prev_iter_list_range, whose len would be 1, and
        # one model from this_iter_list_range, whose len would be 1
        _prev_iter_list_range = list(
            range(
                self.kept_count
                - self.params.iterationSize * 2
                + 1,  # this index might be included
                self.kept_count
                - self.params.iterationSize
                + 1,  # this index cannot be included
            )
        )
        _this_iter_list_range = list(
            range(
                self.kept_count - self.params.iterationSize + 1,
                self.kept_count
                + 1,  # this index cannot be included, but kept_count can be included (this is desired)
            )
        )
        # I filter these to keep only points that I know correspond to models
        prev_iter_list_range = [
            k for k in _prev_iter_list_range if k in self.metric_dict
        ]
        this_iter_list_range = [
            k for k in _this_iter_list_range if k in self.metric_dict
        ]

        logger_trim_sampler.info(
            f"Since iterationSize is {self.params.iterationSize}, "
            f"We now compare models at the following batch indices\n{prev_iter_list_range}\nand\n{this_iter_list_range}"
        )

        scores_previous_iteration = [
            float(self.metric_dict[el]["holdout_score"]) for el in prev_iter_list_range
        ]
        scores_this_iteration = [
            float(self.metric_dict[el]["holdout_score"]) for el in this_iter_list_range
        ]

        logger_trim_sampler.info(
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
                logger_trim_sampler.info(
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
            logger_trim_sampler.warning(
                f"Exception occurred: {e}, should stop will be true."
            )
            mean_ratio = 1
            std_ratio = 1
        logger_trim_sampler.info(
            f"Testing stopping criterion after measuring {self.kept_count} points, "
            "mean_ratio={mean_ratio} and std_ratio={std_ratio}"
        )
        should_stop = stopping_bool_from_ratios(
            mean_ratio=mean_ratio,
            std_ratio=std_ratio,
            mean_ratio_threshold=self.params.stoppingCriterion.meanThreshold,
            std_ratio_threshold=self.params.stoppingCriterion.stdThreshold,
        )

        if should_stop:
            # Stopping info
            self.params.finalModelAutoGluonArgs.tabularPredictorArgs["path"] = (
                self.params.finalModelAutoGluonArgs.tabularPredictorArgs.get(
                    "path", self.params.outputDirectory
                )
                or ""
            ) + "_finalized"

            logger_trim_sampler.info(
                f"Stopping criteria hit after measuring {self.kept_count} entities.\n"
                f"On a iteration of batch size {self.params.iterationSize}.\n"
                "Performance of the model on the holdout set is estimated as:"
                f"Mean performance of the model on the holdout set over the last iteration: {np.array(scores_this_iteration).mean()}"
                f"Standard deviation of the performance of the model on the holdout set over the last iteration: {np.array(scores_this_iteration).std()}"
            )
            _predictor = self.finalize_model(
                discoverySpace=discoverySpace,
            )
            return True

        logger_trim_sampler.info(
            f"Stopping not triggered for kept_count={self.kept_count}"
        )

        return False

    def _core_iterator_logic(
        self,
        discoverySpace: DiscoverySpace,
        list_of_entities: list[Entity],
        batchsize: int,
    ) -> typing.Generator[list[Entity], None, None]:
        """
        Core iterator logic shared between sync and async implementations.
        This is a synchronous generator that yields entities based on the TRIM algorithm.
        """
        # Filter entities that no-priors already flagged as unable to yield a target.
        no_target_entities = self._no_target_entities_from_no_priors(discoverySpace)
        skip_set = {e.identifier for e in no_target_entities}
        if skip_set:
            logger_trim_sampler.warning(
                f"TrimSampleSelector: removing {len(skip_set)} pre-skipped entities "
                f"from list_of_entities before the main loop."
            )
            list_of_entities = [
                e for e in list_of_entities if e.identifier not in skip_set
            ]

        numberEntities = len(list_of_entities)

        if not numberEntities:
            return

        self.initial_source_df, _target_df = get_source_and_target(
            discoverySpace,
            self.params.targetOutput,
        )

        # In InjectDefaultValue mode, synthesise rows for the pre-skipped entities
        # so they are present in initial_source_df for all AutoGluon training.
        if (
            no_target_entities
            and self.params.missingTargetVariables.mode
            == MissingTargetMode.InjectDefaultValue
        ):
            default_val = self.params.missingTargetVariables.defaultValue
            skip_rows = [
                _make_default_row(entity, self.params.targetOutput, default_val)  # type: ignore[arg-type]
                for entity in no_target_entities
            ]
            self.initial_source_df = pd.concat(
                [self.initial_source_df, *skip_rows], ignore_index=True
            )
            logger_trim_sampler.info(
                f"Prepended {len(skip_rows)} default rows for pre-skipped entities "
                f"into initial_source_df (defaultValue={default_val})."
            )

        if logger_trim_sampler.isEnabledFor(logging.DEBUG):
            self.initial_source_df.to_csv(
                os.path.join(self.params.debugDirectory, "initial_source_df.csv")
            )

        train_cols = [
            cp.identifier for cp in discoverySpace.entitySpace.constitutiveProperties
        ]
        train_target_cols = [*train_cols, self.params.targetOutput]
        logger_trim_sampler.info(
            f"Trim iterator will measure up to {numberEntities} entities.\n"
            f"These entities have been ordered using {len(self.initial_source_df)} measurements from the discovery space."
        )

        logger_trim_sampler.info(
            f"Training columns are {train_cols},\nThe dependent variable (target Output) is {train_target_cols[-1]}"
        )

        ############################################################################################################
        ######################################### MAIN LOOP STARTS #################################################
        ############################################################################################################

        self.kept_count = 0
        self.metric_dict = {}
        self._missing_count = 0
        self.train_df = pd.DataFrame({})
        self.last_holdout_df = pd.DataFrame({})
        self.yielded_rows = RowsRing(
            maxlen=(self.params.holdoutSize or self.params.iterationSize)
        )
        self.current_source_df = self.initial_source_df

        last_entity = None
        one_additional_row = None

        # This iterator is can be consumed in 2 ways, async and sync.
        # In async mode we cannot reliably get the results of the entity after yielding it,
        # the wrapper that iterates this generator calls asyncio.sleep() so it's safe to
        # get the results of the last yielded entity

        last_entity = list_of_entities[0]
        logger_trim_sampler.info(f"Yielding {last_entity}")
        yield list_of_entities[0:1]

        for entity in list_of_entities[1:]:
            one_additional_row, skip = self._did_entity_measure_target_output(
                entity=last_entity,
                discoverySpace=discoverySpace,
                additional_info=f"Detected during Iterative Modeling (first entity), when the source space size is {len(self.train_df)}.",
            )

            if skip:
                # Couldn't measure the last entity, try measuring this one and handle it in the next iteration
                last_entity = entity
                yield [entity]
                continue

            stop = self._handle_new_measured_entity(
                entity=last_entity,
                discoverySpace=discoverySpace,
                row_entity=one_additional_row,
            )

            if stop:
                return

            # Haven't gathered enough points yet, yield the entity and handle it in the next iteration
            last_entity = entity
            yield [entity]

        # If we got here, this means that we yielded the very last Entity and need to handle it
        if last_entity is not None:
            one_additional_row, skip = self._did_entity_measure_target_output(
                entity=last_entity,
                discoverySpace=discoverySpace,
                additional_info=f"Detected during Iterative Modeling ({self.kept_count+1} entity), when the source space size is {len(self.train_df)}.",
            )

            if not skip:
                self._handle_new_measured_entity(
                    entity=last_entity,
                    discoverySpace=discoverySpace,
                    row_entity=one_additional_row,
                )

    async def remoteEntityIterator(
        self,
        remoteDiscoverySpace: DiscoverySpaceManager,
        batchsize: int = 1,  # type: ignore[name-defined]
    ) -> typing.AsyncGenerator[list[Entity], None]:
        """Returns a remoteEntityIterator that returns entities in order"""

        logger_trim_sampler.debug(f"Batchsize is {batchsize} (expected 1)")

        logger_trim_sampler.debug(f"Trim starts with parameters:\n{self.params}\n\n")

        await self._setup_debug_directory_async()

        discoverySpace = await remoteDiscoverySpace.discoverySpace.remote()
        list_of_entities, _df_ordered_to_sample = (
            self.entities_for_iterative_modeling_from_discovery_space(
                discoverySpace=discoverySpace
            )
        )

        async def async_wrapper() -> typing.AsyncGenerator[list[Entity], None]:
            await asyncio.sleep(0.001)
            for entity_batch in self._core_iterator_logic(
                discoverySpace, list_of_entities, batchsize
            ):
                yield entity_batch
                await asyncio.sleep(0.001)  # Allow other async tasks to run

        return async_wrapper()

    def entityIterator(
        self, discoverySpace: DiscoverySpace, batchsize: int = 1
    ) -> typing.Generator[list[Entity], None, None]:
        """Returns an entityIterator that returns entities in order"""

        logger_trim_sampler.debug(f"Batchsize is {batchsize} (expected 1)")

        logger_trim_sampler.debug(f"Trim starts with parameters:\n{self.params}\n\n")

        self._setup_debug_directory_sync()

        list_of_entities, _df_ordered_to_sample = (
            self.entities_for_iterative_modeling_from_discovery_space(
                discoverySpace=discoverySpace
            )
        )

        return self._core_iterator_logic(discoverySpace, list_of_entities, batchsize)

    def finalize_model(
        self,
        discoverySpace: DiscoverySpace,
    ) -> TabularPredictor:
        """
        Train a final predictive model on all sampled source space data.

        Args:
            discoverySpace: The discovery space containing the entities

        Returns:
            TabularPredictor: The trained AutoGluon predictor on full source data
        """
        # FIT ON FULL SOURCE SPACE DATA
        source_df, target_df = get_source_and_target(
            discoverySpace,
            self.params.targetOutput,
        )

        # TODO: check why len(source_df) is minor than max(i) of the iterative modeling phase
        logger_trim_sampler.info(
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
        if logger_trim_sampler.isEnabledFor(logging.DEBUG):
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
            # problem_type="regression", # it is inferred atm
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
        logger_trim_sampler.info(
            f"Model finalized using as training set all sampled points, of cardinality {len(train_data)}.\n"
            f"Final model {training_metric}={final_model_metric}."
            f"Saving predicted model to: {self.params.finalModelAutoGluonArgs.tabularPredictorArgs['path']}."
        )

        target_predictions = predictor.predict(pd.DataFrame(target_df[train_cols]))
        target_df_with_predictions = target_df.copy()
        target_df_with_predictions[self.params.targetOutput] = target_predictions
        logger_trim_sampler.info(
            f"Generated predictions for {len(target_df)} target data points."
        )

        source_df_marked = source_df.copy()
        source_df_marked["is_predicted"] = False
        target_df_with_predictions["is_predicted"] = True

        combined_df = pd.concat(
            [source_df_marked, target_df_with_predictions], ignore_index=True
        )

        combined_df_path = os.path.join(predictor.path, "combined_predictions.csv")
        combined_df.to_csv(combined_df_path, index=False)
        logger_trim_sampler.info(f"Saved combined predictions to: {combined_df_path}")

        if final_lb is not None and not final_lb.empty:
            leaderboard_path = os.path.join(predictor.path, "model_leaderboard.csv")
            final_lb.to_csv(leaderboard_path, index=False)
            logger_trim_sampler.info(f"Saved model leaderboard to: {leaderboard_path}")

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
        }

        model_card_path = os.path.join(predictor.path, "model_card.json")
        with open(model_card_path, "w") as f:
            json.dump(model_card, f, indent=2)
        logger_trim_sampler.info(f"Saved model card to: {model_card_path}")

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

        source_df, target_df = get_source_and_target(
            discoverySpace, self.params.targetOutput
        )

        # In InjectDefaultValue mode inject synthetic default rows for any entities
        # from the no-priors phase that produced no target measurement.  This must
        # happen before the minPoints check so those rows count towards the budget.
        if (
            self.params.missingTargetVariables.mode
            == MissingTargetMode.InjectDefaultValue
        ):
            no_target_entities = self._no_target_entities_from_no_priors(discoverySpace)
            if no_target_entities:
                default_val = self.params.missingTargetVariables.defaultValue
                default_rows = [
                    _make_default_row(entity, self.params.targetOutput, default_val)  # type: ignore[arg-type]
                    for entity in no_target_entities
                ]
                source_df = pd.concat([source_df, *default_rows], ignore_index=True)
                logger_trim_sampler.info(
                    f"Injected {len(default_rows)} default rows into source_df "
                    f"(defaultValue={default_val})."
                )

        if logger_trim_sampler.isEnabledFor(logging.DEBUG):
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
            logger_trim_sampler.error(msg)
            raise InsufficientDataError(msg)

        if len(source_df) < self.params.samplingBudget.minPoints:
            info_str = """This may happen because it may be that the target variable cannot be measured for all
            the entities in the space. For example a recommender could be unable to recommend the target variables
            for some entities"""
            missing_points = self.params.samplingBudget.minPoints - len(source_df)
            logger_trim_sampler.error(
                f"Insufficient data: need {self.params.samplingBudget.minPoints}, but only {len(source_df)} available. "
                f"Consider adding {missing_points} more points or adjusting the budget."
            )
            logger_trim_sampler.info(info_str)
            if len(source_df) > 10:
                logger_trim_sampler.info(
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

        if logger_trim_sampler.isEnabledFor(logging.DEBUG):
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
        nan_rows_count = merged_df[[self.params.targetOutput]].isna().any(axis=1).sum()  # type: ignore[union-attr]
        if nan_rows_count != len(target_df):
            msg = (
                f"Validation failed: Expected {len(target_df)} rows with NaNs in {self.params.targetOutput}, "
                f"but found {nan_rows_count}."
            )
            logger_trim_sampler.error(msg)
            raise ValueError(msg)

        # Order merged dataframe by source space feature importance
        merged_df_ordered_by_source_importance = sort_rows_by_column_names(
            merged_df, ordered_features
        )

        # Sampled indices: rows where targetOutput is NOT NaN
        sampled_indices = merged_df_ordered_by_source_importance[
            merged_df_ordered_by_source_importance[self.params.targetOutput].notna()
        ].index.tolist()

        # Compute index order for sampling
        idx_order = get_index_list_van_der_corput(
            len(merged_df_ordered_by_source_importance),
            len(target_df),
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

        if logger_trim_sampler.isEnabledFor(logging.DEBUG):
            ordered_df_path_and_name = os.path.join(
                self.params.debugDirectory, "df_ordered_to_sample_with_id.csv"
            )
            ordered_data_log_string = f"DataFrame successfully ordered, saving it now to {ordered_df_path_and_name}"
            logger_trim_sampler.info(ordered_data_log_string)
            logger_trim_sampler.info(
                f"Ordered list of inferred entities identifiers is:\n{list_of_entities_identifiers}\n"
                "Proceeding to sample entities in this order.\n"
                f"Valid entities are built and validated using the dataframe contained in {ordered_df_path_and_name}"
            )
            df_ordered_to_sample.to_csv(ordered_df_path_and_name)

        return list_of_entities, df_ordered_to_sample

    @classmethod
    def parameters_model(cls) -> type[BaseModel] | None:
        return TrimParameters

    def __init__(self, parameters: TrimParameters) -> None:
        # Sampler configuration parameters.
        self.params = parameters
        # Running count of entities that did not produce a target measurement.
        self._missing_count: int = 0
        # Entities yielded so far; kept for potential post-hoc inspection.
        self.yielded_entities: list[Entity] = []
        # Per-iteration model metrics keyed by kept_count; used by the stopping criterion.
        self.metric_dict: dict = {}

        # The most recent holdout DataFrame; updated each time a new holdout is built.
        self.last_holdout_df: pd.DataFrame = pd.DataFrame({})
        # Training DataFrame for the current AutoGluon fit; equals current_source_df minus NaN rows.
        self.train_df: pd.DataFrame = pd.DataFrame({})
        # Incrementally maintained source DataFrame; one row appended per measured entity.
        self.current_source_df: pd.DataFrame = pd.DataFrame({})
        # Snapshot of the source DataFrame at the start of _core_iterator_logic.
        self.initial_source_df: pd.DataFrame = pd.DataFrame({})
        # Number of entities successfully kept (measured or default-injected, not skipped).
        self.kept_count: int = 0
        # Sliding window of the last holdoutSize measured rows; used to build the rolling holdout.
        self.yielded_rows = RowsRing(
            maxlen=(self.params.holdoutSize or self.params.iterationSize)
        )
