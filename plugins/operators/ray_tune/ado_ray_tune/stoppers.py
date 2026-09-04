# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import logging
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import ray.tune
from ray.tune.stopper import CombinedStopper, Stopper

if TYPE_CHECKING:
    import pandas as pd


class AdoStopperMixin:
    """Status helpers for ado stoppers after a Ray Tune run ends."""

    should_stop: bool

    def did_trigger(self) -> bool:
        """Return True if this stopper decided the experiment should stop."""
        return bool(self.should_stop)

    def progress_message(self) -> str:
        """Return a one-line INFO-level description of current stopping progress."""
        raise NotImplementedError

    def final_report(self) -> str | None:
        """Return a stdout payload describing results, or None if there is none."""
        return None


SAMPLING_BUDGET_HALT_REASON = (
    "RayTune operation stopped because sampling budget reached."
)


class StopperStatus(NamedTuple):
    """Name and optional final log for one stopper after a run."""

    name: str
    log: str | None


class StopperRunReport(NamedTuple):
    """Halt reason and per-stopper status after tuner.fit()."""

    halt_reason: str
    triggered: list[StopperStatus]
    not_triggered: list[StopperStatus]

    def stopper_logs(self) -> dict[str, str]:
        """Return name to log for stoppers that produced a final report.

        Returns:
            A mapping of stopper class name to ``final_report()`` text.
        """
        logs: dict[str, str] = {}
        for status in (*self.triggered, *self.not_triggered):
            if status.log:
                logs[status.name] = status.log
        return logs


def iter_stoppers(stop: Stopper | None) -> list[Stopper]:
    """Return configured stoppers, unwrapping nested CombinedStopper instances.

    Args:
        stop: A Ray Tune stopper, CombinedStopper, or None.

    Returns:
        The leaf stoppers in configuration order.
    """
    if stop is None:
        return []
    if isinstance(stop, CombinedStopper):
        result: list[Stopper] = []
        for child in stop._stoppers:
            result.extend(iter_stoppers(child))
        return result
    if isinstance(stop, Stopper):
        return [stop]
    return []


def stopper_did_trigger(stopper: Stopper) -> bool:
    """Return whether a stopper reported that it ended the experiment.

    Prefers ado ``did_trigger()`` / ``should_stop``, then Ray Tune ``stop_all()``.

    Args:
        stopper: A Ray Tune or ado stopper instance.

    Returns:
        True if the stopper indicates it triggered.
    """
    did_trigger = getattr(stopper, "did_trigger", None)
    if callable(did_trigger):
        return bool(did_trigger())
    should_stop = getattr(stopper, "should_stop", None)
    if should_stop is not None:
        return bool(should_stop)
    return bool(stopper.stop_all())


def _stopper_status(stopper: Stopper) -> StopperStatus:
    """Build a status record from a stopper instance.

    Args:
        stopper: A Ray Tune or ado stopper.

    Returns:
        The stopper class name and ``final_report()`` text, if any.
    """
    final_report = getattr(stopper, "final_report", None)
    log = final_report() if callable(final_report) else None
    return StopperStatus(name=type(stopper).__name__, log=log or None)


def _halt_reason(triggered: list[StopperStatus]) -> str:
    """Return the one-line halt reason for a run.

    Args:
        triggered: Stoppers that reported they ended the experiment.

    Returns:
        Sampling-budget wording, or a sentence naming the triggered stoppers.
    """
    if not triggered:
        return SAMPLING_BUDGET_HALT_REASON
    names = ", ".join(status.name for status in triggered)
    return f"RayTune operation stopped because stopper {names}."


def report_stoppers_after_fit(
    stop: Stopper | None,
    *,
    num_trials: int,
    num_samples: int | None = None,
) -> StopperRunReport:
    """Build the halt summary after tuner.fit().

    Args:
        stop: The RunConfig stopper (possibly a CombinedStopper).
        num_trials: Number of trials in the ResultGrid.
        num_samples: ``tuneConfig.num_samples``, if set.

    Returns:
        Halt reason plus per-stopper names and logs.
    """
    del num_trials, num_samples
    stoppers = iter_stoppers(stop)
    if not stoppers:
        return StopperRunReport(
            halt_reason=SAMPLING_BUDGET_HALT_REASON,
            triggered=[],
            not_triggered=[],
        )

    triggered = [_stopper_status(s) for s in stoppers if stopper_did_trigger(s)]
    not_triggered = [_stopper_status(s) for s in stoppers if not stopper_did_trigger(s)]
    return StopperRunReport(
        halt_reason=_halt_reason(triggered),
        triggered=triggered,
        not_triggered=not_triggered,
    )


def format_stopper_run_report(report: StopperRunReport) -> str:
    """Format the halt summary and stopper logs for stdout.

    Args:
        report: Messages produced by ``report_stoppers_after_fit``.

    Returns:
        Text to print at the end of ``RayTune.run()``.
    """
    lines = [report.halt_reason]
    lines.extend(status.log for status in report.triggered if status.log)
    for status in report.not_triggered:
        lines.append(f"The following stoppers did not fire. {status.name}.")
        if status.log:
            lines.append(f"The final status of the stopper was\n{status.log}")
    return "\n".join(lines)


def emit_stopper_run_report(report: StopperRunReport) -> None:
    """Print the halt summary and stopper logs to stdout.

    Args:
        report: Messages produced by ``report_stoppers_after_fit``.
    """
    print(format_stopper_run_report(report))


class SimpleStopper(AdoStopperMixin, ray.tune.Stopper):
    def __init__(self) -> None:
        self.min_trials = None
        self.trials_num = 0
        self._is_better = None
        self.metric = None
        self.last_metric = None
        self.wait_until_stop = None
        self.stop_on_repeat = None
        self.count_nan = None
        self.experiment_key = None
        self.last_experiment = None
        self.should_stop = False
        self.seen_trial_ids = []
        self.log = logging.getLogger("SimpleStopper")

    # TODO: I don't know why init isn't accepting parameters...
    def set_config(
        self,
        mode: str,
        metric: str,
        experiment_key: str = "config",
        min_trials: int = 5,
        buffer_states: int = 2,
        stop_on_repeat: bool = True,
        count_nan: bool = True,
    ) -> None:
        # self.mode = mode
        self.min_trials = int(min_trials)

        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be either max or min (was {mode})")

        if mode == "max":
            self._is_better = lambda x, y: x > y
            self.last_metric = -float("inf")
        if mode == "min":
            self._is_better = lambda x, y: x < y
            self.last_metric = float("inf")
        self.metric = metric
        self.wait_until_stop = int(buffer_states)
        self.stop_on_repeat = bool(stop_on_repeat)
        self.count_nan = count_nan
        self.experiment_key = experiment_key
        self.last_experiment = None
        self.should_stop = False
        self.log.debug(
            f"configured: experiment_key {experiment_key}; min_trials {min_trials}; "
            f"buffer_states {buffer_states}; stop_on_repeat {stop_on_repeat}"
        )

    def __str__(self) -> str:
        return (
            f"SimpleStopper({self.wait_until_stop}, minimum trials: {self.min_trials})"
        )

    def progress_message(self) -> str:
        """Return a one-line description of SimpleStopper progress."""
        if self.should_stop:
            return f"SimpleStopper: stopping after {self.trials_num} samples."
        if self.min_trials is not None and self.trials_num <= self.min_trials:
            return (
                f"SimpleStopper: not stopping: {self.trials_num}/{self.min_trials} "
                f"samples in grace period."
            )
        return (
            f"SimpleStopper: not stopping after {self.trials_num} samples. "
            f"Remaining samples without improvement before stop: {self.wait_until_stop}."
        )

    def __call__(self, trial_id: str, result: dict[str, Any]) -> bool:
        if trial_id in self.seen_trial_ids:
            self.log.debug(f"Already seen this trial: {trial_id}...")
            return False
        self.trials_num += 1
        self.seen_trial_ids.append(trial_id)
        exp_key = result[self.experiment_key]
        metric = result[self.metric]
        in_grace_period = self.trials_num <= self.min_trials
        self.log.debug(
            f"Trial {trial_id} with {result}; in grace period {in_grace_period}"
        )
        if (
            self.stop_on_repeat
            and exp_key == self.last_experiment
            and not in_grace_period
        ):
            self.log.info(f"observed same experiment twice {exp_key}...stopping...")
            self.should_stop = True
            return True
        self.last_experiment = exp_key
        if (metric is None or np.isnan(metric)) and not self.count_nan:
            self.log.debug(f"Trial {trial_id} has metric value {metric}; skipping...")
            return False
        if not self._is_better(metric, self.last_metric):
            if not in_grace_period:
                self.wait_until_stop -= 1
                if self.wait_until_stop < 0:
                    self.log.info("Metric didn't improve in a while, stopping...")
                    self.should_stop = True
                    return True
        else:
            self.last_metric = metric
        self.log.info(self.progress_message())
        return False

    def stop_all(self) -> bool:
        return self.should_stop


class GrowthStopper(AdoStopperMixin, ray.tune.Stopper):
    def __init__(self) -> None:
        self._fist_call = True
        self.metric = None
        self.last_result = None
        self.grace_trials = None
        self.growth_threshold = None
        self.should_stop = False
        self.seen_trial_ids = []
        self.log = logging.getLogger("GrowthStopper")

    # TODO: I don't know why init isn't accepting parameters...
    def set_config(
        self,
        mode: str,
        metric: str,
        growth_threshold: float = 1.0,
        grace_trials: int = 2,
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be either max or min (was {mode})")

        # if mode == 'max':
        #     self.last_result = -float('inf')
        # if mode == 'min':
        #     self.last_result = float('inf')
        self.last_result = 0.0
        self._fist_call = True
        self.metric = metric
        self.growth_threshold = float(growth_threshold)
        self.grace_trials = int(grace_trials)
        self.should_stop = False
        self.seen_trial_ids = []
        self.log.debug(
            f"configured: growth_threshold {self.growth_threshold}; grace trials {self.grace_trials};"
        )

    def __str__(self) -> str:
        return f"GrowthStopper({self.growth_threshold}, grace {self.grace_trials})"

    def progress_message(self) -> str:
        """Return a one-line description of GrowthStopper progress."""
        n_samples = len(self.seen_trial_ids)
        if self.should_stop:
            return f"GrowthStopper: stopping after {n_samples} samples."
        return (
            f"GrowthStopper: not stopping after {n_samples} samples. "
            f"Remaining grace trials: {self.grace_trials}."
        )

    def __call__(self, trial_id: str, result: dict[str, Any]) -> bool:
        if trial_id in self.seen_trial_ids:
            self.log.debug(f"already seen this trial {trial_id}...")
            return False
        self.seen_trial_ids.append(trial_id)
        metric = result[self.metric]
        diff = np.abs(metric - self.last_result)
        self.log.debug(f"Trial {trial_id} with {result}; with diff {diff};")
        if not self._fist_call:
            if diff < self.growth_threshold:
                self.grace_trials -= 1
                self.log.debug(
                    f"Trial {trial_id}: below threshold; remaining grace trials {self.grace_trials};"
                )
                if self.grace_trials < 0:
                    self.log.info("Metric didn't improve in a while, stopping...")
                    self.should_stop = True
                    return True
        else:
            self._fist_call = False
            self.last_result = metric
        self.log.info(self.progress_message())
        return False

    def stop_all(self) -> bool:
        return self.should_stop


class MaxSamplesStopper(AdoStopperMixin, ray.tune.Stopper):
    # apparently, ray.tune.stopper.maximum_iteration doesn't stop after samples,
    #  but iterations per sample??
    # also, for different optimizers, "num_samples" doesn't mean the same, some count it without random iterations,
    # some then without the historic data, etc.

    def __init__(self) -> None:
        self.max_samples = None
        self.trials_num = 0
        self.should_stop = False
        self.seen_trial_ids = []
        self.log = logging.getLogger("MaxSamplesStopper")

    # TODO: I don't know why init isn't accepting parameters...
    def set_config(self, max_samples: int) -> None:
        self.max_samples = max_samples
        self.log.debug(f"onfigured: max_samples {max_samples}")

    def __str__(self) -> str:
        return f"MaxSamplesStopper({self.max_samples})"

    def progress_message(self) -> str:
        """Return a one-line description of MaxSamplesStopper progress."""
        if self.should_stop:
            return (
                f"MaxSamplesStopper: stopping after {self.trials_num} samples "
                f"(max_samples={self.max_samples})."
            )
        return (
            f"MaxSamplesStopper: not stopping: {self.trials_num}/{self.max_samples} "
            f"samples."
        )

    def __call__(self, trial_id: str, result: dict[str, Any]) -> bool:
        if trial_id in self.seen_trial_ids:
            # it is always called twice per trial...?
            self.log.debug(f"already seen this trial {trial_id}...")
            return False
        self.seen_trial_ids.append(trial_id)
        self.trials_num += 1
        if self.trials_num >= self.max_samples:
            self.log.info(f"maximum trials {self.trials_num} reached, stopping...")
            self.should_stop = True
            return True
        self.log.info(self.progress_message())
        return False

    def stop_all(self) -> bool:
        return self.should_stop


class InformationGainStopper(AdoStopperMixin, ray.tune.Stopper):
    # apparently, ray.tune.stopper.maximum_iteration doesn't stop after samples,
    #  but iterations per sample??
    # also, for different optimizers, "num_samples" doesn't mean the same, some count it without random iterations,
    # some then without the historic data, etc.

    def __init__(self) -> None:
        # to check dependencies...

        self.min_samples = None
        self.trials_num = 0
        self.failed_trials_num = 0
        self.failed_trials_to_consider_num = 0
        self.last_calculation_step_num = 0
        self.consider_pareto_front_convergence = False
        self.consider_invalid_experiment_factor = None
        self.should_stop = False
        self.seen_trial_ids = []
        self.mi_diff_limit = None
        self.samples_below_limit = None
        self.data_columns = []
        self.search_columns = None
        self.total_size = None
        self.targeted_value = None
        self.results_df = None
        self.columns_to_mask = []
        self.diffs_over_time = None
        self.ranks_over_time = None
        self.pareto_selection_over_time = []
        self.no_changes_in_ranks_cnt = 0
        self.all_below_diff_threshold_cnt = 0
        self.last_mi = None
        self.last_entropy = None
        self.cur_coverage = 0.0
        self._invalid_trials = {}
        self.log = logging.getLogger("InformationGain")

    # TODO: I don't know why init isn't accepting parameters...
    def set_config(
        self,
        mi_diff_limit: float,
        samples_below_limit: int,
        consider_pareto_front_convergence: bool,
    ) -> None:
        self.mi_diff_limit = mi_diff_limit
        self.samples_below_limit = samples_below_limit
        self.consider_pareto_front_convergence = consider_pareto_front_convergence
        self.log.debug(
            f"[InformationGainStopper] configured: mi_diff_limit {self.mi_diff_limit}; "
            f"samples_below_limit: {self.samples_below_limit}; "
            f"consider pareto front instead rank changes: {self.consider_pareto_front_convergence}."
        )

    def configure_details(
        self,
        data_columns: list[str],
        targeted_value: str,
        min_samples: int | str = "auto",
        search_columns: list[str] | None = None,
        total_size: int | str = "N/A",
    ) -> None:
        self.data_columns = data_columns
        # self.targeted_value = 'values__' + targeted_value
        self.targeted_value = targeted_value
        if min_samples == "auto":
            if not search_columns:
                raise ValueError("search_columns cannot be None")

            # self.min_samples = 2*len(data_columns)
            self.min_samples = 2 * len(search_columns)
        else:
            self.min_samples = min_samples
        self.search_columns = search_columns
        self.total_size = total_size
        self.should_stop = False
        self.results_df = None
        self.columns_to_mask = []
        self.diffs_over_time = None
        self.ranks_over_time = None
        self.pareto_selection_over_time = []
        self.no_changes_in_ranks_cnt = 0
        self.all_below_diff_threshold_cnt = 0
        self.last_mi = None
        self.last_entropy = None
        self.log.debug(
            f"[InformationGainStopper] configured: mi_diff_limit {self.mi_diff_limit}; "
            f"samples_below_limit: {self.samples_below_limit}; min_samples {self.min_samples}; "
            f"data_columns: {self.data_columns}; targeted_value: {self.targeted_value}."
        )

    def __str__(self) -> str:
        return (
            f"InformationGainStopper({self.mi_diff_limit}, {self.samples_below_limit})"
        )

    def _usable_sample_count(self) -> int:
        """Return trials counted toward min_samples, including failed-trial credit."""
        return self.trials_num + self.failed_trials_to_consider_num

    def _in_grace_period(self) -> bool:
        """Return True if fewer than min_samples usable trials have been seen."""
        return (
            self.min_samples is not None
            and self._usable_sample_count() < self.min_samples
        )

    def _ranking_dataframe(self) -> "pd.DataFrame | None":
        """Build the ranked-dimension table from the last MI calculation.

        Returns:
            A pandas DataFrame sorted by rank, or None if MI has not been computed.
        """
        if not self.ranks_over_time or not self.last_mi:
            return None
        import pandas as pd

        ranks = [(k, self.ranks_over_time[k][-1]) for k in self.ranks_over_time]
        entropy = self.last_entropy or float("nan")
        rows = [
            [
                dim,
                value,
                self.last_mi[dim],
                self.last_mi[dim] / entropy if entropy else float("nan"),
            ]
            for dim, value in ranks
        ]
        df = pd.DataFrame(
            columns=["dimension", "rank", "mi", "uncertainty%"], data=rows
        )
        return df.sort_values("rank", ascending=True)

    def progress_message(self) -> str:
        """Return a one-line description of InformationGainStopper progress."""
        if self.should_stop:
            return f"InformationGainStopper: stopping after {self.trials_num} samples."
        if self._in_grace_period():
            return (
                f"InformationGainStopper: not stopping: "
                f"{self._usable_sample_count()}/{self.min_samples} samples in grace period."
            )
        return (
            f"InformationGainStopper: not stopping after {self.trials_num} samples. "
            f"MI-change below {self.mi_diff_limit} for "
            f"{self.all_below_diff_threshold_cnt}/{self.samples_below_limit} "
            f"consecutive samples; no rank/pareto change for "
            f"{self.no_changes_in_ranks_cnt}/{self.samples_below_limit}."
        )

    def final_report(self) -> str | None:
        """Return the ranked-dimension table for stdout, or None in the grace period."""
        if self._in_grace_period() and not self.should_stop:
            return None
        df = self._ranking_dataframe()
        if self.should_stop:
            header = f"Stopping criteria reached after {self.trials_num} samples."
        else:
            header = (
                f"Stopping criteria were not reached after {self.trials_num} samples. "
                f"Last known ranking:"
            )
        lines = [
            header,
            f"Total search space size is {self.total_size}, search coverage is {self.cur_coverage}.",
        ]
        if self.last_entropy is not None:
            lines.append(
                f"Entropy of target variable clusters: {self.last_entropy} nats."
            )
        if df is not None:
            lines.append(f"Result:\n{df}")
        if self.pareto_selection_over_time:
            lines.append(f"Pareto selection:{self.pareto_selection_over_time[-1]}")
        if df is None and not self.should_stop:
            return None
        return "\n".join(lines)

    def __call__(self, trial_id: str, result: dict[str, Any]) -> bool:
        import pandas as pd

        from .space_analysis import (
            convert_values_of_dict,
            mi_diff_over_time,
        )

        if self.should_stop:
            return True

        if trial_id in self.seen_trial_ids:
            # it is always called twice per trial...?
            self.log.debug(f"[InformationGainStopper] already seen this {trial_id}...")
            return False
        self.seen_trial_ids.append(trial_id)
        self.trials_num += 1

        result.update(result["config"])
        del result["config"]

        # first time setup
        if self.results_df is None:
            df_columns = list(result.keys())
            # del df_columns[df_columns.index('config')]
            # df_columns.extend(list(result['config'].keys()))
            self.results_df = pd.DataFrame([], columns=df_columns)
            self.results_df.set_index("trial_id")

        if result.get("checkpoint_dir_name", "default") != "default":
            result.pop("checkpoint_dir_name")

        try:
            results_casted, non_numeric_cols = convert_values_of_dict(result)
        except ValueError:
            self.log.debug(f"unable to parse {result}. Ignoring...")
            self.trials_num -= 1
            self.failed_trials_num += 1
            self._invalid_trials[trial_id] = result
            if self.consider_invalid_experiment_factor is not None:
                self.failed_trials_to_consider_num = int(
                    max(
                        np.floor(
                            self.failed_trials_num
                            / self.consider_invalid_experiment_factor
                        ),
                        0,
                    )
                )
            else:
                return False
        else:
            diff_cols = list(set(non_numeric_cols) - set(self.columns_to_mask))
            self.columns_to_mask.extend(diff_cols)  # if any
            # store
            self.log.debug(f"storing {results_casted} for {trial_id}.")
            self.results_df.loc[trial_id] = results_casted
        # and process
        if self._in_grace_period():
            self.log.info(self.progress_message())
            return False
        if self._usable_sample_count() == self.last_calculation_step_num:
            # will only trigger on added failed experiments
            self.log.debug("No new data, skipping...")
            return False
        try:
            (
                mi_output,
                all_below_threshold,
                change_in_ranks,
                new_diffs,
                new_ranks,
                new_pareto,
            ) = mi_diff_over_time(
                self.results_df,
                self.data_columns,
                self.columns_to_mask,
                [],
                0,
                self.targeted_value,
                self.diffs_over_time,
                self.last_mi,
                self.mi_diff_limit,
                self.ranks_over_time,
                self.pareto_selection_over_time,
                self.consider_pareto_front_convergence,
            )
        except ValueError as e:
            self.log.warning(
                f"Unable to calculate MI due to {e}. Unable to continue. Stopping..."
            )
            self.should_stop = True
            return True

        new_mi = mi_output.mutual_information
        self.last_calculation_step_num = self._usable_sample_count()
        self.last_mi = new_mi
        self.last_entropy = mi_output.entropy
        self.diffs_over_time = new_diffs
        self.ranks_over_time = new_ranks
        self.pareto_selection_over_time = new_pareto
        self.cur_coverage = self.trials_num / self.total_size
        if all_below_threshold:
            self.all_below_diff_threshold_cnt += 1
        else:
            self.all_below_diff_threshold_cnt = 0
        if not change_in_ranks:
            self.no_changes_in_ranks_cnt += 1
        else:
            self.no_changes_in_ranks_cnt = 0
        if (
            self.no_changes_in_ranks_cnt >= self.samples_below_limit
            and self.all_below_diff_threshold_cnt >= self.samples_below_limit
        ):
            self.should_stop = True
            return True
        self.log.info(self.progress_message())
        return False

    def stop_all(self) -> bool:
        return self.should_stop


class BayesianMetricDifferenceStopper(AdoStopperMixin, ray.tune.Stopper):
    """
    Stopper that uses Bayesian sequential analysis to detect with high confidence
    which side of a threshold the mean difference between two metrics lies on.

    Uses a Bayesian t-posterior with Jeffreys prior for the difference between metrics.
    Stops when we're confident (e.g., 95%) that |A-B| is either above OR below the threshold.

    The stopper is agnostic to interpretation - it simply reports which side with confidence.
    Users interpret the result based on their context (improvement detection, convergence, etc).
    """

    def __init__(self) -> None:
        self.metric_a = None
        self.metric_b = None
        self.threshold = None
        self.target_probability = None
        self.min_samples = 10
        self.differences = []
        self.should_stop = False
        self.stop_reason = None  # "exceeds_threshold" or "within_threshold"
        self.stop_probability = None  # The actual probability when stopped
        self.seen_trial_ids = []
        self.trials_num = 0
        self._last_prob_abs_greater = None
        self.log = logging.getLogger("BayesianMetricDifferenceStopper")

    def set_config(
        self,
        metric_a: str,
        metric_b: str,
        threshold: float,
        target_probability: float = 0.95,
        min_samples: int = 10,
    ) -> None:
        """
        Configure the stopper.

        Args:
            metric_a: Name of the first metric to compare
            metric_b: Name of the second metric to compare
            threshold: Threshold value for |A-B|
            target_probability: Probability threshold for stopping (default: 0.95)
                Stop when we're this confident the difference is above OR below threshold
            min_samples: Minimum number of samples before applying stopping criteria (default: 10)
        """
        self.metric_a = metric_a
        self.metric_b = metric_b
        self.threshold = abs(threshold)  # Ensure threshold is positive
        self.target_probability = target_probability
        self.min_samples = int(min_samples)
        self.differences = []
        self.should_stop = False
        self.stop_reason = None
        self.stop_probability = None
        self.seen_trial_ids = []
        self.trials_num = 0
        self._last_prob_abs_greater = None

        # Validation
        if not 0 < target_probability < 1:
            raise ValueError(
                f"target_probability must be between 0 and 1, got {target_probability}"
            )

        self.log.info(
            f"Configured BayesianMetricDifferenceStopper:\n"
            f"  Metrics: |{metric_a} - {metric_b}|\n"
            f"  Threshold: {self.threshold}\n"
            f"  Target Probability: {target_probability}\n"
            f"  Min Samples: {min_samples}\n"
            f"  Stopping: When {target_probability * 100:.0f}% confident difference is "
            f"above OR below threshold"
        )

    def __str__(self) -> str:
        return (
            f"BayesianMetricDifferenceStopper("
            f"|{self.metric_a}-{self.metric_b}| vs {self.threshold}, "
            f"P={self.target_probability}, min_n={self.min_samples})"
        )

    def progress_message(self) -> str:
        """Return a one-line description of BayesianMetricDifferenceStopper progress."""
        if self.should_stop:
            return (
                f"BayesianMetricDifferenceStopper: stopping after {self.trials_num} "
                f"trials ({self.stop_reason})."
            )
        n_differences = len(self.differences)
        if n_differences < self.min_samples:
            return (
                f"BayesianMetricDifferenceStopper: not stopping: "
                f"{n_differences}/{self.min_samples} usable samples in grace period."
            )
        if self._last_prob_abs_greater is None:
            return (
                f"BayesianMetricDifferenceStopper: not stopping after "
                f"{self.trials_num} trials."
            )
        return (
            f"BayesianMetricDifferenceStopper: not stopping after {self.trials_num} "
            f"trials. Last P(|{self.metric_a}-{self.metric_b}| > {self.threshold}) = "
            f"{self._last_prob_abs_greater:.4f}."
        )

    def _compute_bayesian_t_probability(
        self, differences: list, threshold: float
    ) -> dict:
        """
        Compute Bayesian t-posterior probability for the difference.

        Uses Jeffreys prior: p(μ, σ²) ∝ 1/σ²
        The posterior for μ is: μ | data ~ t(n-1, x̄, s/√n)

        Args:
            differences: List of observed differences (A-B)
            threshold: Threshold value to test against

        Returns:
            Dictionary with probabilities and statistics
        """
        import scipy.stats as stats

        n = len(differences)
        if n < 2:
            return {
                "n": n,
                "mean": np.nan if n == 0 else differences[0],
                "std": np.nan,
                "se": np.nan,
                "prob_greater_than_threshold": 0.0,
                "prob_less_than_neg_threshold": 0.0,
                "prob_abs_greater_than_threshold": 0.0,
            }

        # Compute sample statistics
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)  # Sample std deviation

        # Avoid division by zero
        if std_diff < 1e-10:
            # If std is essentially zero, use deterministic decision
            if abs(mean_diff) > threshold:
                prob_greater = 1.0 if mean_diff > threshold else 0.0
                prob_less = 1.0 if mean_diff < -threshold else 0.0
            else:
                prob_greater = 0.0
                prob_less = 0.0
        else:
            # Standard error of the mean
            se = std_diff / np.sqrt(n)

            # Degrees of freedom for t-distribution
            df = n - 1

            # Compute P(difference > threshold) using t-distribution
            # t = (threshold - mean_diff) / se
            # P(diff > threshold) = P(T > t) where T ~ t(df)
            t_stat_upper = (threshold - mean_diff) / se
            prob_greater = 1.0 - stats.t.cdf(t_stat_upper, df)

            # Compute P(difference < -threshold)
            # P(diff < -threshold) = P(T < t) where t = (-threshold - mean_diff) / se
            t_stat_lower = (-threshold - mean_diff) / se
            prob_less = stats.t.cdf(t_stat_lower, df)

        # Total probability P(|difference| > threshold)
        prob_abs_greater = prob_greater + prob_less

        return {
            "n": n,
            "mean": mean_diff,
            "std": std_diff,
            "se": std_diff / np.sqrt(n),
            "prob_greater_than_threshold": prob_greater,
            "prob_less_than_neg_threshold": prob_less,
            "prob_abs_greater_than_threshold": prob_abs_greater,
        }

    def __call__(self, trial_id: str, result: dict[str, Any]) -> bool:
        """
        Check if stopping criteria is met for this trial.

        Args:
            trial_id: Unique identifier for the trial
            result: Dictionary containing trial results including metrics

        Returns:
            True if stopping criteria is met, False otherwise
        """
        if self.should_stop:
            return True

        if trial_id in self.seen_trial_ids:
            self.log.debug(f"Already seen trial {trial_id}, skipping...")
            return False

        self.seen_trial_ids.append(trial_id)
        self.trials_num += 1

        # Extract metrics
        metric_a_value = result.get(self.metric_a)
        metric_b_value = result.get(self.metric_b)

        # Check if both metrics are available
        if metric_a_value is None:
            self.log.warning(
                f"Metric '{self.metric_a}' not found in trial {trial_id} results"
            )
            return False

        if metric_b_value is None:
            self.log.warning(
                f"Metric '{self.metric_b}' not found in trial {trial_id} results"
            )
            return False

        # Check for NaN values
        if np.isnan(metric_a_value) or np.isnan(metric_b_value):
            self.log.debug(f"Trial {trial_id} has NaN metric values, skipping...")
            return False

        # Compute difference A - B
        difference = metric_a_value - metric_b_value
        self.differences.append(difference)

        self.log.debug(
            f"Trial {trial_id}: {self.metric_a}={metric_a_value:.4f}, "
            f"{self.metric_b}={metric_b_value:.4f}, difference={difference:.4f}"
        )

        # Compute Bayesian posterior probabilities
        stats_result = self._compute_bayesian_t_probability(
            self.differences, self.threshold
        )

        prob_abs_greater = stats_result["prob_abs_greater_than_threshold"]
        mean_diff = stats_result["mean"]
        self._last_prob_abs_greater = prob_abs_greater

        # Check if we have enough usable samples (differences collected)
        n_differences = len(self.differences)
        if n_differences < self.min_samples:
            self.log.info(self.progress_message())
            return False

        # Apply stopping criterion - check if confident about EITHER side
        # Stop when we're target_probability confident difference is above OR below threshold
        prob_abs_less = 1.0 - prob_abs_greater  # P(|diff| ≤ threshold)

        # Check if difference significantly EXCEEDS threshold
        if prob_abs_greater >= self.target_probability:
            self.should_stop = True
            self.stop_reason = "exceeds_threshold"
            self.stop_probability = prob_abs_greater

            self.log.info(
                f"  Stopping after {self.trials_num} trials  - usable differences collected {len(self.differences)} \n"
                f"  {prob_abs_greater * 100:.1f}% confident mean difference is ABOVE threshold\n"
                f"  Mean difference: {mean_diff:.4f}\n"
                f"  Standard error: ±{stats_result['se']:.4f}\n"
                f"  Threshold: {self.threshold}\n"
                f"  P(|{self.metric_a} - {self.metric_b}| > {self.threshold}) = {prob_abs_greater:.4f}"
            )
            return True

        # Check if difference is confidently WITHIN threshold
        if prob_abs_less >= self.target_probability:
            self.should_stop = True
            self.stop_reason = "within_threshold"
            self.stop_probability = prob_abs_less

            self.log.info(
                f"  Stopping after {self.trials_num} trials\n"
                f"  {prob_abs_less * 100:.1f}% confident mean difference is BELOW threshold\n"
                f"  Mean difference: {mean_diff:.4f}\n"
                f"  Standard error: ±{stats_result['se']:.4f}\n"
                f"  Threshold: {self.threshold}\n"
                f"  P(|{self.metric_a} - {self.metric_b}| < {self.threshold}) = {prob_abs_less:.4f}"
            )
            return True

        self.log.info(self.progress_message())
        return False

    def stop_all(self) -> bool:
        """Check if all trials should be stopped."""
        return self.should_stop
