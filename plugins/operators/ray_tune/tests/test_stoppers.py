# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for ado Ray Tune stopper status reporting."""

from ado_ray_tune.stoppers import (
    SAMPLING_BUDGET_HALT_REASON,
    BayesianMetricDifferenceStopper,
    InformationGainStopper,
    MaxSamplesStopper,
    SimpleStopper,
    format_stopper_run_report,
    iter_stoppers,
    report_stoppers_after_fit,
)
from ray.tune.stopper import CombinedStopper, TimeoutStopper


def _information_gain_stopper(
    min_samples: int = 4,
    samples_below_limit: int = 5,
) -> InformationGainStopper:
    """Return a configured InformationGainStopper for unit tests."""
    stopper = InformationGainStopper()
    stopper.set_config(
        mi_diff_limit=1.0,
        samples_below_limit=samples_below_limit,
        consider_pareto_front_convergence=True,
    )
    stopper.configure_details(
        data_columns=["nodes", "cpu_family"],
        targeted_value="wallClockRuntime",
        min_samples=min_samples,
        search_columns=["nodes", "cpu_family"],
        total_size=48,
    )
    return stopper


def _with_ranking(stopper: InformationGainStopper) -> InformationGainStopper:
    """Attach a last-known MI ranking so final_report() can build a table."""
    stopper.trials_num = 14
    stopper.last_mi = {"nodes": 0.491089, "cpu_family": 0.352622}
    stopper.last_entropy = 0.6517565611726529
    stopper.ranks_over_time = {"nodes": [1], "cpu_family": [2]}
    stopper.pareto_selection_over_time = [["cpu_family", "nodes"]]
    stopper.cur_coverage = 14 / 48
    stopper.all_below_diff_threshold_cnt = 2
    stopper.no_changes_in_ranks_cnt = 1
    return stopper


def _ig_trial_result(trial_id: str, nodes: int, runtime: float) -> dict:
    """Build a trial result dict as Ray Tune would pass to a stopper."""
    return {
        "trial_id": trial_id,
        "config": {"nodes": nodes, "cpu_family": 1},
        "wallClockRuntime": runtime,
        "checkpoint_dir_name": "default",
    }


class TestIterStoppers:
    """Tests for unwrapping CombinedStopper."""

    def test_none_returns_empty(self) -> None:
        """None yields no stoppers."""
        assert iter_stoppers(None) == []

    def test_single_stopper(self) -> None:
        """A single stopper is returned as a one-element list."""
        stopper = MaxSamplesStopper()
        stopper.set_config(max_samples=10)
        assert iter_stoppers(stopper) == [stopper]

    def test_combined_stopper_unwraps_children(self) -> None:
        """CombinedStopper children are returned in order."""
        first = MaxSamplesStopper()
        first.set_config(max_samples=10)
        second = SimpleStopper()
        second.set_config(mode="min", metric="loss")
        combined = CombinedStopper(first, second)
        assert iter_stoppers(combined) == [first, second]

    def test_nested_combined_stopper(self) -> None:
        """Nested CombinedStopper (e.g. TimeoutStopper wrap) is flattened."""
        ado_stopper = MaxSamplesStopper()
        ado_stopper.set_config(max_samples=10)
        timeout = TimeoutStopper(timeout=30)
        nested = CombinedStopper(CombinedStopper(ado_stopper), timeout)
        unwrapped = iter_stoppers(nested)
        assert unwrapped == [ado_stopper, timeout]


class TestInformationGainStopperStatus:
    """Tests for InformationGainStopper progress and final reports."""

    def test_grace_period_progress_has_no_table(self) -> None:
        """Before min_samples, progress mentions grace and final_report has no table."""
        stopper = _information_gain_stopper(min_samples=4)
        stopped = stopper("trial-1", _ig_trial_result("trial-1", nodes=1, runtime=10.0))

        assert stopped is False
        assert stopper.did_trigger() is False
        message = stopper.progress_message()
        assert "not stopping" in message
        assert "grace period" in message
        assert "1/4" in message
        assert stopper.final_report() is None

    def test_not_stopping_report_includes_ranking_table(self) -> None:
        """When MI has been computed but criteria are not met, report the last ranking."""
        stopper = _with_ranking(_information_gain_stopper())
        report = stopper.final_report()

        assert stopper.did_trigger() is False
        assert report is not None
        assert "were not reached" in report
        assert "Last known ranking" in report
        assert "nodes" in report
        assert "cpu_family" in report
        assert "Pareto selection" in report
        progress = stopper.progress_message()
        assert "not stopping after 14 samples" in progress
        assert "2/5" in progress

    def test_stopping_report_includes_ranking_table(self) -> None:
        """When criteria are met, did_trigger is True and the report says so."""
        stopper = _with_ranking(_information_gain_stopper())
        stopper.should_stop = True
        stopper.all_below_diff_threshold_cnt = 5
        stopper.no_changes_in_ranks_cnt = 5
        report = stopper.final_report()

        assert stopper.did_trigger() is True
        assert report is not None
        assert "Stopping criteria reached after 14 samples" in report
        assert "nodes" in report
        assert "Pareto selection" in report


class TestReportStoppersAfterFit:
    """Tests for the end-of-run halt summary."""

    def test_no_stoppers_is_sampling_budget(self) -> None:
        """With no stopper configured, halt because the sampling budget was reached."""
        report = report_stoppers_after_fit(None, num_trials=10, num_samples=10)
        assert report.halt_reason == SAMPLING_BUDGET_HALT_REASON
        assert report.triggered == []
        assert report.not_triggered == []
        assert report.stopper_logs() == {}
        assert format_stopper_run_report(report) == SAMPLING_BUDGET_HALT_REASON

    def test_information_gain_did_not_fire_uses_budget_and_last_status(
        self,
    ) -> None:
        """When InformationGain did not fire, report budget halt and its last log."""
        stopper = _with_ranking(_information_gain_stopper())
        report = report_stoppers_after_fit(stopper, num_trials=32, num_samples=32)

        assert report.halt_reason == SAMPLING_BUDGET_HALT_REASON
        assert report.triggered == []
        assert len(report.not_triggered) == 1
        assert report.not_triggered[0].name == "InformationGainStopper"
        assert report.not_triggered[0].log is not None
        assert "Last known ranking" in report.not_triggered[0].log
        assert report.stopper_logs() == {
            "InformationGainStopper": report.not_triggered[0].log
        }

        formatted = format_stopper_run_report(report)
        assert formatted.startswith(SAMPLING_BUDGET_HALT_REASON)
        assert (
            "The following stoppers did not fire. InformationGainStopper." in formatted
        )
        assert "The final status of the stopper was" in formatted
        assert "Last known ranking" in formatted

    def test_information_gain_fired_includes_triggered_log(self) -> None:
        """When InformationGain fired, halt names it and prints its log."""
        stopper = _with_ranking(_information_gain_stopper())
        stopper.should_stop = True
        report = report_stoppers_after_fit(stopper, num_trials=14, num_samples=32)

        assert (
            report.halt_reason
            == "RayTune operation stopped because of stopper InformationGainStopper."
        )
        assert len(report.triggered) == 1
        assert report.triggered[0].name == "InformationGainStopper"
        assert report.triggered[0].log is not None
        assert "Stopping criteria reached after 14 samples" in report.triggered[0].log
        assert report.not_triggered == []
        assert report.stopper_logs() == {
            "InformationGainStopper": report.triggered[0].log
        }

        formatted = format_stopper_run_report(report)
        assert formatted.startswith(
            "RayTune operation stopped because of stopper InformationGainStopper."
        )
        assert "Stopping criteria reached after 14 samples" in formatted
        assert "The following stoppers did not fire" not in formatted

    def test_other_stopper_triggered_reports_information_gain_did_not_fire(
        self,
    ) -> None:
        """If MaxSamples ended the run, InformationGain is listed as not fired."""
        max_samples = MaxSamplesStopper()
        max_samples.set_config(max_samples=10)
        max_samples.should_stop = True
        information_gain = _with_ranking(_information_gain_stopper())
        combined = CombinedStopper(max_samples, information_gain)

        report = report_stoppers_after_fit(combined, num_trials=10, num_samples=100)

        assert (
            report.halt_reason
            == "RayTune operation stopped because of stopper MaxSamplesStopper."
        )
        assert [status.name for status in report.triggered] == ["MaxSamplesStopper"]
        assert report.triggered[0].log is None
        assert [status.name for status in report.not_triggered] == [
            "InformationGainStopper"
        ]
        assert report.not_triggered[0].log is not None
        assert "Last known ranking" in report.not_triggered[0].log
        assert report.stopper_logs() == {
            "InformationGainStopper": report.not_triggered[0].log
        }

        formatted = format_stopper_run_report(report)
        assert formatted.startswith(
            "RayTune operation stopped because of stopper MaxSamplesStopper."
        )
        assert (
            "The following stoppers did not fire. InformationGainStopper." in formatted
        )
        assert "The final status of the stopper was" in formatted
        assert "Last known ranking" in formatted


class TestSimpleAndBayesianStopperStatus:
    """Tests for status methods on other ado stoppers."""

    def test_simple_stopper_grace_and_trigger(self) -> None:
        """SimpleStopper reports grace progress and did_trigger after stopping."""
        stopper = SimpleStopper()
        stopper.set_config(
            mode="min",
            metric="loss",
            min_trials=1,
            buffer_states=0,
            stop_on_repeat=False,
        )

        first = stopper("t1", {"config": {"x": 1}, "loss": 1.0})
        assert first is False
        assert "grace period" in stopper.progress_message()
        assert stopper.did_trigger() is False

        second = stopper("t2", {"config": {"x": 2}, "loss": 2.0})
        assert second is True
        assert stopper.did_trigger() is True
        assert "stopping" in stopper.progress_message()

    def test_bayesian_stopper_progress_message(self) -> None:
        """BayesianMetricDifferenceStopper exposes did_trigger and progress."""
        stopper = BayesianMetricDifferenceStopper()
        stopper.set_config(
            metric_a="a",
            metric_b="b",
            threshold=1.0,
            min_samples=10,
        )
        assert stopper.did_trigger() is False
        message = stopper.progress_message()
        assert message
        assert "BayesianMetricDifferenceStopper" in message
        assert "not stopping" in message
