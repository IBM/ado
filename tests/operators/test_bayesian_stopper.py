# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import numpy as np
from ado_ray_tune.stoppers import BayesianMetricDifferenceStopper


def test_bayesian_stopper_difference_exceeds_threshold() -> None:
    """Test that stopper correctly detects when difference exceeds threshold."""

    stopper = BayesianMetricDifferenceStopper()
    stopper.set_config(
        metric_a="accuracy",
        metric_b="baseline_accuracy",
        threshold=0.05,
        target_probability=0.95,
        min_samples=10,
    )

    # Simulate trials where difference is ~0.10 (above threshold of 0.05)
    rng = np.random.default_rng(42)
    stopped = False

    for i in range(50):
        trial_id = f"trial_{i}"

        # Generate metrics with true difference of ~0.10 (above threshold of 0.05)
        baseline = 0.70 + rng.normal(0, 0.02)
        accuracy = baseline + 0.10 + rng.normal(0, 0.02)

        result = {
            "accuracy": accuracy,
            "baseline_accuracy": baseline,
            "trial_id": trial_id,
        }

        should_stop = stopper(trial_id, result)

        if should_stop:
            stopped = True
            break

    assert stopped, "Expected stopper to stop when difference exceeds threshold"
    assert stopper.stop_reason == "exceeds_threshold"
    assert stopper.stop_probability >= 0.95


def test_bayesian_stopper_difference_within_threshold() -> None:
    """Test that stopper correctly detects when difference is within threshold."""

    stopper = BayesianMetricDifferenceStopper()
    stopper.set_config(
        metric_a="accuracy",
        metric_b="baseline_accuracy",
        threshold=0.10,
        target_probability=0.95,
        min_samples=10,
    )

    # Simulate trials where difference is ~0.02 (below threshold of 0.10)
    rng = np.random.default_rng(123)

    for i in range(50):
        trial_id = f"trial_{i}"

        # Generate metrics with small difference ~0.02 (below threshold of 0.10)
        baseline = 0.70 + rng.normal(0, 0.03)
        accuracy = baseline + 0.02 + rng.normal(0, 0.03)

        result = {
            "accuracy": accuracy,
            "baseline_accuracy": baseline,
            "trial_id": trial_id,
        }

        should_stop = stopper(trial_id, result)

        if should_stop:
            break

    # NOW EXPECT IT TO STOP (was broken before - this is the fix!)
    assert should_stop, "Expected stopper to stop when confident difference < threshold"
    assert stopper.stop_reason == "within_threshold"
    assert stopper.stop_probability >= 0.95


def test_bayesian_stopper_convergence_case() -> None:
    """Test convergence detection use case (train vs val loss)."""

    stopper = BayesianMetricDifferenceStopper()
    stopper.set_config(
        metric_a="train_loss",
        metric_b="val_loss",
        threshold=0.10,
        target_probability=0.95,
        min_samples=10,
    )

    # Simulate converged metrics (small difference ~0.02)
    rng = np.random.default_rng(456)

    for i in range(50):
        trial_id = f"trial_{i}"

        # Generate metrics with small difference ~0.02 (well within threshold of 0.10)
        val_loss = 0.50 + rng.normal(0, 0.01)
        train_loss = val_loss + 0.02 + rng.normal(0, 0.01)

        result = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "trial_id": trial_id,
        }

        should_stop = stopper(trial_id, result)

        if should_stop:
            break

    assert should_stop, "Expected stopper to detect convergence"
    assert stopper.stop_reason == "within_threshold"  # Converged = within threshold


def test_bayesian_stopper_divergence_case() -> None:
    """Test divergence detection use case (train vs val loss diverging)."""

    stopper = BayesianMetricDifferenceStopper()
    stopper.set_config(
        metric_a="train_loss",
        metric_b="val_loss",
        threshold=0.10,
        target_probability=0.95,
        min_samples=10,
    )

    # Simulate diverging metrics (large difference ~0.50)
    rng = np.random.default_rng(789)

    for i in range(50):
        trial_id = f"trial_{i}"

        # Much lower train loss - indicates overfitting
        val_loss = 0.50 + rng.normal(0, 0.02)
        train_loss = val_loss - 0.50 + rng.normal(0, 0.02)

        result = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "trial_id": trial_id,
        }

        should_stop = stopper(trial_id, result)

        if should_stop:
            break

    assert should_stop, "Expected stopper to detect divergence"
    assert stopper.stop_reason == "exceeds_threshold"  # Diverged = exceeds threshold


def test_min_samples_requirement() -> None:
    """Test that stopper waits for minimum samples before applying criteria."""

    stopper = BayesianMetricDifferenceStopper()
    stopper.set_config(
        metric_a="metric_a",
        metric_b="metric_b",
        threshold=0.01,
        target_probability=0.95,
        min_samples=15,
    )

    should_stop = False
    for i in range(14):  # Just below min_samples
        trial_id = f"trial_{i}"
        result = {
            "metric_a": 1.0,
            "metric_b": 0.0,  # Massive difference
            "trial_id": trial_id,
        }

        should_stop = stopper(trial_id, result)

        if should_stop:
            break

    assert not should_stop, "Stopper should not trigger before min samples (15) reached"

    # Now on the 15th trial, it should trigger
    result = {"metric_a": 1.0, "metric_b": 0.0, "trial_id": "trial_15"}
    should_stop = stopper("trial_15", result)

    assert should_stop, "Stopper should  trigger when min samples (15) reached"


def test_min_samples_with_skipped_trials() -> None:
    """Test that skipped trials (missing/NaN metrics) don't count toward min_samples."""

    stopper = BayesianMetricDifferenceStopper()
    stopper.set_config(
        metric_a="metric_a",
        metric_b="metric_b",
        threshold=0.01,
        target_probability=0.95,
        min_samples=10,
    )

    print(f"\nStopper: {stopper}")
    print("\nSimulating 15 trials with 5 having missing/NaN metrics...")

    # Run 15 trials total, but 5 will be skipped
    i = 0
    for i in range(15):
        trial_id = f"trial_{i}"

        # Every 3rd trial has missing metric
        if i % 3 == 0:
            result = {
                "metric_a": 1.0,
                # metric_b is missing
                "trial_id": trial_id,
            }
        else:
            result = {
                "metric_a": 1.0,
                "metric_b": 0.0,
                "trial_id": trial_id,
            }

        should_stop = stopper(trial_id, result)

        # Should not stop until we have 10 usable samples
        if should_stop and len(stopper.differences) < 10:
            break

    assert len(stopper.differences) == 10, "Expected stopper to stop at 10 samples"
    assert i == 14, "Expected 15 trails to be run"
