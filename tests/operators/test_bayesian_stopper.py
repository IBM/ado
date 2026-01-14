# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import numpy as np
from ado_ray_tune.stoppers import BayesianMetricDifferenceStopper


def test_bayesian_stopper_mode_greater_difference_is_greater():
    """Test that stopper correctly detects a significant difference."""

    # Create stopper
    stopper = BayesianMetricDifferenceStopper()
    stopper.set_config(
        metric_a="accuracy",
        metric_b="baseline_accuracy",
        threshold=0.05,
        target_probability=0.95,
        mode="greater",
        min_samples=10,
    )

    # Simulate trials where metric_a is consistently higher than metric_b
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

    assert (
        stopped
    ), "Expected stopper to stop if mean difference 0.1 and threshold 0.05 within 50 samples"


def test_bayesian_stopper_mode_greater_difference_is_less():
    """Test that stopper does not trigger when difference is below threshold."""

    # Create stopper
    stopper = BayesianMetricDifferenceStopper()
    stopper.set_config(
        metric_a="accuracy",
        metric_b="baseline_accuracy",
        threshold=0.10,  # Larger threshold
        target_probability=0.95,
        mode="greater",
        min_samples=10,
    )

    # Simulate trials where difference is small (below threshold)
    rng = np.random.default_rng(123)
    stopped = False

    for i in range(30):
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

    assert (
        not stopped
    ), "Did not expect stopper to trigger for mean difference 0.02 and threshold 0.10 in 30 samples"


def test_bayesian_stopper_mode_less_difference_is_less():

    # Create stopper in 'less' mode
    stopper = BayesianMetricDifferenceStopper()
    stopper.set_config(
        metric_a="train_loss",
        metric_b="val_loss",
        threshold=0.10,  # Threshold of 0.10
        target_probability=0.95,  # Stop when P(|diff| ≤ 0.10) ≥ 0.95
        mode="less",
        min_samples=10,
    )

    # Simulating trials where difference is ~0.02 (well within threshold of 0.10)..."

    # Simulate trials where difference is small (metrics converged)
    rng = np.random.default_rng(456)
    stopped = False

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
            stopped = True
            break

    assert (
        stopped
    ), "Stopper in 'less' mode should trigger when difference is within threshold"


def test_min_samples_requirement():
    """Test that stopper waits for minimum samples before applying criteria."""

    # Create stopper with min_samples=15
    stopper = BayesianMetricDifferenceStopper()
    stopper.set_config(
        metric_a="metric_a",
        metric_b="metric_b",
        threshold=0.01,
        target_probability=0.95,
        mode="greater",
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


def test_min_samples_with_skipped_trials():
    """Test that skipped trials (missing/NaN metrics) don't count toward min_samples."""

    # Create stopper with min_samples=10
    stopper = BayesianMetricDifferenceStopper()
    stopper.set_config(
        metric_a="metric_a",
        metric_b="metric_b",
        threshold=0.01,
        target_probability=0.95,
        mode="greater",
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
