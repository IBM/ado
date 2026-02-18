# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


def stopping_bool_from_ratios(
    mean_ratio: float,
    std_ratio: float,
    mean_ratio_threshold: float = 0.9,
    std_ratio_threshold: float = 0.75,
) -> bool:
    """
    Determine whether sampling should stop based on mean and standard deviation ratios.

    The function evaluates whether the mean ratio lies within a symmetric threshold
    range around 1, and whether the standard deviation ratio is below its threshold.
    It returns a boolean indicating if all conditions are satisfied.

    Parameters
    ----------
    mean_ratio : float
        Ratio of the current mean compared to a reference mean.
    std_ratio : float
        Ratio of the current standard deviation compared to a reference standard deviation.
    mean_ratio_threshold : float, optional
        Lower bound threshold for the mean ratio.
        The upper bound is taken as the reciprocal (1 / mean_ratio_threshold).
    std_ratio_threshold : float, optional
        Upper bound threshold for the standard deviation ratio.

    Returns
    -------
    bool
        True if mean_ratio is greater than `mean_ratio_threshold` and less than
        `1 / mean_ratio_threshold`, and std_ratio is less than `1 / std_ratio_threshold`.
        False otherwise.

    Notes
    -----
    This logic works for both maximum- and minimum-based metrics, ensuring
    ratios remain within acceptable bounds before stopping.
    """
    return (
        (mean_ratio > mean_ratio_threshold)
        and (mean_ratio < 1 / mean_ratio_threshold)
        and (std_ratio < 1 / std_ratio_threshold)
    )
