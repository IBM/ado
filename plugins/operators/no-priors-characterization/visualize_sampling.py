# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""
Visualization script for comparing sampling strategies.

This script demonstrates the distribution patterns of different sampling
strategies (random, CLHS, Sobol) in a 2D grid space.
"""

import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.axes import Axes
except ModuleNotFoundError:
    print("matplotlib not found. Please install it to run the visualization.")
    print("pip install matplotlib")
    sys.exit(1)

from no_priors_characterization.utils.high_dimensional_sampling import (
    concatenated_latin_hypercube_sampling,
    random_high_dimensional_sampling,
    sobol_sampling,
)


def plot_grid(
    ax: Axes,
    dimensions: list[int] | tuple[int, int],
    points: np.ndarray | list[list[int]],
    title: str,
) -> None:
    """
    Plot a 2D grid visualization of sampled points with overlap detection.

    Args:
        ax: Matplotlib axes object to draw on.
        dimensions: Dimensions of the grid [width, height].
        points: List of sampled points as [x, y] coordinates.
        title: Title for the plot.
    """
    from collections import defaultdict

    import matplotlib.patches as patches

    nx, ny = dimensions[0], dimensions[1]

    # Setup grid
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xticks(range(nx + 1))
    ax.set_yticks(range(ny + 1))
    ax.grid(True, color="black", linewidth=1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, pad=10)

    # Track points in each cell to handle overlaps
    # Maps (x, y) -> list of time indices (1-based)
    grid_content = defaultdict(list)

    # points is a list of [x, y], enumerate gives us the time index (0-based)
    for time, point in enumerate(points):
        x, y = int(point[0]), int(point[1])  # Ensure integers
        if 0 <= x < nx and 0 <= y < ny:
            # Store t + 1 so the first sample is '1'
            grid_content[(x, y)].append(time + 1)

    # Draw squares and text
    for (x, y), indices in grid_content.items():
        count = len(indices)
        # Darker alpha if multiple points hit the same square
        alpha = min(0.4 + 0.2 * count, 1.0)
        rect = patches.Rectangle(
            (x, y), 1, 1, linewidth=0, facecolor="#ff0000", alpha=alpha
        )
        ax.add_patch(rect)

        # Label is the comma-separated list of indices
        label = ",".join(map(str, indices))

        # Add text with shadow effect
        ax.text(
            x + 0.52,
            y + 0.52,
            label,
            ha="center",
            va="center",
            color="#D4FF00",
            fontweight="bold",
        )
        ax.text(
            x + 0.5,
            y + 0.5,
            label,
            ha="center",
            va="center",
            color="#000000",
            fontweight="bold",
        )


def main() -> None:
    """Run the sampling visualization comparison."""
    # Configuration
    dimensions = [20, 6]  # 20 columns, 6 rows (Total 120 cells)
    N = 30  # Number of samples to draw
    SEED = 42

    # Plotting
    _fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Random Sampling
    pts_rnd = random_high_dimensional_sampling(dimensions, N, seed=SEED)
    plot_grid(axes[0], dimensions, pts_rnd, f"Random Sampling (N={N})\n(Clumps & Gaps)")

    # 2. Concatenated LHS
    pts_lhs = concatenated_latin_hypercube_sampling(dimensions, N, seed=SEED)
    plot_grid(
        axes[1], dimensions, pts_lhs, f"Concatenated LHS (N={N})\n(Uniform Rows/Cols)"
    )

    # 3. Sobol Sequence
    pts_sobol = sobol_sampling(dimensions, N, seed=SEED)
    plot_grid(
        axes[2], dimensions, pts_sobol, f"Sobol Sequence (N={N})\n(Maximal Spreading)"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
