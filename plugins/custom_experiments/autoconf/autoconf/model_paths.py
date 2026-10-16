# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

from pathlib import Path

MODEL_VERSION = "4.0.0"
MODEL_DIRECTORY = "v4-0-0"
# Anchor to the package location so this resolves correctly regardless of cwd.
# Equivalent to: <repo>/plugins/custom_experiments/autoconf/autoconf/models/
_PACKAGE_ROOT = Path(__file__).parent
DEFAULT_MODEL_ROOT = _PACKAGE_ROOT / "models"


def model_path(model_root: Path | None = None) -> Path:
    """Return the path of the locally generated AutoConf model.

    Args:
        model_root: Optional model root override.

    Returns:
        The directory containing model version 4.0.0.
    """
    return (model_root or DEFAULT_MODEL_ROOT) / MODEL_DIRECTORY
