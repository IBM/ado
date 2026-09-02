# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

from pathlib import Path

MODEL_VERSION = "4.0.0"
MODEL_DIRECTORY = "v4-0-0"
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parent / "models"


def model_path(model_root: Path | None = None) -> Path:
    """Return the path of the locally generated AutoConf model.

    Args:
        model_root: Optional model root override.

    Returns:
        The directory containing model version 4.0.0.
    """
    return (model_root or DEFAULT_MODEL_ROOT) / MODEL_DIRECTORY
