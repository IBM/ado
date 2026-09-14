# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Tests for ensure_dataset in ml_classifier."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import HfHubHTTPError

from autoconf.utils.autoconf_build.ml_classifier import (
    DatasetDownloadError,
    ensure_dataset,
)

_REPO_ID = "ibm-research/LLMFineTuningBench"
_FILENAME = "ado-sfttrainer-v1-0-0.csv"


def test_existing_file_returned_without_download(tmp_path: Path) -> None:
    """When data_path already exists, hf_hub_download is never called."""
    dest = tmp_path / "dataset.csv"
    dest.write_text("col\n1\n")
    with patch("huggingface_hub.hf_hub_download") as mock_dl:
        assert ensure_dataset(_REPO_ID, _FILENAME, dest) == dest
    mock_dl.assert_not_called()


def test_successful_download_copies_file_and_creates_parents(tmp_path: Path) -> None:
    """hf_hub_download result is copied to data_path, creating missing parent dirs."""
    dest = tmp_path / "deep" / "nested" / "dataset.csv"
    cached = tmp_path / "cached.csv"
    cached.write_text("col\n1\n")
    with patch("huggingface_hub.hf_hub_download", return_value=str(cached)):
        assert ensure_dataset(_REPO_ID, _FILENAME, dest) == dest
    assert dest.is_file()
    assert dest.read_text() == "col\n1\n"


@pytest.mark.parametrize(
    "error",
    [
        HfHubHTTPError("403 Forbidden", response=MagicMock(status_code=403)),
        OSError("Disk full"),
    ],
)
def test_download_errors_raise_dataset_download_error(
    tmp_path: Path, error: Exception
) -> None:
    """HfHubHTTPError and OSError from hf_hub_download become DatasetDownloadError."""
    with (
        patch("huggingface_hub.hf_hub_download", side_effect=error),
        pytest.raises(DatasetDownloadError, match="Could not download"),
    ):
        ensure_dataset(_REPO_ID, _FILENAME, tmp_path / "dataset.csv")
