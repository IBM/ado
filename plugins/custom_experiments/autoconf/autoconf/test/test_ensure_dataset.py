# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Tests for ensure_dataset in ml_classifier."""

from pathlib import Path
from unittest.mock import patch

import pytest

from autoconf.utils.autoconf_build.ml_classifier import (
    DatasetDownloadError,
    ensure_dataset,
)

_REPO_ID = "ibm-research/LLMFineTuningBench"
_FILENAME = "ado-sfttrainer-v1-0-0.csv"


class TestEnsureDatasetCached:
    """ensure_dataset skips download when the destination already exists."""

    def test_existing_file_returned_without_download(self, tmp_path: Path) -> None:
        """When data_path already exists, hf_hub_download is never called."""
        dest = tmp_path / "dataset.csv"
        dest.write_text("col\n1\n")

        with patch("huggingface_hub.hf_hub_download") as mock_dl:
            result = ensure_dataset(_REPO_ID, _FILENAME, dest)

        assert result == dest
        mock_dl.assert_not_called()


class TestEnsureDatasetSuccess:
    """ensure_dataset copies the cached file to data_path on success."""

    def test_successful_download_copies_file(self, tmp_path: Path) -> None:
        """hf_hub_download result is copied to data_path and path is returned."""
        dest = tmp_path / "data" / "dataset.csv"
        cached = tmp_path / "cached.csv"
        cached.write_text("col\n1\n")

        with patch(
            "huggingface_hub.hf_hub_download",
            return_value=str(cached),
        ):
            result = ensure_dataset(_REPO_ID, _FILENAME, dest)

        assert result == dest
        assert dest.is_file()
        assert dest.read_text() == "col\n1\n"

    def test_parent_directory_created(self, tmp_path: Path) -> None:
        """Missing parent directories are created before download."""
        dest = tmp_path / "deep" / "nested" / "dataset.csv"
        cached = tmp_path / "cached.csv"
        cached.write_text("x\n")

        with patch(
            "huggingface_hub.hf_hub_download",
            return_value=str(cached),
        ):
            ensure_dataset(_REPO_ID, _FILENAME, dest)

        assert dest.parent.is_dir()


class TestEnsureDatasetHTTPError:
    """ensure_dataset wraps HfHubHTTPError as DatasetDownloadError."""

    def test_http_error_raises_dataset_download_error(self, tmp_path: Path) -> None:
        """HfHubHTTPError from hf_hub_download becomes DatasetDownloadError."""
        from unittest.mock import MagicMock

        from huggingface_hub.errors import HfHubHTTPError

        dest = tmp_path / "dataset.csv"
        mock_response = MagicMock()
        mock_response.status_code = 403

        with (
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=HfHubHTTPError("403 Forbidden", response=mock_response),
            ),
            pytest.raises(DatasetDownloadError, match="Could not download"),
        ):
            ensure_dataset(_REPO_ID, _FILENAME, dest)


class TestEnsureDatasetOSError:
    """ensure_dataset wraps OSError as DatasetDownloadError."""

    def test_oserror_raises_dataset_download_error(self, tmp_path: Path) -> None:
        """OSError from hf_hub_download becomes DatasetDownloadError."""
        dest = tmp_path / "dataset.csv"

        with (
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=OSError("Disk full"),
            ),
            pytest.raises(DatasetDownloadError, match="Could not download"),
        ):
            ensure_dataset(_REPO_ID, _FILENAME, dest)
