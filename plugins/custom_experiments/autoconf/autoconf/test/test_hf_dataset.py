# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT
# ruff: noqa: S101

"""Tests for autoconf.utils.autoconf_build.hf_dataset."""

import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from autoconf.utils.autoconf_build.hf_dataset import (
    DatasetDownloadError,
    HuggingFaceDatasetURL,
    _sanitise_url,
    ensure_dataset,
)

_VALID_URL = "https://huggingface.co/datasets/ibm-research/LLMFineTuningBench/resolve/main/ado-sfttrainer-dataset.csv"


class TestEnsureDatasetUrlUnreachable:
    """ensure_dataset raises DatasetDownloadError when the host is unreachable."""

    def test_url_unreachable_raises_dataset_download_error(
        self, tmp_path: Path
    ) -> None:
        """A URLError (e.g. no network / DNS failure) becomes DatasetDownloadError."""
        dest = tmp_path / "data" / "dataset.csv"
        with (
            patch(
                "autoconf.utils.autoconf_build.hf_dataset.urllib.request.urlretrieve",
                side_effect=urllib.error.URLError(
                    reason="[Errno -2] Name or service not known"
                ),
            ),
            pytest.raises(DatasetDownloadError, match="Check the network connection"),
        ):
            ensure_dataset(_VALID_URL, dest)

    def test_url_unreachable_leaves_no_partial_file(self, tmp_path: Path) -> None:
        """The .part file is cleaned up even when urlretrieve raises URLError."""
        dest = tmp_path / "data" / "dataset.csv"
        with (
            patch(
                "autoconf.utils.autoconf_build.hf_dataset.urllib.request.urlretrieve",
                side_effect=urllib.error.URLError(reason="Connection refused"),
            ),
            pytest.raises(DatasetDownloadError),
        ):
            ensure_dataset(_VALID_URL, dest)

        partial = dest.with_suffix(f"{dest.suffix}.part")
        assert not partial.exists()


class TestEnsureDatasetWrongUrl:
    """ensure_dataset raises DatasetDownloadError when the URL returns an HTTP error."""

    def test_http_404_raises_dataset_download_error(self, tmp_path: Path) -> None:
        """An HTTP 404 response becomes DatasetDownloadError."""
        dest = tmp_path / "dataset.csv"
        error = urllib.error.HTTPError(
            url=_VALID_URL,
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,  # type: ignore[arg-type]
        )
        with (
            patch(
                "autoconf.utils.autoconf_build.hf_dataset.urllib.request.urlretrieve",
                side_effect=error,
            ),
            pytest.raises(DatasetDownloadError, match="HTTP 404"),
        ):
            ensure_dataset(_VALID_URL, dest)

    def test_http_403_raises_dataset_download_error(self, tmp_path: Path) -> None:
        """An HTTP 403 response becomes DatasetDownloadError."""
        dest = tmp_path / "dataset.csv"
        error = urllib.error.HTTPError(
            url=_VALID_URL,
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=None,  # type: ignore[arg-type]
        )
        with (
            patch(
                "autoconf.utils.autoconf_build.hf_dataset.urllib.request.urlretrieve",
                side_effect=error,
            ),
            pytest.raises(DatasetDownloadError, match="HTTP 403"),
        ):
            ensure_dataset(_VALID_URL, dest)

    def test_wrong_url_leaves_no_partial_file(self, tmp_path: Path) -> None:
        """The .part file is cleaned up even when urlretrieve raises HTTPError."""
        dest = tmp_path / "dataset.csv"
        error = urllib.error.HTTPError(
            url=_VALID_URL,
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,  # type: ignore[arg-type]
        )
        with (
            patch(
                "autoconf.utils.autoconf_build.hf_dataset.urllib.request.urlretrieve",
                side_effect=error,
            ),
            pytest.raises(DatasetDownloadError),
        ):
            ensure_dataset(_VALID_URL, dest)

        partial = dest.with_suffix(f"{dest.suffix}.part")
        assert not partial.exists()


class TestEnsureDatasetDirectoryCannotBeCreated:
    """ensure_dataset raises DatasetDownloadError when mkdir fails."""

    def test_mkdir_failure_raises_dataset_download_error(self, tmp_path: Path) -> None:
        """An OSError from mkdir becomes DatasetDownloadError."""
        dest = tmp_path / "data" / "dataset.csv"
        with (
            patch(
                "autoconf.utils.autoconf_build.hf_dataset.Path.mkdir",
                side_effect=OSError("Permission denied"),
            ),
            pytest.raises(DatasetDownloadError, match="Could not create the directory"),
        ):
            ensure_dataset(_VALID_URL, dest)

    def test_mkdir_failure_does_not_attempt_download(self, tmp_path: Path) -> None:
        """urlretrieve is never called when the destination directory cannot be created."""
        dest = tmp_path / "data" / "dataset.csv"
        with (
            patch(
                "autoconf.utils.autoconf_build.hf_dataset.Path.mkdir",
                side_effect=OSError("Read-only file system"),
            ),
            patch(
                "autoconf.utils.autoconf_build.hf_dataset.urllib.request.urlretrieve"
            ) as mock_retrieve,
        ):
            with pytest.raises(DatasetDownloadError):
                ensure_dataset(_VALID_URL, dest)
            mock_retrieve.assert_not_called()


class TestEnsureDatasetUnsupportedScheme:
    """ensure_dataset raises ValueError for unsupported URL schemes."""

    @pytest.mark.parametrize("scheme", ["ftp", "s3", "gs", "ssh", ""])
    def test_unsupported_scheme_raises_value_error(
        self, tmp_path: Path, scheme: str
    ) -> None:
        """Any scheme other than http, https, or file raises ValueError."""
        url = f"{scheme}://example.com/dataset.csv" if scheme else "dataset.csv"
        dest = tmp_path / "dataset.csv"
        with pytest.raises(ValueError, match="Unsupported dataset URL scheme"):
            ensure_dataset(url, dest)

    def test_unsupported_scheme_does_not_attempt_download(self, tmp_path: Path) -> None:
        """urlretrieve is never called when the URL scheme is unsupported."""
        dest = tmp_path / "dataset.csv"
        mock_retrieve = patch(
            "autoconf.utils.autoconf_build.hf_dataset.urllib.request.urlretrieve"
        ).start()
        try:
            with pytest.raises(ValueError, match="Unsupported dataset URL scheme"):
                ensure_dataset("ftp://example.com/dataset.csv", dest)
            mock_retrieve.assert_not_called()
        finally:
            patch.stopall()


class TestEnsureDatasetUrlSanitisation:
    """ensure_dataset rejects URLs with disallowed characters or non-.csv paths."""

    @pytest.mark.parametrize(
        "bad_url",
        [
            # shell metacharacters
            "https://huggingface.co/datasets/org/repo/resolve/main/data.csv;rm%20-rf",
            "https://huggingface.co/datasets/org/repo/resolve/main/data.csv|cat",
            "https://huggingface.co/datasets/org/repo/resolve/main/data.csv&id",
            "https://huggingface.co/datasets/org/repo/resolve/main/data.csv`id`",
            "https://huggingface.co/datasets/org/repo/resolve/main/$(cmd).csv",
            # query string / fragment
            "https://huggingface.co/datasets/org/repo/resolve/main/data.csv?token=x",
            "https://huggingface.co/datasets/org/repo/resolve/main/data.csv#anchor",
            # spaces / angle brackets
            "https://huggingface.co/datasets/org/repo/resolve/main/data file.csv",
            "https://huggingface.co/datasets/org/repo/resolve/main/<data>.csv",
        ],
    )
    def test_disallowed_characters_raise_value_error(
        self, tmp_path: Path, bad_url: str
    ) -> None:
        """A URL containing disallowed characters raises ValueError."""
        dest = tmp_path / "dataset.csv"
        with pytest.raises(ValueError, match="contains disallowed characters"):
            ensure_dataset(bad_url, dest)

    @pytest.mark.parametrize(
        "non_csv_url",
        [
            "https://huggingface.co/datasets/org/repo/resolve/main/data.parquet",
            "https://huggingface.co/datasets/org/repo/resolve/main/data.json",
            "https://huggingface.co/datasets/org/repo/resolve/main/data",
        ],
    )
    def test_non_csv_url_raises_value_error(
        self, tmp_path: Path, non_csv_url: str
    ) -> None:
        """A URL whose path does not end with .csv raises ValueError."""
        dest = tmp_path / "dataset.csv"
        with pytest.raises(ValueError, match=r"must point to a \.csv file"):
            ensure_dataset(non_csv_url, dest)

    def test_sanitisation_does_not_attempt_download(self, tmp_path: Path) -> None:
        """urlretrieve is never called when the URL fails sanitisation."""
        dest = tmp_path / "dataset.csv"
        with (
            patch(
                "autoconf.utils.autoconf_build.hf_dataset.urllib.request.urlretrieve"
            ) as mock_retrieve,
            pytest.raises(ValueError, match="contains disallowed characters"),
        ):
            ensure_dataset(
                "https://huggingface.co/datasets/org/repo/resolve/main/data.csv|ls",
                dest,
            )
        mock_retrieve.assert_not_called()


class TestSanitiseUrl:
    """Unit tests for the _sanitise_url helper."""

    def test_valid_url_passes(self) -> None:
        """A well-formed HuggingFace CSV URL passes without raising."""
        _sanitise_url(_VALID_URL)

    def test_valid_url_with_percent_encoding_passes(self) -> None:
        """Percent-encoded characters in the path are allowed."""
        _sanitise_url(
            "https://huggingface.co/datasets/org/my%20repo/resolve/main/data.csv"
        )

    @pytest.mark.parametrize(
        "char",
        ["|", ";", "&", "$", "`", "!", "?", "#", " ", "<", ">", '"', "'"],
    )
    def test_single_disallowed_character_raises(self, char: str) -> None:
        """Each individual disallowed character triggers ValueError."""
        url = f"https://huggingface.co/datasets/org/repo/resolve/main/data{char}.csv"
        with pytest.raises(ValueError, match="contains disallowed characters"):
            _sanitise_url(url)

    def test_non_csv_extension_raises(self) -> None:
        """A URL whose path ends with .parquet raises ValueError."""
        with pytest.raises(ValueError, match=r"must point to a \.csv file"):
            _sanitise_url(
                "https://huggingface.co/datasets/org/repo/resolve/main/data.parquet"
            )

    def test_csv_extension_case_insensitive(self) -> None:
        """The .csv check is case-insensitive."""
        _sanitise_url("https://huggingface.co/datasets/org/repo/resolve/main/DATA.CSV")


class TestEnsureDatasetHappyPath:
    """ensure_dataset returns the local path on success."""

    def test_existing_file_is_returned_without_download(self, tmp_path: Path) -> None:
        """When the destination file already exists, no download is attempted."""
        dest = tmp_path / "dataset.csv"
        dest.write_text("col\n1\n")
        with patch(
            "autoconf.utils.autoconf_build.hf_dataset.urllib.request.urlretrieve"
        ) as mock_retrieve:
            result = ensure_dataset(_VALID_URL, dest)
        assert result == dest
        mock_retrieve.assert_not_called()

    def test_successful_download_returns_dest_path(self, tmp_path: Path) -> None:
        """A successful download writes the file and returns the destination path."""
        dest = tmp_path / "dataset.csv"

        def _fake_retrieve(url: str, filename: str) -> None:
            Path(filename).write_text("col\n1\n")

        with patch(
            "autoconf.utils.autoconf_build.hf_dataset.urllib.request.urlretrieve",
            side_effect=_fake_retrieve,
        ):
            result = ensure_dataset(_VALID_URL, dest)

        assert result == dest
        assert dest.is_file()


# ---------------------------------------------------------------------------
# HuggingFaceDatasetURL
# ---------------------------------------------------------------------------

_VALID_BASE = "https://huggingface.co"
_VALID_REPO = "ibm-research/LLMFineTuningBench"
_VALID_BRANCH = "main"
_VALID_FILENAME = "ado-sfttrainer-dataset.csv"


class TestHuggingFaceDatasetURLHappyPath:
    """HuggingFaceDatasetURL accepts well-formed components and assembles them."""

    def test_to_url_assembles_components(self) -> None:
        """to_url joins the four components with slashes correctly."""
        hf_url = HuggingFaceDatasetURL(
            base_url=_VALID_BASE,
            dataset_repo=_VALID_REPO,
            branch=_VALID_BRANCH,
            filename=_VALID_FILENAME,
        )
        assert hf_url.to_url() == (
            "https://huggingface.co/datasets/ibm-research/LLMFineTuningBench"
            "/resolve/main/ado-sfttrainer-dataset.csv"
        )

    def test_to_url_strips_redundant_slashes(self) -> None:
        """Leading/trailing slashes on base_url and dataset_repo are normalised."""
        hf_url = HuggingFaceDatasetURL(
            base_url="https://huggingface.co/",
            dataset_repo="/ibm-research/LLMFineTuningBench/",
            branch=_VALID_BRANCH,
            filename=_VALID_FILENAME,
        )
        assert hf_url.to_url() == _VALID_URL

    def test_http_scheme_is_accepted(self) -> None:
        """http:// is a supported scheme."""
        HuggingFaceDatasetURL(
            base_url="http://huggingface.co",
            dataset_repo=_VALID_REPO,
            branch=_VALID_BRANCH,
            filename=_VALID_FILENAME,
        )

    def test_file_scheme_is_accepted(self) -> None:
        """file:// is a supported scheme."""
        HuggingFaceDatasetURL(
            base_url="file://localhost",
            dataset_repo=_VALID_REPO,
            branch=_VALID_BRANCH,
            filename=_VALID_FILENAME,
        )


class TestHuggingFaceDatasetURLBaseUrlValidation:
    """HuggingFaceDatasetURL raises ValueError for invalid base_url values."""

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_base_url_raises(self, blank: str) -> None:
        """A blank base_url raises ValueError."""
        with pytest.raises(ValueError, match="base_url must not be blank"):
            HuggingFaceDatasetURL(
                base_url=blank,
                dataset_repo=_VALID_REPO,
                branch=_VALID_BRANCH,
                filename=_VALID_FILENAME,
            )

    @pytest.mark.parametrize("scheme", ["ftp", "s3", "gs", "ssh"])
    def test_unsupported_scheme_raises(self, scheme: str) -> None:
        """An unsupported URL scheme raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dataset URL scheme"):
            HuggingFaceDatasetURL(
                base_url=f"{scheme}://huggingface.co",
                dataset_repo=_VALID_REPO,
                branch=_VALID_BRANCH,
                filename=_VALID_FILENAME,
            )

    def test_missing_host_raises(self) -> None:
        """A base_url without a host component raises ValueError."""
        with pytest.raises(ValueError, match="must include a host"):
            HuggingFaceDatasetURL(
                base_url="https://",
                dataset_repo=_VALID_REPO,
                branch=_VALID_BRANCH,
                filename=_VALID_FILENAME,
            )

    def test_base_url_with_path_raises(self) -> None:
        """A base_url that includes a path segment raises ValueError."""
        with pytest.raises(ValueError, match="must not contain a path"):
            HuggingFaceDatasetURL(
                base_url="https://huggingface.co/datasets",
                dataset_repo=_VALID_REPO,
                branch=_VALID_BRANCH,
                filename=_VALID_FILENAME,
            )


class TestHuggingFaceDatasetURLDatasetRepoValidation:
    """HuggingFaceDatasetURL raises ValueError for invalid dataset_repo values."""

    @pytest.mark.parametrize("blank", ["", "   ", "/", "//"])
    def test_blank_dataset_repo_raises(self, blank: str) -> None:
        """A blank or slash-only dataset_repo raises ValueError."""
        with pytest.raises(ValueError, match="dataset_repo must not be blank"):
            HuggingFaceDatasetURL(
                base_url=_VALID_BASE,
                dataset_repo=blank,
                branch=_VALID_BRANCH,
                filename=_VALID_FILENAME,
            )

    def test_dataset_repo_ending_with_filename_raises(self) -> None:
        """A dataset_repo whose last segment looks like a file raises ValueError."""
        with pytest.raises(ValueError, match="must not include the filename"):
            HuggingFaceDatasetURL(
                base_url=_VALID_BASE,
                dataset_repo=f"{_VALID_REPO}/{_VALID_FILENAME}",
                branch=_VALID_BRANCH,
                filename=_VALID_FILENAME,
            )


class TestHuggingFaceDatasetURLBranchValidation:
    """HuggingFaceDatasetURL raises ValueError for invalid branch values."""

    @pytest.mark.parametrize("blank", ["", "   ", "/", "//"])
    def test_blank_branch_raises(self, blank: str) -> None:
        """A blank or slash-only branch raises ValueError."""
        with pytest.raises(ValueError, match="branch must not be blank"):
            HuggingFaceDatasetURL(
                base_url=_VALID_BASE,
                dataset_repo=_VALID_REPO,
                branch=blank,
                filename=_VALID_FILENAME,
            )


class TestHuggingFaceDatasetURLFilenameValidation:
    """HuggingFaceDatasetURL raises ValueError for invalid filename values."""

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_filename_raises(self, blank: str) -> None:
        """A blank filename raises ValueError."""
        with pytest.raises(ValueError, match="filename must not be blank"):
            HuggingFaceDatasetURL(
                base_url=_VALID_BASE,
                dataset_repo=_VALID_REPO,
                branch=_VALID_BRANCH,
                filename=blank,
            )

    @pytest.mark.parametrize("sep", ["/", "\\"])
    def test_filename_with_path_separator_raises(self, sep: str) -> None:
        """A filename containing a path separator raises ValueError."""
        with pytest.raises(ValueError, match="must be a plain filename"):
            HuggingFaceDatasetURL(
                base_url=_VALID_BASE,
                dataset_repo=_VALID_REPO,
                branch=_VALID_BRANCH,
                filename=f"subdir{sep}dataset.csv",
            )

    @pytest.mark.parametrize("name", ["dataset.parquet", "dataset.json", "dataset"])
    def test_non_csv_filename_raises(self, name: str) -> None:
        """A filename that does not end with .csv raises ValueError."""
        with pytest.raises(ValueError, match=r"must end with '\.csv'"):
            HuggingFaceDatasetURL(
                base_url=_VALID_BASE,
                dataset_repo=_VALID_REPO,
                branch=_VALID_BRANCH,
                filename=name,
            )
