# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

"""Utilities for downloading and storing a HuggingFace dataset CSV locally."""

import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SUPPORTED_SCHEMES = {"file", "http", "https"}

# Characters allowed anywhere in a dataset URL path/host segment.
# Covers unreserved chars (RFC 3986 §2.3) plus the colon and slash needed
# for scheme/host/path, the @ for userinfo, and % for percent-encoding.
# Deliberately excludes shell metacharacters (|, ;, &, $, `, !),
# query strings (?), fragments (#), and angle-brackets / spaces.
_URL_SAFE_RE = re.compile(r"^[A-Za-z0-9\-._~/:@%]+$")


class DatasetDownloadError(RuntimeError):
    """Raised when the training dataset cannot be downloaded."""


@dataclass(frozen=True)
class HuggingFaceDatasetURL:
    """A validated HuggingFace dataset URL split into its four components.

    Attributes:
        base_url: Scheme and host, e.g. ``"https://huggingface.co"``.
        dataset_repo: Organisation and repository on the host,
            e.g. ``"ibm-research/LLMFineTuningBench"``.
        branch: Branch or revision, e.g. ``"main"``.
        filename: The CSV filename, e.g. ``"ado-sfttrainer-dataset.csv"``.
    """

    base_url: str
    dataset_repo: str
    branch: str
    filename: str

    def __post_init__(self) -> None:
        """Validate each component on construction."""
        self._validate_base_url(base_url=self.base_url)
        self._validate_dataset_repo(self.dataset_repo)
        self._validate_branch(self.branch)
        self._validate_filename(self.filename)

    @staticmethod
    def _validate_base_url(base_url: str) -> None:
        """Validate the base URL component.

        Args:
            base_url: Scheme-and-host part of the URL.

        Raises:
            ValueError: If the base URL is blank, has an unsupported scheme,
                or contains a path component.
        """
        if not base_url or not base_url.strip():
            raise ValueError("base_url must not be blank")
        parsed = urllib.parse.urlparse(base_url)
        scheme = parsed.scheme
        if scheme not in _SUPPORTED_SCHEMES:
            raise ValueError(
                f"Unsupported dataset URL scheme: {scheme!r}. "
                f"Must be one of {sorted(_SUPPORTED_SCHEMES)}"
            )
        if not parsed.netloc:
            raise ValueError(
                f"base_url {base_url!r} must include a host (e.g. 'https://huggingface.co')"
            )
        if parsed.path.strip("/"):
            raise ValueError(
                f"base_url {base_url!r} must not contain a path; "
                "put the path in dataset_path"
            )

    @staticmethod
    def _validate_dataset_repo(dataset_repo: str) -> None:
        """Validate the dataset repository component.

        Args:
            dataset_repo: Organisation/repository segment of the URL,
                e.g. ``"ibm-research/LLMFineTuningBench"``.

        Raises:
            ValueError: If the dataset repository is blank or contains a
                filename component (i.e. ends with a file extension).
        """
        if not dataset_repo or not dataset_repo.strip():
            raise ValueError("dataset_repo must not be blank")
        stripped = dataset_repo.strip("/")
        if not stripped:
            raise ValueError("dataset_repo must not be blank")
        last_segment = stripped.split("/")[-1]
        if "." in last_segment:
            raise ValueError(
                f"dataset_repo {dataset_repo!r} must not include the filename; "
                "put the filename in the filename field"
            )

    @staticmethod
    def _validate_branch(branch: str) -> None:
        """Validate the branch component.

        Args:
            branch: Branch or revision name, e.g. ``"main"``.

        Raises:
            ValueError: If the branch is blank or consists only of slashes.
        """
        if not branch or not branch.strip():
            raise ValueError("branch must not be blank")
        if not branch.strip("/"):
            raise ValueError("branch must not be blank")

    @staticmethod
    def _validate_filename(filename: str) -> None:
        """Validate the filename component.

        Args:
            filename: The CSV filename at the end of the URL.

        Raises:
            ValueError: If the filename is blank, contains path separators, or
                does not end with ``.csv``.
        """
        if not filename or not filename.strip():
            raise ValueError("filename must not be blank")
        if "/" in filename or "\\" in filename:
            raise ValueError(
                f"filename {filename!r} must be a plain filename with no path separators"
            )
        if not filename.lower().endswith(".csv"):
            raise ValueError(f"filename {filename!r} must end with '.csv'")

    def to_url(self) -> str:
        """Assemble the four components into a single URL string.

        Returns:
            The full dataset URL.
        """
        base = self.base_url.rstrip("/")
        repo = self.dataset_repo.strip("/")
        branch = self.branch.strip("/")
        return f"{base}/datasets/{repo}/resolve/{branch}/{self.filename}"


def _sanitise_url(dataset_url: str) -> None:
    """Sanitise a dataset URL to ensure it contains no executable payload.

    Checks that every character in the URL belongs to the safe allowlist and
    that the URL path ends with ``.csv``.  This guards against shell
    meta-characters, query strings, fragments, and other characters that
    could be used to inject commands or redirect the download.

    Args:
        dataset_url: The raw URL string to validate.

    Raises:
        ValueError: If the URL contains characters outside the allowed set or
            does not end with ``.csv``.
    """
    if not _URL_SAFE_RE.match(dataset_url):
        raise ValueError(
            f"Dataset URL {dataset_url!r} contains disallowed characters. "
            "Only alphanumerics and '-._~/:@%' are permitted."
        )
    parsed_path = urllib.parse.urlparse(dataset_url).path
    if not parsed_path.lower().endswith(".csv"):
        raise ValueError(f"Dataset URL {dataset_url!r} must point to a .csv file.")


def ensure_dataset(dataset_url: str, data_path: Path) -> Path:
    """Download the Hugging Face dataset when it is not already available.

    Args:
        dataset_url: URL of the source CSV on Hugging Face.
        data_path: Local destination for the CSV.

    Returns:
        The local dataset path.

    Raises:
        DatasetDownloadError: If the dataset cannot be downloaded or the
            destination directory cannot be created.
        ValueError: If the dataset URL uses an unsupported scheme or contains
            disallowed characters.
    """
    if data_path.is_file():
        logger.info("Using existing dataset at %s", data_path)
        return data_path

    scheme = urllib.parse.urlparse(dataset_url).scheme
    if scheme not in {"file", "http", "https"}:
        raise ValueError(f"Unsupported dataset URL scheme: {scheme}")

    # Sanitise and check the URL to validate that there is no executable payload present.
    # The URL must have only allowed characters and end in csv.
    _sanitise_url(dataset_url)

    try:
        data_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DatasetDownloadError(
            f"Could not create the directory {data_path.parent}: {error}. "
            "Check filesystem permissions."
        ) from error

    partial_path = data_path.with_suffix(f"{data_path.suffix}.part")
    logger.info("Downloading dataset from %s to %s", dataset_url, data_path)
    try:
        urllib.request.urlretrieve(dataset_url, partial_path)  # noqa: S310
        partial_path.replace(data_path)
    except urllib.error.HTTPError as error:
        raise DatasetDownloadError(
            f"Could not download the dataset from {dataset_url}: HTTP "
            f"{error.code} ({error.reason}). The dataset may have moved. "
            "Provide a valid URL with --dataset-url or an existing local CSV "
            "with --data-path."
        ) from error
    except urllib.error.URLError as error:
        raise DatasetDownloadError(
            f"Could not download the dataset from {dataset_url}: {error.reason}. "
            "Check the network connection, provide a valid URL with "
            "--dataset-url, or use an existing local CSV with --data-path."
        ) from error
    finally:
        partial_path.unlink(missing_ok=True)

    return data_path
