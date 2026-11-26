# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for the structured review model of the ado-review-pr skill."""

import importlib.util
import json
import pathlib
import types

import pydantic
import pytest
import typer.testing

HEAD_SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def pr_review_model(agents_dir: pathlib.Path) -> types.ModuleType:
    """Load the skill's script as a module."""
    path = agents_dir / "skills/ado-review-pr/scripts/pr_review_model.py"
    spec = importlib.util.spec_from_file_location("pr_review_model", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_review_lifecycle(pr_review_model: types.ModuleType) -> None:
    """Create, dump, and recreate a Review and verify field transformations."""
    review = pr_review_model.Review(
        commit_id=HEAD_SHA,
        summary="Looks fine",
        comments=[
            {
                "path": "./ado/x.py",
                "line": 9,
                "start_line": 7,
                "body": "Use a constant",
                "suggestion": None,
            }
        ],
        replies=[
            {"in_reply_to": 42, "body": "Addressed", "thread_state": "resolved"},
            {"in_reply_to": 43, "body": "Question answered"},
        ],
    )

    dumped = review.model_dump()
    recreated = pr_review_model.Review.model_validate(dumped)

    assert recreated == review
    assert recreated.comments[0].path == "ado/x.py"
    assert recreated.comments[0].suggestion == ""
    assert [reply.thread_state for reply in recreated.replies] == [
        "resolved",
        "unchanged",
    ]


@pytest.mark.parametrize(
    "comment",
    [
        {"path": "a.py", "line": 3, "start_line": 5, "body": "x"},
        {"path": "./", "line": 3, "body": "x"},
    ],
)
def test_review_comment_rejects_invalid_input(
    pr_review_model: types.ModuleType, comment: dict[str, object]
) -> None:
    """Reject a ReviewComment with start_line after line, or an empty path after stripping ./."""
    with pytest.raises(pydantic.ValidationError):
        pr_review_model.ReviewComment.model_validate(comment)


def test_review_requires_full_commit_sha(pr_review_model: types.ModuleType) -> None:
    """Reject a Review whose commit_id is not a full 40-character hex SHA."""
    with pytest.raises(pydantic.ValidationError, match="commit_id"):
        pr_review_model.Review(commit_id="abc123", summary="x")


def test_main_normalizes_file_input(
    pr_review_model: types.ModuleType,
    tmp_path: pathlib.Path,
) -> None:
    """Write a valid review to a file, invoke main, and check normalized JSON output."""
    review_file = tmp_path / "review.json"
    review_file.write_text(
        json.dumps({"commit_id": HEAD_SHA, "summary": "line one\nline two"}),
        encoding="utf-8",
    )

    result = typer.testing.CliRunner().invoke(pr_review_model.app, [str(review_file)])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "commit_id": HEAD_SHA,
        "summary": "line one\nline two",
        "comments": [],
        "replies": [],
    }


def test_main_reports_invalid_json_with_location(
    pr_review_model: types.ModuleType,
) -> None:
    """Report a JSON syntax error with its line and column to stderr."""
    result = typer.testing.CliRunner().invoke(
        pr_review_model.app, input='{"summary": "say "hi" now"}'
    )

    assert result.exit_code == 1
    assert "Invalid JSON" in result.stderr
    assert "line 1 column" in result.stderr


def test_main_reads_stdin_and_prints_schema(
    pr_review_model: types.ModuleType,
) -> None:
    """Accept review JSON from stdin and print the JSON schema when --schema is passed."""
    runner = typer.testing.CliRunner()

    from_stdin = runner.invoke(
        pr_review_model.app,
        input=json.dumps({"commit_id": HEAD_SHA, "summary": "ok"}),
    )
    schema = runner.invoke(pr_review_model.app, ["--schema"])

    assert from_stdin.exit_code == 0
    assert json.loads(from_stdin.stdout)["summary"] == "ok"
    assert schema.exit_code == 0
    assert (
        "thread_state"
        in json.loads(schema.stdout)["$defs"]["CommentReply"]["properties"]
    )


def test_main_reports_every_problem(
    pr_review_model: types.ModuleType,
    tmp_path: pathlib.Path,
) -> None:
    """Report all validation errors (commit_id, summary, reply id) in a single run."""
    review_file = tmp_path / "review.json"
    review_file.write_text(
        json.dumps(
            {
                "commit_id": "abc",
                "summary": "",
                "replies": [{"in_reply_to": 0, "body": "x"}],
            }
        ),
        encoding="utf-8",
    )

    result = typer.testing.CliRunner().invoke(pr_review_model.app, [str(review_file)])

    assert result.exit_code == 1
    error_output = result.stderr
    assert "commit_id" in error_output
    assert "summary" in error_output
    assert "replies.0.in_reply_to" in error_output


def test_main_reports_root_location_for_non_object_document(
    pr_review_model: types.ModuleType,
) -> None:
    """Use <root> as the location when the top-level JSON value is not an object."""
    result = typer.testing.CliRunner().invoke(pr_review_model.app, input="[1]")

    assert result.exit_code == 1
    assert "<root>:" in result.stderr


def test_main_truncates_long_received_values(
    pr_review_model: types.ModuleType,
) -> None:
    """Truncate received values longer than 80 characters with an ellipsis in error output."""
    result = typer.testing.CliRunner().invoke(
        pr_review_model.app,
        input=json.dumps({"commit_id": "z" * 200, "summary": "x"}),
    )

    assert result.exit_code == 1
    assert "z" * 200 not in result.stderr
    assert "...)" in result.stderr


def test_main_rejects_missing_file(
    pr_review_model: types.ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exit with status 2 and report 'does not exist' when the FILE argument is missing."""
    monkeypatch.chdir(tmp_path)

    result = typer.testing.CliRunner().invoke(pr_review_model.app, ["missing.json"])

    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_main_reports_non_utf8_file_as_invalid(
    pr_review_model: types.ModuleType,
    tmp_path: pathlib.Path,
) -> None:
    """Report a non-UTF-8 file as an invalid review with exit status 1."""
    review_file = tmp_path / "review.json"
    review_file.write_bytes(b'{"commit_id": "\xff\xfe", "summary": "x"}')

    result = typer.testing.CliRunner().invoke(pr_review_model.app, [str(review_file)])

    assert result.exit_code == 1
    assert "Invalid review" in result.stderr
