# Developing the Core of ado

This guide is for contributors who want to work on the core `ado` framework —
its schema, CLI, metastore, execution engine, or test suite. It's your starting
point for getting set up and understanding our development standards.

If you want to write new experiments or analysis tools, please see
[Developing ado Plugins](plugin-developers.md) instead.

---

## 1. Your Development Environment

Full instructions for setting up a core development environment, including `uv`
installation and pre-commit hooks, are in our main
[Developing ado](./developing.md) guide.

The quick start to get you running is:

    git clone https://github.com/IBM/ado.git
    cd ado
    uv sync --group test --reinstall
    source .venv/bin/activate

---

## 2. Core Contributor Checklist

The **authoritative developer documentation** for our tooling and processes
lives in [Developing ado](./developing.md) and
[Contributing](./contributing.md).

Before you submit a pull request, ensure your changes adhere to this checklist:

- **Code Style:** Follow **PEP8** naming, include all **type annotations**, and
  use **Google-style docstrings**.
- **Formatting & Linting:** Code must be formatted with **ruff format** and pass
  **ruff check**.
- **Testing:** All new features or fixes must include or update **tests**. We
  prefer an integration-first, TDD-style workflow.
- **Commit Hooks:** All changes must pass our **pre-commit** hooks.

Install the hooks with `pre-commit install` right after you set up your
environment. This will automate checks for formatting, secrets, headers, and
conventional commit messages.

---

## 3. Harnessing Coding Agents

This repository is designed to help coding agents help _you_. By opening the
cloned repo in an agent-enabled editor, you activate several tools that simplify
development and enforce our standards.

- **Automated Workflows:** The agent uses
  [SKILLS](https://github.com/IBM/ado/tree/main/.cursor/skills) to automate
  common tasks.
- **Contextual Rules:** It understands our coding patterns by reading from files
  like [`AGENTS.md`](https://github.com/IBM/ado/blob/main/AGENTS.md).

Most importantly for a core developer, you can **ask the agent to review your
work** before you even create a pull request. The template in
[`.cursor/commands/review.md`](https://github.com/IBM/ado/blob/main/.cursor/commands/review.md)
instructs an agent to walk through your branch, check it against our guidelines,
and produce a structured report.

---

## 4. Running the Test Suite

To run the full test suite in parallel, use the following command:

    uv run pytest -n auto tests/

Please refer to the developer and contribution guides for details on test
layout, fixtures, and our expectations for test coverage.
