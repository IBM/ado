# Developing ado

## Project Setup

To start developing ado, you need to set up a Python environment. We use `uv`
for project and dependency management.

### Installing uv

To install `uv`, refer to the
[official installation documentation](https://docs.astral.sh/uv/getting-started/installation/#installing-uv)
and choose your preferred method.

### Creating a development virtual environment

Create a development virtual environment by executing the following commands in
the top-level of the `ado` repository:

> [!CAUTION]
>
> If you create a development in a different location you must direct `uv sync`
> explicitly to use it with `--active` If you do not, it will default to using
> `.venv` in the project top-level directory. See the
> [using a custom location for the venv](#using-a-custom-location-for-the-venv)
> section for instructions on how to do this.

```commandline
uv sync
source .venv/bin/activate
```

> [!NOTE]
>
> This installs `ado` in editable mode.

<!-- markdownlint-disable-next-line no-blanks-blockquote -->
> [!NOTE]
>
> In line with uv's defaults, the `uv sync` command creates a `.venv` in the
> top-level of the project's repository. Note that environments created by
> `uv sync` **are intended only to be used when developing a specific project
> and should not be shared across projects.**

<!-- markdownlint-disable-next-line no-blanks-blockquote -->
> [!CAUTION]
>
> `uv sync` ensures a reproducible development environment is created by using a
> lock-file, `uv.lock`. Only packages in the lockfile are installed, and other
> packages found in the virtual environment **will be deleted**. See
> [Making changes to dependencies](#making-changes-to-dependencies) for how to
> add packages to the lockfile.

#### Using a custom location for the venv

If you want to create your development virtual environment at an alternate
location, $LOCATION, then run:

```commandline
uv venv $LOCATION
source $LOCATION/bin/activate
uv sync --active
```

## Code style

> [!NOTE]
>
> See the
> [Automating checks with pre-commit](#automating-checks-with-pre-commit)
> section to automate this.

This repository follows the [`black`](https://black.readthedocs.io/en/stable/)
style for formatting.

You can format your code by:

- Manually running `black tests/ orchestrator/ plugins/`
- Setting up PyCharm to use the `black` integration:
  <https://www.jetbrains.com/help/pycharm/reformat-and-rearrange-code.html#format-python-code-with-black>
- Using the
  ["Black formatter" extension for VSCode](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
  and setting it as the default formatter:
  <https://code.visualstudio.com/docs/python/formatting#_set-a-default-formatter>

## Linting code with ruff

> [!NOTE]
>
> See the
> [Automating checks with pre-commit](#automating-checks-with-pre-commit)
> section to automate this.

This repository uses `ruff` to enforce linting rules. Install it using one of
the methods described in the
[official `ruff` documentation](https://docs.astral.sh/ruff/installation/). To
run linting checks, execute:

```commandline
ruff check --exclude website
```

## Linting markdown with markdownlint-cli2

> [!NOTE]
>
> See the
> [Automating checks with pre-commit](#automating-checks-with-pre-commit)
> section to automate this.

This repository uses `markdownlint-cli2` to enforce linting rules on markdown
files. Install it using one of the methods described in the
[official documentation](https://github.com/DavidAnson/markdownlint-cli2?tab=readme-ov-file#install).
To run linting checks, execute:

```commandline
markdownlint-cli2 "**/*.md" "#.venv" --fix
```

### Prettier for lines too long

> [!WARNING]
>
> Prettier might undo some changes that `markdownlint-cli2` has done. A common
> error is adding a line after the `markdownlint-disable-next-line` comments

Line-too-long errors do not get automatically fixed by `markdownlint-cli2`. We
recommend using `prettier` to autoformat markdown in that case. The official
website provides instructions to:

- [Install prettier on your machine](https://prettier.io/docs/install).
- [Integrate prettier on your editor of choice](https://prettier.io/docs/editors).

Prettier can be run as a CLI tool with:

```commandline
prettier -w "**/*.md"
```

## Secret scanning

> [!NOTE]
>
> See the
> [Automating checks with pre-commit](#automating-checks-with-pre-commit)
> section to automate this.

This repository uses IBM's
[detect-secrets](https://github.com/ibm/detect-secrets) to scan for secrets
before the code is pushed to GitHub. Follow installation instructions in their
repository:
<https://github.com/ibm/detect-secrets?tab=readme-ov-file#example-usage>

To update the secrets database manually, run:

```commandline
detect-secrets scan --update .secrets.baseline
```

To audit detected secrets, use:

```commandline
detect-secrets audit .secrets.baseline
```

If the pre-commit hook raises an error but the audit command succeeds with just
`Nothing to audit!` then run `detect-secrets scan --update .secrets.baseline`
to perform a full scan and then repeat the `audit` command.

## Commit style

We require commit messages to use the
[conventional commit style](https://www.conventionalcommits.org/en/v1.0.0/).

Conventional Commit messages follow the pattern (**NOTE**: the scope is
optional):

```text
type(scope): subject

extended body
```

Where type is one of: build, chore, ci, docs, feat, fix, perf, refactor, revert,
style, test.

## Copyright and license headers

> [!NOTE]
>
> See the
> [Automating checks with pre-commit](#automating-checks-with-pre-commit)
> section to automate this.

We require copyright and SPDX license headers to be added to the source code.
This can be automated by using Hashicorp's Copywrite tool:
<https://github.com/hashicorp/copywrite>

Once installed, run

```shell
copywrite headers
```

## YAML file formatting

> [!NOTE]
>
> See the
> [Automating checks with pre-commit](#automating-checks-with-pre-commit)
> section to automate this.

We require YAML files to be properly formatted. This can be automated with
yamlfmt: <https://github.com/google/yamlfmt>.

Once installed, run

```shell
yamlfmt .
```

## Website link checking

To make it less likely for us to push commits with broken links, we use
[linkcheck](https://github.com/filiph/linkcheck) to check if our website
contains broken links. You can install it with your preferred method from the
ones provided
[in the official documentation](https://github.com/filiph/linkcheck?tab=readme-ov-file#installation).

On one terminal, navigate to the website directory and start serving the website
using mkdocs:

```commandline
cd website && mkdocs serve --clean
```

On a different terminal, run the linkchecker:

```commandline
linkcheck http://127.0.0.1:8000/ado/ \
                --skip-file=.linkchecker_skip
```

## Automating checks with pre-commit

To automate the checks for code style, linting, and security, you can utilize
the provided pre-commit hooks.

### Installing the hooks

> [!IMPORTANT]
>
> Before installing the hooks, make sure you have the following prerequisites:
>
> - A developer virtual environment
>   [created with uv](#creating-a-development-virtual-environment)
> - A recent version of [NodeJS](https://nodejs.org/en/download/)
>   - On MacOS we suggest
>     [installing it via brew](https://formulae.brew.sh/formula/node) for ease
>     of use
> - [Copywrite](https://github.com/hashicorp/copywrite) installed
>   - On MacOS we suggest
>     [installing it via brew](https://github.com/hashicorp/copywrite#getting-started)
>     for ease of use.

```commandline
pre-commit install
```

This command will configure pre-commit to run automatically before each commit,
highlighting any issues and preventing the commit if problems are found.

### Handling pre-commit failures

1. **Black code formatting failures**: try committing again, `black` might have
   reformatted your code in-place.
   - If black fails to format your code, your files have syntax errors.
     [Try manually running black](#code-style).
2. **Ruff linter failures**: run `ruff` as specified in
   [Linting code with ruff](#linting-code-with-ruff) and fix the code that is
   causing the failures.
   - In case of false positives, you might need to add `#noqa` annotations.
   - If your local ruff installation does not detect any failure you may be
     using an old version that needs updating.
3. **Detect secrets failures**: include `.secrets.baseline` in your commit, it
   was updated by the pre-commit hook.
   - If secrets are detected, audit them as specified in
     [Secret scanning](#secret-scanning).
4. **Commit style failures**: change your commit message to match conventional
   commits. See [Commit style](#commit-style) for more in-depth information.
5. **Missing headers**: commit the updated files. It has been updated following
   running of `copywrite`.
6. **Misspellings detected by codespell**: fix the misspellings reported or
   [add an inline ignore comment](https://github.com/codespell-project/codespell?tab=readme-ov-file#inline-ignore).
7. **uv export failures**: commit the updated `requirements.txt` file. It has
   been updated following changes to the lock file.
8. **Markdown linter failures**: `markdownlint-cli2` usually fixes most issues
   automatically. If you review its error message and still don’t see a clear
   explanation or solution, try recommitting your changes and let the tool
   re-run.

## Making changes to dependencies

As mentioned in [Project Setup](#project-setup), we use `uv` to manage
dependencies. This means that all changes to dependencies **must** be done via
`uv`, and not by manually editing `pyproject.toml`.

<!-- markdownlint-disable descriptive-link-text -->
The relevant documentation on `uv`'s website is available
[here](https://docs.astral.sh/uv/concepts/projects/dependencies/#managing-dependencies)
, but at a glance:
<!-- markdownlint-enable descriptive-link-text -->

### Adding base dependencies

If you are adding (or updating) base dependencies for `ado`, you should use the
[`uv add` command](https://docs.astral.sh/uv/concepts/projects/dependencies/#adding-dependencies):

> [!NOTE]
>
> You can optionally add specific version selectors. By default, `uv` will add
> `>=CURRENT_VERSION`.

```commandline
uv add pydantic
```

### Adding optional dependencies

Dependencies may be optional, making them available only when using extras, such
as `ado-core[my-extra]`. To add these kind of dependencies, use the
[`uv add --optional` command](https://docs.astral.sh/uv/concepts/projects/dependencies/#optional-dependencies):

```commandline
uv add --optional validation pydantic
```

### Adding dependency groups

Sometimes we might want to include dependencies that have a specific purpose,
like testing the code, linting it, or building the documentation. This is a
perfect use case for dependency groups, sets of dependencies that do not get
published to indices like PyPI and are not installed with ado. A noteworthy
dependency group is the `dev` group, which `uv` installs by default when syncing
dependencies.

Users are highly encouraged to read the documentation available both on uv's and
Python's website:

- <https://docs.astral.sh/uv/concepts/projects/dependencies/#development-dependencies>
- <https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups>
- <https://packaging.python.org/en/latest/specifications/dependency-groups>

With `uv` you can add dependencies to groups using `uv add --group NAME`:

> [!NOTE]
>
> For the `dev` group there is the shorthand `--dev` that replaces
> `--group dev`.

```commandline
uv add --group dev pytest
```
