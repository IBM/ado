# Developing ado

## Project Setup

To start developing ado, you need to set up a Python environment. We use `uv` for project and dependency management.

### Installing uv

To install `uv`, refer to
the [official installation documentation](https://docs.astral.sh/uv/getting-started/installation/#installing-uv) and
choose your preferred method.

### Creating a development virtual environment

Create a development virtual environment by executing the following commands in the top-level of the `ado` repository:

```commandline
uv sync
source .venv/bin/activate
```

> [!NOTE]
> 
> This installs `ado` in editable mode.

> [!NOTE]
> 
> In line with uv's defaults, the `uv sync` command creates a `.venv` in the top-level of the project's repository.
> Note that environments created by `uv sync` are intended only to be used when developing a specific project and should not be shared across projects.

> [!CAUTION]
> 
> `uv sync` ensures a reproducible development environment is created by using a lock-file, `uv.lock`.
>  Only packages in the lockfile are installed, and other packages found in the virtual environment will be deleted.
>  See [Making changes to dependencies](#making-changes-to-dependencies) for how to add packages to the lockfile.

If you want to create your development virtual environment at an alternate location, $PATH, then 
```commandline
uv venv $PATH
source $PATH/bin/activate
uv sync --active
```
> [!CAUTION]
> 
> If you create a development in a different location you must direct `uv sync` explicitly to use it with `--active`
> If you do not it will default to using `.venv` in the project top-level directory. 


## Code style

> [!NOTE]  
> 
> See the [Automating checks with pre-commit](#automating-checks-with-pre-commit) section to automate this.

This repository follows the [`black`](https://black.readthedocs.io/en/stable/) style for formatting. 

You can format your code by:

- Manually running `black <the folder containing files to format>`
- Setting up PyCharm to use the `black`
  integration: https://www.jetbrains.com/help/pycharm/reformat-and-rearrange-code.html#format-python-code-with-black
- Using
  the ["Black formatter" extension for VSCode](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
  and setting it as the default formatter: https://code.visualstudio.com/docs/python/formatting#_set-a-default-formatter

## Linting with ruff

> [!NOTE]  
> 
> See the [Automating checks with pre-commit](#automating-checks-with-pre-commit) section to automate this.

This repository uses `ruff` to enforce linting rules. Install it using one of the methods described in the [official
`ruff` documentation](https://docs.astral.sh/ruff/installation/).
To run linting checks, execute:

```commandline
ruff check --exclude website
```

## Secret scanning

> [!NOTE]  
> 
> See the [Automating checks with pre-commit](#automating-checks-with-pre-commit) section to automate this.

This repository uses IBM's [detect-secrets](https://github.com/ibm/detect-secrets) to scan for secrets before the code
is pushed to GitHub.
Follow installation instructions in their
repository: https://github.com/ibm/detect-secrets?tab=readme-ov-file#example-usage

To update the secrets database manually, run:

```commandline
detect-secrets scan --update .secrets.baseline
```

To audit detected secrets, use:

```commandline
detect-secrets audit .secrets.baseline
```

## Commit style

We require commit messages to use the [conventional commit style](https://www.conventionalcommits.org/en/v1.0.0/).

Conventional Commit messages follow the following pattern (**NOTE**: the scope is optional):

```
type(scope): subject

extended body
```

Where type is one of: build, chore, ci, docs, feat, fix, perf, refactor, revert, style, test.

## Copyright and license headers

We require copyright and SPDX license headers to be added to the source code.
This can be automated by using Hashicorp's Copywrite tool: https://github.com/hashicorp/copywrite

Once installed, run

```shell
copywrite headers
```

## Automating checks with pre-commit

To automate the checks for code style, linting, and security, you can utilize the provided pre-commit hooks.

### Installing the hooks

After [having synced the dependencies](#creating-a-development-virtual-environment), install the hooks with:

```commandline
pre-commit install
```

This command will configure pre-commit to run automatically before each commit, highlighting any issues and preventing
the commit if problems are found.

### Handling pre-commit failures

1. **Black code formatting failures**: try committing again, `black` might have reformatted your code in-place.
     - If black fails to format your code, your files have syntax errors. [Try manually running black](#code-style).
2. **Ruff linter failures**: run `ruff` as specified in [Linting with ruff](#linting-with-ruff) and fix the
   code that is causing the failures.
     - In case of false positives, you might need to add `#noqa` annotations.
     - If your local ruff installation does not detect any failure you may be using an old version that needs updating.
3. **Detect secrets failures**: include `.secrets.baseline` in your commit, it was updated by the pre-commit hook.
     - If secrets are detected, audit them as specified in [Secret scanning](#secret-scanning).
4. **Commit style failures**: change your commit message to match conventional commits. 
   See [Commit style](#commit-style) for more in-depth information.
5. **Misspellings detected by codespell**: fix the misspellings reported or 
   [add an inline ignore comment](https://github.com/codespell-project/codespell?tab=readme-ov-file#inline-ignore).
6. **uv export failures**: commit the updated `requirements.txt` file. It has been
   updated following changes to the lock file.

## Making changes to dependencies

As mentioned in [Project Setup](#project-setup), we use `uv` to manage dependencies.
This means that all changes to dependencies **must** be done via `uv`, and not by manually editing `pyproject.toml`.

The relevant documentation on `uv`'s website is
available [here](https://docs.astral.sh/uv/concepts/projects/dependencies/#managing-dependencies), but at a glance:

### Adding base dependencies

If you are adding (or updating) base dependencies for `ado`, you should use the [
`uv add` command](https://docs.astral.sh/uv/concepts/projects/dependencies/#adding-dependencies):

> [!NOTE]  
> 
> You can optionally add specific version selectors. By default, `uv` will add `>=CURRENT_VERSION`.

```commandline
uv add pydantic
```

### Adding optional dependencies

Dependencies may be optional, making them available only when using extras, such as `ado-core[my-extra]`. To add
these kind of dependencies, use the [
`uv add --optional` command](https://docs.astral.sh/uv/concepts/projects/dependencies/#optional-dependencies):

```commandline
uv add --optional validation pydantic
```

### Adding dependency groups

Sometimes we might want to include dependencies that have a specific purpose, like testing the code, linting it, or
building the documentation. This is a perfect use case for dependency groups, sets of dependencies that do not get
published to indices like PyPI and are not installed with ado.
A noteworthy dependency group is the `dev` group, which `uv` installs by default when syncing dependencies.

Users are highly encouraged to read the documentation available both on uv's and Python's website:

- https://docs.astral.sh/uv/concepts/projects/dependencies/#development-dependencies
- https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups
- https://packaging.python.org/en/latest/specifications/dependency-groups

With `uv` you can add dependencies to groups using `uv add --group NAME`:

> [!NOTE]  
> 
> For the `dev` group there is the shorthand `--dev` that replaces `--group dev`.

```commandline
uv add --group dev pytest
```
