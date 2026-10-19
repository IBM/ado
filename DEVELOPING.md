# Developing ado

## Project Setup

To start developing ado, you need to set up a Python environment. We use `uv`
for project and dependency management.

### Installing uv

To install `uv`, refer to the
[official installation documentation](https://docs.astral.sh/uv/getting-started/installation/#installing-uv)
and choose your preferred method. Use uv 0.13.0 or newer to match the
repository's pre-commit hooks and lockfile format.

### Supported development platforms

Linux and macOS support the core and plugin test suites, subject to each
plugin's Python version requirements. Native Windows x64 supports core
development with Python 3.11, selected by `.python-version`. The Windows test
environment includes AutoConf and TRIM, and excludes SFTTrainer because its
`aimrocks` dependency has no Windows distribution.
The vLLM actuator client is included; its optional vLLM runtime runs on Linux.

[Ray's Windows support is in beta](https://docs.ray.io/en/latest/ray-overview/installation.html#windows-support).
Use a local Ray instance for native development. Use Linux for multi-node
clusters and the full plugin test matrix.

### Native Windows setup (PowerShell)

Install Git for Windows, uv 0.13.0 or newer, and a recent version of
[Node.js](https://nodejs.org/en/download/). Enable
[Windows Developer Mode](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development)
or obtain equivalent symlink privileges **before cloning**. The repository
contains symbolic links used by documentation and agent configuration.

Clone into the Windows filesystem with symlink support:

```powershell
git clone -c core.symlinks=true https://github.com/IBM/ado.git
cd ado
git config --get core.symlinks
```

The last command should print `true`. Verify every committed symlink resolves
before installing or running hooks:

```powershell
git ls-files --stage | ForEach-Object {
    if ($_ -match '^120000 \S+ \d+\t(.+)$') {
        $linkPath = $Matches[1]
        $link = Get-Item -LiteralPath $linkPath -Force
        if ($link.LinkType -ne 'SymbolicLink' -or
            -not (Test-Path -LiteralPath $linkPath)) {
            throw "Expected a working symbolic link: $linkPath"
        }
    }
}
```

For example, `(Get-Item docs/MAINTAINERS.md).LinkType` should report
`SymbolicLink`. If an existing checkout contains ordinary files holding link
targets, preserve your local changes and make a fresh clone after enabling
symlink support. Changing `core.symlinks` alone does not recreate those links.

Install the development environment and hooks from the repository root:

```powershell
uv sync --locked --reinstall --group test --group dev
uv run pre-commit install --install-hooks
uv run pre-commit run -a
```

`uv` installs the selected Python version when needed. Use `uv run` to execute
tools in the environment; shell activation is optional. Markdownlint uses
Node.js through pre-commit. Docker is required for the MySQL integration tests.

Run the schema and utility tests without Docker:

```powershell
uv run --group test pytest -n 2 tests/schema tests/utilities
```

#### Docker prerequisites for integration tests

Install and start
[Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
with Linux containers enabled. Its WSL 2 backend requires hardware
virtualization and a supported WSL installation; follow Docker's installation
guide for the current system requirements. Run `ado` and pytest from Windows
PowerShell in the repository's Windows checkout.

Open a new PowerShell terminal after installing Docker Desktop so the Docker
CLI is available on `PATH`. Verify both the client and engine before testing:

```powershell
Get-Command docker
docker version
docker info --format '{{.OSType}}'
docker system info
```

`docker version` should show both Client and Server sections, and the operating
system check should print `linux`. Wait for Docker Desktop's engine to start
before continuing. Docker Desktop's integrated terminal is optional.

If `Get-Command docker` cannot find the CLI, add the `resources/bin` directory
under your Docker Desktop installation to the current terminal's `PATH`. For
the default per-user installation:

```powershell
$dockerBin = Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin'
$env:PATH = "$dockerBin;$env:PATH"
docker system info
```

For an all-users installation, the default CLI directory is
`C:/Program Files/Docker/Docker/resources/bin`.

The tests use Testcontainers to start and remove temporary MySQL containers
from `mirror.gcr.io/mysql:8`. Allow access to the Docker engine and image
registry. Docker maps MySQL to an available host port, so a separately installed
MySQL server is optional. Confirm the Python Docker client can reach the engine:

```powershell
uv run --group test python -c "import docker; print(docker.from_env().ping())"
```

This should print `True`. Run the full core suite with Docker Desktop running.
Use two workers to limit concurrent Ray and MySQL instances:

```powershell
uv run --group test pytest -n 2 tests/
```

The equivalent isolated Tox environment runs the core suite with locked
dependencies:

```powershell
uv run tox -e py311-locked-windows
```

The Windows Tox target defaults to two workers. Each worker starts a Ray
instance and may start a MySQL container. Use fewer workers to limit memory
usage, for example:

```powershell
uv run tox -e py311-locked-windows -- -n 1 tests/
```

To run just the Docker-free subset through Tox:

```powershell
uv run tox -e py311-locked-windows -- tests/schema tests/utilities
```

SFTTrainer tests and the full Linux/macOS plugin test matrix run on Linux or
macOS. The native Windows Tox target runs `tests/`.

### Windows with WSL2

WSL2 provides the Linux workflow for Windows contributors who need the full
plugin test matrix. Install a Linux distribution and enable Docker Desktop's
integration for that distribution, or provide another Docker-compatible runtime.
Clone inside the Linux filesystem, for example under `~/src/ado`.

Run these commands in the Linux shell, with Node.js available:

```shell
git clone https://github.com/IBM/ado.git ~/src/ado
cd ~/src/ado
uv sync --locked --reinstall --group test --group dev
uv run pre-commit install --install-hooks
docker system info
uv run pre-commit run -a
uv run tox -e py311-locked-linux
```

### Creating a development virtual environment

On native Windows, follow [Native Windows setup](#native-windows-setup-powershell).
The shell activation commands below are for Linux, macOS, and WSL2.

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
uv sync --locked --group test --group dev --reinstall
source .venv/bin/activate
```

<!-- markdownlint-disable no-blanks-blockquote -->

> [!NOTE]
>
> This installs `ado` in editable mode.

> [!NOTE]
>
> In line with uv's defaults, the `uv sync` command creates a `.venv` in the
> top-level of the project's repository. Note that environments created by
> `uv sync` **are intended only to be used when developing a specific project
> and should not be shared across projects.**

> [!CAUTION]
>
> `uv sync` ensures a reproducible development environment is created by using a
> lock-file, `uv.lock`. Only packages in the lockfile are installed, and other
> packages found in the virtual environment **will be deleted**. See
> [Making changes to dependencies](#making-changes-to-dependencies) for how to
> add packages to the lockfile.

<!-- markdownlint-enable no-blanks-blockquote -->

#### Using a custom location for the venv

If you want to create your development virtual environment at an alternate
location, $LOCATION, then run:

```commandline
uv venv $LOCATION
source $LOCATION/bin/activate
uv sync --active --group test --reinstall
```

## Code style

> [!NOTE]
>
> See the
> [Automating checks with pre-commit](#automating-checks-with-pre-commit)
> section to automate this.

This repository uses [`ruff format`](https://docs.astral.sh/ruff/formatter/) for
formatting.

You can format your code by:

- Manually running `ruff format tests/ ado/ plugins/`
- Using the
  [Ruff extension for VSCode](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
  and setting it as the default formatter:
  <https://docs.astral.sh/ruff/editors/settings/>

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

> [!CAUTION]
>
> **Using Prettier with MkDocs / MkDocs‑Material**
>
> Some MkDocs and MkDocs‑Material features rely on Markdown behavior that isn't
> part of the standard spec. Since Prettier reformats Markdown strictly
> according to the spec, it can rewrite certain patterns in ways that break
> rendering.
>
> Common issues include:
>
> - **Nested lists** that must stay indented with 4 spaces (Prettier collapses
>   to 2).
> - **Preprocessor tags** like `{%` that must not change.
> - **Components using `---` fences** (cards, metadata blocks) that get
>   converted to `***`.
> - **Admonitions** (`!!! info`, `> [!CAUTION]`): if there's no blank line after
>   the admonition header, Prettier may reflow the first content line into the
>   header, causing it to be rendered as the title.
> - **markdownlint-cli2**: `markdownlint-disable-next-line` may break if
>   Prettier inserts a blank line below it; use `markdownlint-disable` and
>   `markdownlint-enable` instead
>
> If a page uses these patterns, consider disabling Prettier for that section.
> **NOTE**: a blank line is required before the annotations for them to be
> picked up by Prettier:
>
> ```markdown
> <!-- prettier-ignore-start -->
>
> ...content...
>
> <!-- prettier-ignore-end -->
> ```

Line-too-long errors do not get automatically fixed by `markdownlint-cli2`. We
recommend using `prettier` to autoformat markdown in that case. The official
website provides instructions to:

- [Install prettier on your machine](https://prettier.io/docs/install).
- [Integrate prettier on your editor of choice](https://prettier.io/docs/editors).

Prettier can be run as a CLI tool with:

```commandline
npx prettier -w "**/*.md"
```

## Secret scanning

> [!NOTE]
>
> See the
> [Automating checks with pre-commit](#automating-checks-with-pre-commit)
> section to automate this.

This repository uses IBM's
[detect-secrets](https://github.com/ibm/detect-secrets) to scan for secrets
before the code is pushed to GitHub. The `detect-secret` CLI is automatically
installed as part of the [project setup](#project-setup).

To update the secrets database manually, run:

```commandline
detect-secrets scan --update .secrets.baseline
```

To audit detected secrets, use:

```commandline
detect-secrets audit .secrets.baseline
```

If the pre-commit hook raises an error but the audit command succeeds with just
`Nothing to audit!` then run `detect-secrets scan --update .secrets.baseline` to
perform a full scan and then repeat the `audit` command.

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

On one terminal, start serving the website using mkdocs (from the repo root):

```commandline
mkdocs serve --clean
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

```commandline
uv run pre-commit install --install-hooks
```

This command will configure pre-commit to run automatically before each commit,
highlighting any issues and preventing the commit if problems are found.
Run all checks manually with `uv run pre-commit run -a`. The Markdownlint hook
uses Node.js; Docker is required for MySQL integration tests.

### Handling pre-commit failures

1. **Ruff format failures**: try committing again, `ruff format` might have
   reformatted your code in-place.
   - If ruff format fails to format your code, your files have syntax errors.
     [Try manually running ruff format](#code-style).
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
   automatically. If you review its error message and still don't see a clear
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

## Verifying lockfile integrity

After making changes to dependencies or before committing changes, you should
verify that the `uv.lock` lockfile is synchronized with `pyproject.toml`. This
ensures that the lockfile accurately reflects the current project dependencies.

You can do this with:

```commandline
uv lock --check
```

If the check passes, no output is produced and the command exits with status
code 0.

### Fixing lockfile issues

If `uv lock --check` fails, it means the lockfile is out of sync with
`pyproject.toml`. To fix this, regenerate the lockfile by running:

```commandline
uv lock
```

After running this command, commit the updated `uv.lock` file along with your
other changes.
