# Contributing

## Contributing In General

Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

To contribute code or documentation, please submit a
[pull request](https://github.com/ibm/ado/pulls).

A good way to familiarize yourself with the codebase and contribution process is
to look for and tackle low-hanging fruit in the
[issue tracker](https://github.com/ibm/ado/issues). Before embarking on a more
ambitious contribution, please quickly [get in touch](#communication) with us.

**Note: We appreciate your effort, and want to avoid a situation where a
contribution requires extensive rework (by you or by us), sits in backlog for a
long time, or cannot be accepted at all!**

### Proposing new features

If you would like to implement a new feature, please
[raise an issue](https://github.com/ibm/ado/issues) before sending a pull
request so the feature can be discussed. This is to avoid you wasting your
valuable time working on a feature that the project developers are not
interested in accepting into the code base.

### Fixing bugs

If you would like to fix a bug, please
[raise an issue](https://github.com/ibm/ado/issues) before sending a pull
request so it can be tracked.

### Merge approval

The project maintainers use LGTM (Looks Good To Me) in comments on the code
review to indicate acceptance. A change requires LGTMs from one of the
maintainers.

For a list of the maintainers, see the [MAINTAINERS.md](MAINTAINERS.md) page.

## Legal

Each source file must include a license header for the MIT License. Using the
SPDX format is the simplest approach. e.g.

```text
# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
```

<!-- markdownlint-disable line-length -->

We have tried to make it as easy as possible to make contributions. This applies
to how we handle the legal aspects of contribution. We use the same approach -
the
[Developer's Certificate of Origin 1.1 (DCO)](https://github.com/hyperledger/fabric/blob/master/docs/source/DCO1.1.txt) -
that the Linux® Kernel
[community](https://elinux.org/Developer_Certificate_Of_Origin) uses to manage
code contributions.

<!-- markdownlint-enable line-length -->

We simply ask that when submitting a patch for review, the developer must
include a sign-off statement in the commit message.

Here is an example Signed-off-by line, which indicates that the submitter
accepts the DCO:

```text
Signed-off-by: John Doe <john.doe@example.com>
```

You can include this automatically when you commit a change to your local git
repository using the following command:

```commandline
git commit -s
```

## Communication

You can get in touch with us by starting a
[discussion](https://github.com/IBM/ado/discussions) on GitHub.

## Setup

To set up your development environment, follow the instructions in our
[development guide](https://ibm.github.io/ado/getting-started/developing) file.

## Testing

We use [Tox](https://github.com/tox-dev/tox) to run unit tests for our code. To
run tests for Python 3.10, you can run the following command:

```commandline
export TOX_ENV=py310-test-pipenv-optimizers
tox --colored yes --stderr-color RESET -r -e "$TOX_ENV" -vvv
```

Similarly, you can test different Python versions by changing `py310` to `py311`
or `py312`.

## Commit and PR title guidelines

We require commits and PR titles to conform to the
[conventional commits standard](https://www.conventionalcommits.org/en/v1.0.0/)
and follow the
[Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines).
In a nutshell, the commit message should be structured as follows:

> [!TIP]
>
> It is highly recommended to include the scope in the PR title and in all the
> commits

```plaintext
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Where `type` is one of the following:

- **build**: Changes that affect the build system (e.g., pyproject.toml files,
  Dockerfiles, etc.) or external dependencies.
- **ci**: Changes to CI-related configuration files and scripts
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **style**: Changes that do not affect the meaning of the code (white-space,
  formatting, etc)
- **test**: Adding missing tests or correcting existing tests
- **chore** (discouraged): Minor changes that don't fit in other categories

## Coding style guidelines

We require code and markup to adhere to certain rules. We enforce these rules
though the tools we mention in
[our development guide](https://ibm.github.io/ado/getting-started/developing),
namely:

- [Black](https://ibm.github.io/ado/getting-started/developing#code-style)
- [Ruff](https://ibm.github.io/ado/getting-started/developing#linting-code-with-ruff)
- [Copywrite](https://ibm.github.io/ado/getting-started/developing#copyright-and-license-headers)
- [Markdownlint-cli2](https://ibm.github.io/ado/getting-started/developing#linting-markdown-with-markdownlint-cli2)
- [Yamlfmt](https://github.com/google/yamlfmt)

To verify that your code conforms to these rules you can run the following
commands:

```commandline
black --check . --extend-exclude website
ruff check --exclude website
copywrite headers --plan
markdownlint-cli2 "**/*.md" "#.venv"
detect-secrets scan --update .secrets.baseline
detect-secrets audit .secrets.baseline --fail-on-unaudited --fail-on-live --fail-on-audited-real
yamlfmt -lint .
```

## Website checks

To minimize broken links, we use linkcheck to spot issues before things are
merged. After you've installed it and have started serving the website
[using the instructions provided](https://ibm.github.io/ado/getting-started/developing#website-link-checking),
you can check for broken links with:

```commandline
linkcheck http://127.0.0.1:8000/ado/ \
                --skip-file=.linkchecker_skip
```
