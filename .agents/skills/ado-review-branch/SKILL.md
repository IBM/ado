---
name: ado-review-branch
description: >-
  Review committed changes in the current branch. Use when the user asks to
  review, audit, or critique a branch — covers code quality, test quality,
  skill quality, bugs, linting issues, and adherence to project guidelines.
  Produces a structured review report written to a markdown file.
metadata:
  user-invocable: "true"
  disable-model-invocation: "true"
---

# Review a branch

## General Rules

- Review the committed changes in the current branch
- Run mutation checks, linting, and tests in a detached worktree of `HEAD`, so
  uncommitted work is neither touched nor linted
  - create it with `git worktree add --detach "$REVIEW_DIR" HEAD`, where
    `REVIEW_DIR` is a fresh directory from `mktemp -d`
  - run `uv` commands from inside it
  - remove it with `git worktree remove --force "$REVIEW_DIR"` when finished
- Verify every claim by running it — a short script, a test, or a command that
  reproduces the behaviour — before reporting it. Reading the code is not
  verification; use it only when the claim cannot be executed.
- Validate claims against the target branch — the target is `main` unless the
  user names another. Use `git diff <target>...HEAD` for the scope,
  `git show <target>:<path>` to read the old code, and a second detached
  worktree of `<target>` (created and removed like the review worktree) to run
  the old behaviour. Use this to verify whether a behaviour or bug is new or
  pre-existing; report pre-existing ones as Observations.
- Report every deviation from the guidelines, including Minor ones
- Write the review to `<branch_name>-review.md` at the root of the main
  checkout (not a review worktree, which is removed) when finished, where
  `<branch_name>` is the current branch name with `/` replaced by `-`, unless
  the user names another path

### Rules for findings

- One problem per finding — keep each finding strictly focused on a single
  distinct issue. Do not combine behavioural bugs with doc omissions or commit
  message critiques.
- For `bug` findings, focus on the problem — describe the functional and
  behavioural defect from the perspective of callers, users, or system
  contracts; do not narrate internal code mechanics or execution flow.
- For `bug` findings that are regressions, contrast behaviour — state it before
  and after: "Before this PR, [behaviour on target branch]. This PR changes it
  so that [new behaviour], which [why it is a problem]."
- Attribute mutations — when a mutation made during verification introduced a
  change or failure, state it explicitly so it is not confused with PR code.
- Provide a clear fix — propose a concrete, actionable fix for every finding.
  Issues without one are Observations.
- Limit length and formatting:
  - report only problems; do not praise or describe correct code
  - report each issue once, under one heading
  - limit each finding to three sentences: the problem, why it matters if not
    obvious, and the proposed fix
  - anchor location once per finding as a single markdown link
    (`[file:line](path:line)`); refer to additional locations as plain
    `path:line` text (for `docs/`, link the page)
  - use code formatting only for paths, commands, and ambiguous identifiers
  - do not link or code-format the same name more than once per finding
  - avoid fenced code blocks unless a short snippet is the clearest fix

## Code Review

Evaluate changes to code against the guidelines in
[AGENTS.md](../../../AGENTS.md) and
[ado-develop-plugin](../ado-develop-plugin/SKILL.md).

### Code quality checklist

Read [AGENTS.md](../../../AGENTS.md) in full and check every changed Python
file against all rules it contains. Pay particular attention to these
commonly missed areas:

- **Fixture reuse** — see the fixture rules in AGENTS.md; report violations as
  `test` findings.
- **Fixture placement** — a fixture used only within one subdirectory but
  defined in a parent `conftest.py` is misplaced; report it as a `test`
  finding.
- **Naming** — see the naming rules in AGENTS.md; report violations as
  `design` findings.
- **Docstring accuracy** — for every changed function or method, read the
  implementation body and verify the docstring matches what the code actually
  does. A docstring that describes behaviour the function no longer has, or
  omits behaviour it has gained, is stale; report it as a `docs` finding even
  if the docstring was not touched in this diff.
- **Private helpers** — every new private function or method must satisfy the
  helper rules in AGENTS.md; report violations as `design` findings. The fix is
  to inline the helper.
- **Nesting depth** — an `if` whose condition could be inverted to return,
  continue, or raise early and remove a level of indentation (a guard clause)
  is a `design` finding at Minor severity; report it only when the inversion
  clearly improves readability.

### Test quality

For every test added or modified in the branch, read the test body and the code
it exercises, and check it against the test rules in AGENTS.md. Report
violations as `test` findings, and additionally:

- **Redundant tests** — search `tests/` for existing tests of the same code
  path. Name the existing test in the finding and suggest removing the new one
  or extending the existing one.
- **Missing tests** — for each changed behaviour whose regression would
  matter, verify that at least one test would fail if that behaviour were
  reverted. Do not report a missing test for code AGENTS.md says not to test.
- **Verify by mutation** — when a test looks weak, confirm it by temporarily
  breaking the code under test and running that test; do not report a test as
  banal on appearance alone. Edit only files in the review worktree and restore
  each with `git restore <file>` afterwards.

### Plugin versioning

For any change under `plugins/`, check the versioning rules in
[ado-develop-plugin](../ado-develop-plugin/SKILL.md#plugin-versioning) and
report violations with these severities:

- `VERSION` not bumped for a bug fix, new feature, or breaking change: medium
- `version=` argument in `@characterize_operation` (or equivalent) differs from
  `VERSION`: critical

### Linting

Run `uv run pre-commit run -a` in the review worktree and report each failure
as a finding tagged `lint`. These issues block commits.

Several hooks rewrite files (for example `ruff-format`, `yamlfmt`, and
`markdownlint-cli2-docker --fix`). Treat those edits as mutations of the
worktree, not as branch code. The `markdownlint-cli2-docker` hook needs Docker;
if it cannot run, list it under checks not run in the report header.

## Skill Review

If the branch adds or changes anything under `.agents/skills/`, read
[references/skill-review.md](references/skill-review.md) and follow it for each
new or modified skill.

## Report Format

Write the report with these sections, omitting any that are empty:

1. **Header** (2-4 lines)
   - scope: the commits reviewed and the size of the diff
   - checks that were not run, and why
   - one or two sentences on what the change does, only if the commit messages
     do not already say it
2. **Summary of Issues** — at most two sentences naming the highest-severity
   themes; do not list individual findings
3. **Findings** — a single list ordered by severity under the headings
   Critical, Medium, and Minor
   - default severity when none is specified elsewhere: `bug` Critical,
     `lint`, `test`, and `docs` Medium, `design` Minor
   - number findings continuously (1, 2, 3, ...) so they can be referenced
   - start each finding with a bold heading of the form `tag: short title`,
     where the tag is one of `bug`, `design`, `test`, `docs`, or `lint`
   - test gaps and documentation gaps are findings with the matching tag, not
     separate sections
4. **Observations** — issues that cannot be tied to a specific location,
   concerns without a clear fix, or pre-existing bugs identified during the
   review, so that tracking issues can be opened
   - number them continuously after the findings
   - give each a title, the location if known, and why no fix is proposed
5. **Skill Reviews** — one subsection per new or modified skill, using the
   per-skill structure in [references/skill-review.md](references/skill-review.md)
6. **Verdict** (1-2 lines) — merge as is, merge after the Critical findings are
   fixed, or do not merge; do not restate the findings

Example findings (file paths and line numbers are illustrative):

```markdown
### Critical

1. **bug: duplicate identifiers silently shadow legacy experiments** —
   [experiments.py:59](ado/example/experiments.py:59) Before this PR, when a
   legacy and a new experiment shared an identifier, the legacy experiment was
   loaded. This PR makes the last decorated experiment win, so legacy
   experiments are silently replaced; restore legacy precedence or raise on
   duplicate identifiers.

### Minor

2. **design: private helper called from a single place** —
   [create.py:120](ado/example/create.py:120) `_format_error` wraps one call and
   is used only at `ado/example/create.py:98`. Inline it.
```
