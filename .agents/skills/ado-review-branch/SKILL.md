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
- Do not assume — verify claims by reading the relevant code or running a short
  script. Never report a finding based on what code looks like it might do;
  confirm it first.
- Validate claims against the target branch — check the branch the PR targets
  (e.g. `main`) to verify whether a behaviour or bug is new or pre-existing.
  If pre-existing, note it under Observations so an issue can be opened.
- Be picky: report every deviation from the guidelines, including minor ones
- Write the review to `<branch_name>-review.md` at the repo root when finished,
  where `<branch_name>` is the current branch name with `/` replaced by `-`,
  unless the user names another path

### Rules for findings

- Focus on the problem — describe the functional and behavioural defect from the
  perspective of callers, users, or system contracts; do not narrate internal
  code mechanics or execution flow.
- One problem per finding — keep each finding strictly focused on a single
  distinct issue. Do not combine behavioural bugs with doc omissions or commit
  message critiques.
- Contrast regressions clearly — for regressions, state behaviour before and
  after: "Before this PR, [behaviour on target branch]. This PR changes it so
  that [new behaviour], which [why it is a problem]."
- Attribute mutations — when a mutation made during verification introduced a
  change or failure, state it explicitly so it is not confused with PR code.
- Provide a clear fix — propose a concrete, actionable fix for every finding.
  If an issue lacks a specific location or clear fix, place it under
  Observations.
- Limit length and formatting:
  - report only problems; do not praise or describe correct code
  - limit each finding to three sentences: the problem, why it matters if not
    obvious, and the proposed fix
  - anchor location once per finding as a single markdown link
    (`[file:line](path:line)`); refer to additional locations as plain
    `path:line` text (for `docs/`, link the page)
  - use code formatting only for paths, commands, and ambiguous identifiers
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
  each with `git restore <file>` afterwards. Be clear in your finding when a
  mutation you made introduced a change or failure.

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
2. **Summary of Issues** — a high-level overview explaining the key problems and
   regressions discovered across the branch in plain, accessible terms before
   diving into detailed findings
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
   review, described clearly so that tracking issues can be opened
5. **Skill Reviews** — one subsection per new or modified skill, using the
   per-skill structure in [references/skill-review.md](references/skill-review.md)
6. **Verdict** (1-2 lines) — merge as is, merge after the Critical findings are
   fixed, or do not merge; do not restate the findings

Example findings:

```markdown
1. **design: new module-level registry inverts experiment loading order** —
   [custom_experiments.py:59](ado/modules/actuators/custom_experiments.py:59)
   allows only one custom experiment per identifier. Before this PR, if a legacy
   experiment and new experiment shared an identifier, the legacy experiment was
   loaded; this PR inverts that precedence so the last decorated experiment
   silently overwrites earlier ones. Restore legacy precedence or raise on
   duplicate identifiers.

2. **bug: exit swallowed by broad except** —
   [create.py:160](ado/cli/resources/operation/create.py:160) catches
   Exception, changing behaviour from re-raising typer.Exit (on main) to
   swallowing it and printing a generic failure message. Re-raise typer.Exit
   before the except clause.
```
