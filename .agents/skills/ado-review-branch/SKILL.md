---
name: ado-review-branch
description: >-
  Review committed changes in the current branch. Use when the user asks to
  review, audit, or critique a branch — covers code quality, skill quality,
  bugs, linting issues, and adherence to project guidelines. Produces a
  structured review report written to a markdown file.
metadata:
  user-invocable: true
  disable-model-invocation: true
---

# Review a branch

## General Rules

- Review the committed changes in the current branch
- Be picky: report every deviation from the guidelines, including minor ones
- Be concise
  - report only problems; do not describe code that is correct
  - report each issue once, under one heading
  - limit each finding to three sentences: the problem, why it matters if not
    obvious, and the fix
  - drop any finding that cannot be tied to a location and a fix
- Write the review to a markdown file when finished
- Anchor each review comment to the file and line it refers to
  - give the location once per comment as a single markdown link to the
    primary file and line
  - refer to any further locations in that comment as plain `path:line` text
  - for pages under `docs/`, use a markdown link to the page, not a plain
    section name
- Keep formatting light
  - use code formatting only for file paths, commands, and identifiers that
    would be ambiguous in prose; write other terms as plain text
  - do not add fenced code blocks unless a short snippet is the clearest way to
    show a proposed fix
  - do not link or code-format the same name more than once per comment

## Code Review

Evaluate changes to code against the guidelines in
[AGENTS.md](../../../AGENTS.md) and
[ado-develop-plugin](../ado-develop-plugin/SKILL.md).

### Plugin versioning

For any change under `plugins/`, check the versioning rules in
[ado-develop-plugin](../ado-develop-plugin/SKILL.md#plugin-versioning) and
report violations with these severities:

- `VERSION` not bumped for a bug fix, new feature, or breaking change: medium
- `version=` argument in `@characterize_operation` (or equivalent) differs from
  `VERSION`: critical

### Linting

Run `uv run pre-commit run -a` and report each failure as a finding tagged
`lint`. These issues block commits.

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
2. **Findings** — a single list ordered by severity under the headings
   Critical, Medium, and Minor
   - number findings continuously (1, 2, 3, ...) so they can be referenced
   - start each finding with a short title and a tag: `bug`, `design`, `test`,
     `docs`, or `lint`
   - test gaps and documentation gaps are findings with the matching tag, not
     separate sections
3. **Skill Reviews** — one subsection per new or modified skill, using the
   per-skill structure in [references/skill-review.md](references/skill-review.md)
4. **Verdict** (1-2 lines) — merge as is, merge after the Critical findings are
   fixed, or do not merge; do not restate the findings

Example finding:

```markdown
3. **Exit swallowed by broad except** `bug` —
   [create.py:160](ado/cli/resources/operation/create.py:160) catches
   Exception, which includes typer.Exit, so the specific error is followed by a
   generic failure message. Re-raise typer.Exit before the except clause.
```
