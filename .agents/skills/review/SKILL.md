---
name: review
description: >-
  Review committed changes in the current branch. Use when the user asks to
  review, audit, or critique a branch — covers code quality, skill quality,
  bugs, linting issues, and adherence to project guidelines. Produces a
  structured review report written to a markdown file.
metadata:
  user-invocable: true
  disable-model-invocation: true
---

# review

## General Rules

- Review the committed changes in the current branch
- Be picky
- Write review to md file when finished
- Link review comments to the specific file and line the comment pertains to
  - this includes references to website pages
  - for website pages create markdown links, not just plain section names

## Code Review

When evaluating changes to code evaluate against the guidelines in
[AGENTS.md](../../AGENTS.md) and
[plugin-development.mdc](../rules/plugin-development.mdc)

### Plugin versioning

For any PR that touches a plugin directory (`plugins/`):

- If the change is a bug fix, new feature, or breaking change, verify `VERSION`
  was bumped (patch / minor / major respectively per semver). Report it as a
  medium issue if it was not.
- If the plugin has a `version=` argument in a `@characterize_operation` (or
  equivalent) decorator, verify it matches `VERSION`. Report a critical issue if
  it does not.

- Format the review report with these sections (omit if not relevant):
  - Overview: What is changed
  - Bugs: Any potential bugs
  - Skill Reviews: One subsection per changed/new skill using the structure
    above
  - Code Critical Issues: Critical problems with code implementation or
    structure
  - Code Medium Issues
  - Linting Issues: Issues raised by ruff format, ruff check, or
    markdownlint-cli2 (these block commits)

## Skill Review

Agent skills are text documents under `.agents/skills/`.

- Evaluate based on [Agent Skills Guidelines](../../AGENTS.md#agent-skills).
- For each skill (new or modified) ensure you check the following for related
  content
  - `.agents/skills/`, `examples/`, `docs/`, `AGENTS.md`

When evaluating new or modified skills (new or changes) use this structure per
skill:

1. **What the skill adds** — one/two sentence summary
2. **Related existing documentation** - a bullet list outlining related skills,
   examples and website docs, with a note on why they are related
3. **New information** — bullet list of genuinely new content not already in
   other skills, AGENTS.md, plugin-development.mdc, or linked docs. **Before
   assessing this:**
   - Read every file listed in the skill's References section
   - Check descriptions of all skills and identify any whose scope overlaps with
     the skill under review
4. **Critical Issues**
   - Subdivide this section into "Duplications" and "Misplacements"
   - For additions that duplicate existing content:
     - name the section
     - state what it duplicates
     - give the exact link or section to replace it with
       - or state "remove — covered via general link to X")
   - For new information that is misplaced:
     - This can be content that is outside the scope of the skill's
       `description` field
     - It could also be content that is does not match the objective of a skill
     - state where it should be moved to (another skill, AGENTS.md,
       plugin-development.mdc, etc.)
5. **Medium Issues**
   - In this section include issues like the following:
     - Inline code that should link to an existing example file instead
     - Formatting problems, structural issues, or misleading text
     - Mistakes in code examples
     - Inconsistencies with other skill files or agent files
     - File or directory paths referenced in the skill that do not exist in the
       repo
6. **Summary**
   - Summarize the review.
   - Be clear if the skill should be (a) largely kept as is; (b) requires major
     changes (c) is not required
   - If major changes are required provide a sketch of what should be in the
     revised skill
