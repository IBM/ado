---
name: generate-release-notes
description: >-
  Generate a structured changelog for GitHub release notes from changes since
  the last release. Use when the user asks to generate release notes, a
  changelog, or a summary of changes since the last tag/release.
---

# Generate Release Notes

Produce a concise, well-organized GitHub release notes changelog from the
changes since the last release using cocogitto (`cog`).

## Step 1 — Find the Latest Semver Tag

Run the following to identify the most recent semver tag for the main package
(plain `MAJOR.MINOR.PATCH` form, not component-prefixed):

```bash
git tag --sort=-version:refname | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | head -1
```

Store this as `$LATEST_SEMVER_TAG`.

## Step 2 — Fetch the Raw Changelog

Run cocogitto to get the full conventional-commit changelog since that tag:

```bash
cog changelog $LATEST_SEMVER_TAG..
```

The output may include these section headings:

- `#### Features`
- `#### Bug Fixes`
- `#### Performance Improvements`
- `#### Refactoring`
- `#### Documentation`
- `#### Build system`
- `#### Tests`
- `#### Miscellaneous Chores`

Each entry is formatted as:
`- (**scope**) description (#PR) - (shortHash) - Author`

Breaking changes have an HTML `<span>` badge before the scope:
`- <span ...>BREAKING</span>(**scope**) description …`

## Step 3 — Resolve Full Commit Hashes

The Complete changes section (Step 6) links each commit hash to GitHub. Build
a short-to-full-hash lookup for all commits in range:

```bash
git log --format="%h %H" $LATEST_SEMVER_TAG.. | head -500
```

The GitHub repo is `https://github.com/IBM/ado`. A commit link looks like:
`([abc1234](https://github.com/IBM/ado/commit/FULLHASH))`

## Step 4 — Find the Latest Component Tags

Run the following to get the latest tag for each component:

```bash
git tag --sort=-version:refname \
  | grep -E \
    '^(autoconf|sfttrainer|vllm-performance|trim|cplex-mip|anomalous-series|ray-tune|profile-space|example-actuator)/'
```

For each component, the latest tag has the form `<component>/<version>`.

## Step 5 — Compose the Synthesized Changelog

The main changelog body uses **thematic prose summaries**, not one bullet per
commit. Group related commits into bold-labelled bullets within each section.
Write from the perspective of a user reading the release notes: what changed,
why it matters.

### Source entries for general sections

The general sections cover **only** entries whose scope is **not** a component
plugin. The complete list of component scopes to exclude is:

- `autoconf`
- `sfttrainer`, `sft_trainer`
- `vllm_performance`, `vllm-performance`
- `trim`
- `cplex-mip`, `cplex_mip`
- `anomalous-series`, `anomalous_series`
- `profile-space`, `profile_space`
- `example-actuator`, `example_actuator`

**Do not mention any of these components, their features, or their fixes
anywhere in the general sections.** If an entry's scope is in the list above,
it belongs exclusively in the component section after `---`.

### No PR links, hashes, or author names in synthesized sections

The synthesized sections (Highlights through Refactoring, and each component
section) contain **only plain prose**. Do not include PR numbers (`#NNN`),
commit hashes, or author names anywhere in these sections — GitHub renders
`#NNN` as an issue link, which is undesirable here. Those details appear
exclusively in `## Complete changes`.

### Output structure

Produce these sections in order, omitting any with no content:

1. `## 🏅 Highlights` — 2–4 bullets calling out the most impactful themes
   across the whole release (cross-cutting improvements, major new
   capabilities, significant breaking changes). This section always comes
   first. Draw only from non-component changes.
2. `## ✨ Features` — new user-visible capabilities from non-component scopes,
   grouped by area (e.g. **CLI enhancements**, **Core improvements**)
3. `## 🐛 Fixes` — bug fixes from non-component scopes, grouped by area
4. `## ⚡ Performance` — performance improvements from non-component scopes
5. `## 🧰 Build` — build system, packaging, dependency changes from
   non-component scopes
6. `## 📝 Docs` — documentation updates from non-component scopes
7. `## 🧹 Refactoring` — internal refactoring from non-component scopes.
   List any **Breaking changes** explicitly with a sub-bullet or inline note.

After all general sections, add a `---` horizontal rule, then a dedicated
section for each component that has entries, in this order:

- `## ⚙️ Autoconf X.Y.Z` (scopes: `autoconf`)
- `## 🧠 SFTTrainer X.Y.Z` (scopes: `sfttrainer`, `sft_trainer`)
- `## 📈 vLLM Performance X.Y.Z` (scopes: `vllm_performance`,
  `vllm-performance`)
- `## ✂️ TRIM X.Y.Z` (scope: `trim`)
- `## 🧮 CPLEX-MIP X.Y.Z` (scopes: `cplex-mip`, `cplex_mip`)
- `## 🚨 Detect Anomalous Series X.Y.Z` (scopes: `anomalous-series`,
  `anomalous_series`)
- `## 🎛️ Profile Space X.Y.Z` (scopes: `profile-space`, `profile_space`)
- `## 📋 Example Actuator X.Y.Z` (scopes: `example-actuator`,
  `example_actuator`)

Replace `X.Y.Z` with the version from the component's latest tag (Step 4).
If no tag exists for the component, omit the version from the header.

**Only include a component section if it has at least one non-trivial entry.**
If a component has no entries at all, or only filtered-out entries (deps,
hooks, DRL-NextGen), omit the section entirely — do not include an empty
section or a header with no bullets.

Component sections use synthesized prose bullets (not raw commit lines), with
no PR numbers, hashes, or author names.

### Filtering

Exclude these from all synthesized sections:

- Routine dependency bumps (scope `deps`, `hooks`, description
  `update dependencies` / `update pre-commit hooks` with no further detail)
- Automated changelog/release commits (author `DRL-NextGen`)
- Test-only infrastructure changes with no user-visible impact
- `#### Miscellaneous Chores` entries (omit entirely from synthesis)

### Breaking changes in Refactoring

cog marks breaking changes with an HTML `<span>BREAKING</span>` badge. In the
synthesized `## 🧹 Refactoring` section, call these out explicitly.

## Step 6 — Append Complete Changes

After the component sections, add another `---` horizontal rule, then a
`## Complete changes` section containing the **full verbatim cog output**,
with one modification: replace each short hash `(abc1234)` with a linked
version `([abc1234](https://github.com/IBM/ado/commit/FULLHASH))` using the
lookup built in Step 3.

Also replace the HTML `<span ...>BREAKING</span>` badge with the GitHub-badge
form: `![BREAKING](https://img.shields.io/badge/BREAKING-red)` so it renders
on GitHub.

## Step 7 — Wrap in Quadruple Backticks

Wrap the entire output from Steps 5–6 in a quadruple-backtick markdown code
block so it can be pasted directly into GitHub:

````markdown
## 🏅 Highlights

…

---

## ⚙️ Autoconf X.Y.Z

…

---

## Complete changes

…
````
