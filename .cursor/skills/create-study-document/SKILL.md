---
name: create-study-document
description: >
  Create and update ado study documents (document resources) that track an
  in-progress study: motivation, question, objectives, materials, study labels,
  and a todo list of next steps. Use when starting a study, creating a study
  document, updating study todos/objectives, or when the user mentions
  study-$ID or study tracking.
---

# Creating a Study Document

A **study document** describes a study to perform, or already underway, with
ado. It is stored as a `document` resource in the metastore.

For generic document create/query, see
[resource-yaml-creation — Document](../resource-yaml-creation/SKILL.md#document).
For CLI syntax, see [using-ado-cli](../using-ado-cli/SKILL.md).

## Naming and metadata

`metadata.name` must be `study-$ID` where `$ID` is the short study slug
(e.g. `study-cplex-mip`).

Required `metadata` fields:

- `name` — `study-$ID`
- `description` — one-line summary of the study
- `todo` — YAML list of current next-step strings

Leave `relatedResources` empty. Associate spaces and operations with the study
via **labels** only (avoids colliding with op/space report queries on
`relatedResources`).

## Content sections

Use this reading order. Exact heading names matter for consistency.

### Required

- `## Study question` — what the study aims to learn (investigable questions)
- `## Study objective` — current campaigns / methods to answer those questions
- `## Study labels` — labels that **must** be applied to spaces and operations
  in this study (include at least `study: $ID`)

### Optional

- `## Motivation` — why the study exists (stakeholder framing; keep stable).
  Place **before** Study question when present.
- `## Materials` — instances, formulations, input files, and related paths.
  Place **after** Study objective and **before** Study labels when present.
  Prefer a short summary here and link out for long formulations.

Do not fold motivation into Study question, or formulation/instance detail into
Study objective.

## Template

```yaml
# study-$ID_document.yaml  (temp file, not committed)
metadata:
  name: study-cplex-mip
  description: Compare TPE vs LHS for CPLEX MIP gap on bab6
  todo:
    - Run TPE baseline (60m, 200 trials)
    - Analyse mip-gap histograms
content: |
  ## Motivation

  Establish a well-tuned classical baseline for this problem class.

  ## Study question

  Which sampler finds better MIP gaps faster on bab6?

  ## Study objective

  Run comparable TPE and LHS campaigns; compare mip-gap and solve-time
  distributions.

  ## Materials

  - Instance: bab6.mps (MIPLIB 2017)
  - Paths: examples/cplex_mip_bab6/

  ## Study labels

  - study: cplex-mip
```

Omit Motivation and/or Materials when they add nothing.

```bash
uv run ado create document -f study-cplex-mip_document.yaml --dry-run
uv run ado create document -f study-cplex-mip_document.yaml
```

Apply the study labels to every space and operation created for the study
(see [resource-yaml-creation](../resource-yaml-creation/SKILL.md) metadata).

## Query and update

```bash
# Find a study document by name
uv run ado get document -q 'config.metadata.name=study-$ID' --details

# List documents and select names matching study-*
uv run ado get document --details

# Spaces / operations in the study
uv run ado get spaces -l study=$ID --details
uv run ado get operations -l study=$ID --details
```

Fetch the body with `uv run ado get document DOCUMENT_ID -o yaml`.

**Updates:**

- Prefer `ado edit document DOCUMENT_ID` for `todo` and `description`.
- Full body refresh: create a replacement only after the user agrees; same
  keep-history vs replace policy as project reports in
  [examining-ado-project](../examining-ado-project/SKILL.md).

## Related

- [conduct-empirical-study](../conduct-empirical-study/SKILL.md) — create or
  refresh a study document when running a study
- [examining-ado-project](../examining-ado-project/SKILL.md) — surfaces study
  documents in project overviews
- **Future-friendly** (not yet updated): `ado-project-maintenance` should treat
  `study-*` docs separately from duplicate per-resource reports;
  examining-ops/spaces should ignore `study-*` when matching reports via
  `relatedResources`
