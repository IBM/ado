---
name: ado-project-maintenance
description: >-
  Delete ado resources already marked for cleanup, then scan a project's
  metastore for stale/redundant resources (empty stale spaces, failed/error
  operations, superseded operator runs, orphaned datacontainers, duplicate
  report documents, resources labeled provisional, pre-release operator
  versions), mark newly-found candidates, report them for review, and review
  existing labels/descriptions for accuracy. Use when asked to clean up,
  tidy, or run maintenance on an ado project/context; to find what can be
  deleted; or to review/curate resource metadata. For guidance on labeling a
  resource as provisional (testing/debug/temporary) at creation time, see
  resource-yaml-creation instead.
---

# ado Project Maintenance

Identifies and removes stale/redundant resources from an ado project's
metastore, and reviews existing metadata for accuracy. This skill is scoped
to **cleanup and review only** — for guidance on labeling a resource as
`provisional` at creation time, see
[resource-yaml-creation](../resource-yaml-creation/SKILL.md#the-provisional-label)
instead (see [Scope split](#scope-split-with-resource-yaml-creation)).

Run all commands from the repository root with `uv run` (see
[using-ado-cli](../using-ado-cli/SKILL.md)).

## Workflow

Each invocation is self-contained and ordered:

1. **[Step 1 — delete previously-marked resources](#step-1-detail--delete-previously-marked-resources)**
   — act on resources left marked `for_deletion` since a prior run (the
   user's implicit approval).
2. **[Step 2 — mark new candidates](#step-2-detail--mark-new-candidates)** —
   scan the project against the
   [conditions](#conditions-that-qualify-a-resource-for-marking) below and
   label any newly-qualifying resource. Never deletes anything — even the
   "always qualifies" conditions are only marked, not deleted, on first
   discovery.
3. **[Step 3 — report](#step-3-detail--report)** — present what was newly
   marked in Step 2 and how to unmark it before the next run deletes it.
4. **[Step 4 — review existing labels/descriptions](#step-4-detail--review-existing-labelsdescriptions)**
   — surface metadata that looks stale or missing, as suggestions only.

This means each run cleans up what was approved last time and produces a
fresh review list for next time.

## Label & condition definitions

Three labels make up the maintenance scheme. `provisional` is set
proactively by users/agents at creation time and is **defined in
[resource-yaml-creation](../resource-yaml-creation/SKILL.md#the-provisional-label)**
— this skill only consumes it (condition 8). `for_deletion` and
`deletion_reason` are written by this skill.

- **`provisional: <reason>`** (defined in resource-yaml-creation) — values
  consumed by condition 8: `testing`, `debug`, `temporary`. A resource
  carrying it is auto-marked for deletion during Step 2 without needing to
  satisfy any of the other conditions.
- **`for_deletion: "true"`** — marker label applied during Step 2, queryable
  via `ado get $TYPE -l for_deletion=true`.
- **`deletion_reason: <short-code>`** — audit trail recording why a resource
  was marked, one of: `empty-space-stale`, `failed-no-entities`,
  `error-exit-state`, `duplicate-operator-major-version`,
  `orphaned-datacontainer`, `duplicate-report`, `duplicate-project-report`,
  `provisional-testing`, `provisional-debug`, `provisional-temporary`,
  `prerelease-operator-version`.

Mark a resource:

```bash
uv run ado edit TYPE ID -p "labels: {for_deletion: 'true', deletion_reason: <code>}"
```

**Unmarking**: label patches merge as `old | new` — they only add/overwrite
keys, they cannot remove one (`ado/cli/utils/resources/handlers.py`). To
unmark a resource the user disagrees with, use an interactive edit and
delete the `for_deletion`/`deletion_reason` lines:

```bash
uv run ado edit TYPE ID --editor
```

(Leave `provisional` alone unless it no longer applies — see
[Step 4](#step-4-detail--review-existing-labelsdescriptions).)

### Conditions that qualify a resource for marking

1. **Stale empty spaces**: no measured entities, older than a week.

   ```bash
   uv run ado get spaces -o stats --details --output-file spaces-stats.txt
   ```

   Filter rows where `MEASURED_ENTITIES == 0`. There is no CLI age-range
   filter (`--filter` only supports equality/`JSON_CONTAINS` containment,
   not comparisons — see `docs/resources/metastore.md`), so check `created`
   client-side for candidates:

   ```bash
   uv run ado get space SPACE_ID -o yaml --output-file SPACE_ID.yaml
   ```

   Mark if `created` is more than 7 days ago. Reason: `empty-space-stale`.

2. **Failed operations that sampled no entities**: operation finished with
   `exit_state: fail` and made zero measurement requests.

   ```bash
   uv run ado show stats operation \
     --filter 'status=[{"event":"finished","exit_state":"fail"}]' \
     -o csv --output-file operations-fullstats.csv
   ```

   Filter rows where `TOTAL_REQUESTS == 0`. Reason: `failed-no-entities`.

3. **Error-state operations**: any operation that finished with
   `exit_state: error` qualifies — no entity-count check needed.

   ```bash
   uv run ado get operations \
     --filter 'status=[{"event":"finished","exit_state":"error"}]' --details
   ```

   Reason: `error-exit-state`.

4. **Superseded operator runs**: multiple operations applying the same
   operator to the same inputs, at different versions.

   Group operations by `operatorIdentifier` (`ado/core/operation/resource.py`
   — form `name@MAJOR.MINOR.PATCH`, strict release semver only) plus input
   space (`config.spaces`) and `config.operation.parameters`:

   ```bash
   uv run ado get operations -o yaml --output-file operations.yaml
   ```

   Within each (operator name, inputs) group, cluster by MAJOR version; in
   any cluster with more than one operation, keep the highest MINOR.PATCH
   and mark the rest. Reason: `duplicate-operator-major-version`. There is
   no single CLI filter that performs this grouping — do it as a
   script/manual pass over the YAML dump.

5. **Orphaned datacontainers**: datacontainers belonging to an operation
   that qualifies under conditions 2-4 (whether newly marked or already
   marked). Datacontainers are children of operations
   (`ado/metastore/sql/statements.py`), so this must run after operations
   are identified:

   ```bash
   uv run ado show related operation OP_ID
   ```

   Mark any datacontainer returned. Reason: `orphaned-datacontainer`.

6. **Duplicate per-resource reports**: more than one `document` report
   related to the same space or operation.

   ```bash
   uv run ado get document -q 'config.relatedResources=RESOURCE_ID' --details
   ```

   Keep the newest by `created`, mark the rest (mirrors the replace-report
   pattern in
   [examining-ado-operations](../examining-ado-operations/SKILL.md) and
   [examining-discovery-spaces](../examining-discovery-spaces/SKILL.md)).
   Reason: `duplicate-report`.

7. **Duplicate project reports**: more than one `document` with
   `metadata.name: project_report`.

   ```bash
   uv run ado get document -q 'config.metadata.name=project_report' --details
   ```

   Keep the newest, mark the rest (mirrors
   [examining-ado-project](../examining-ado-project/SKILL.md) step 4).
   Reason: `duplicate-project-report`.

8. **Labeled `provisional`**: any resource carrying the `provisional` label
   (see [definition](#label--condition-definitions)). Reason:
   `provisional-testing` / `provisional-debug` / `provisional-temporary`
   (mirror the label's value).

9. **Pre-release/testing operator package version**: an operation whose
   `provenance.operators.<operatorIdentifier>.distributionVersion`
   (`ado/core/metadata.py`, PEP 440 version string) contains a pre-release,
   dev, or local segment (`a`, `b`, `rc`, `.dev`, `+`). This field has no
   dedicated CLI filter (its key is dynamic), so fetch operation YAML and
   inspect it client-side:

   ```bash
   uv run ado get operations -o yaml --output-file operations.yaml
   ```

   Reason: `prerelease-operator-version`.

## Step 1 detail — delete previously-marked resources

1. Query every deletable type for existing marks:

   ```bash
   uv run ado get operation -l for_deletion=true --details
   uv run ado get discoveryspace -l for_deletion=true --details
   uv run ado get datacontainer -l for_deletion=true --details
   uv run ado get document -l for_deletion=true --details
   ```

   If nothing is marked, state that explicitly and move on to Step 2.

2. Show the combined list and get explicit confirmation before deleting —
   deletion is irreversible.

3. Delete children before parents. Deleting a resource with children raises
   `ResourceHasChildrenError` (`docs/resources/index.md`). Order:
   `datacontainer` → `operation` → `discoveryspace`; `document` has no
   ordering constraint and can be deleted at any point.

   ```bash
   uv run ado delete datacontainer ID [ID...]
   uv run ado delete operation ID [ID...]
   uv run ado delete document ID [ID...]
   uv run ado delete discoveryspace ID [ID...]
   ```

   `--force` is only needed for deleting a non-empty samplestore or an
   operation while other operations are still running
   (`ado/cli/commands/delete.py`) — not expected here.

4. Report the success/failure summary `ado delete` already prints.

## Step 2 detail — mark new candidates

Walk conditions 1-9 above. For each match not already labeled
`for_deletion: "true"`, apply it plus the matching `deletion_reason`:

```bash
uv run ado edit TYPE ID -p "labels: {for_deletion: 'true', deletion_reason: <code>}"
```

Skip anything already marked — leave its existing mark/reason untouched; it
will be handled by a future Step 1.

## Step 3 detail — report

Print a summary table (type / id / name / `deletion_reason`) of everything
marked in *this run's* Step 2 only — not resources already marked before
this run (those were reported previously and/or just deleted in Step 1).
Remind the user how to unmark incorrect candidates before the next
invocation deletes them:

```bash
uv run ado edit TYPE ID --editor
```

## Step 4 detail — review existing labels/descriptions

This is this skill's label-curation responsibility; *setting* labels
correctly at creation time is covered by resource-yaml-creation, not here.

- Using the `--details` listings already fetched in Steps 1-2, check
  labels/descriptions are still accurate: a stale `provisional` label on a
  resource that turned out to be long-lived (recommend removing it), missing
  `project`/`team` labels worth adding, descriptions that no longer match
  the resource's actual use.
- Informational note, not an action this skill takes: a pre-release/dev
  operator *package* version is already caught automatically by condition
  9, so plugin developers should bump to a final release semver once
  validated to avoid unnecessary future cleanup churn.
- Add/overwrite labels: `uv run ado edit TYPE ID -p "labels: {...}"`. Remove
  a single label key: `uv run ado edit TYPE ID --editor` (interactive).
- Present findings as suggestions the user can act on — do not edit labels
  in this step without the user's confirmation (unlike Step 2, which acts
  autonomously on the fixed, unambiguous conditions).

## Scope split with resource-yaml-creation

- **resource-yaml-creation** (creation-time): defines the `provisional`
  label (values `testing`/`debug`/`temporary`, with rationale for each) and
  instructs setting it in `metadata.labels` when a resource is known at
  creation time to be exploratory/temporary.
- **ado-project-maintenance** (this skill): consumes `provisional` as a
  cleanup trigger (condition 8) and owns the `for_deletion`/`deletion_reason`
  labels end-to-end; also owns periodic review of all labels/descriptions
  (Step 4), but not the initial creation-time guidance.

## Related Resources

- [resource-yaml-creation](../resource-yaml-creation/SKILL.md) — `provisional`
  label definition and general metadata guidance
- [query-ado-data](../query-ado-data/SKILL.md) — `--filter`/`--label` syntax
  and resource statistics
- [using-ado-cli](../using-ado-cli/SKILL.md) — CLI command syntax and
  shortcuts
- [examining-ado-project](../examining-ado-project/SKILL.md) — project report
  conventions
- [AGENTS.md](../../../AGENTS.md) — general development guidelines
