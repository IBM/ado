---
name: ado-project-maintenance
description: >-
  Describes how to maintain an ado project including how to identify
  resources that are candidates for deletion, and how to label testing
  or provisional resources. Use when asked to clean up,
  tidy, or run maintenance on an ado project/context; to find what can be
  deleted; or to review/curate resource metadata.
---

# ado Project Maintenance

The skill explains how to maintain an ado project including

- how to identify and label stale resources
- when stale resources can be deleted
- how to ensure up-to-date metadata

## Workflow

1. [Delete marked resources](#step-1-delete-marked-resources)
2. [Identify candidates for deletion](#step-2-identify-candidates-for-deletion)
3. [Review existing labels/descriptions](#step-3-review-metadata)
4. [Report](#step-4-report)

### Step 1: Delete marked resources

Query every deletable type for existing marks:

   ```bash
   uv run ado get operation -l for_deletion=true --details
   uv run ado get discoveryspace -l for_deletion=true --details
   uv run ado get datacontainer -l for_deletion=true --details
   uv run ado get document -l for_deletion=true --details
   ```

If nothing is marked, move on to Step 2.

Delete children before parents. Deleting a resource with children raises
`ResourceHasChildrenError` Order:
   `datacontainer` → `operation` → `discoveryspace`; `document` has no
   ordering constraint and can be deleted at any point.

   ```bash
   uv run ado delete datacontainer ID [ID...]
   uv run ado delete operation ID [ID...]
   uv run ado delete document ID [ID...]
   uv run ado delete discoveryspace ID [ID...]
   ```

## Step 2: Identify candidates for deletion

Check all the [conditions](#conditions-that-qualify-a-resource-for-deletion)
For each match add `for_deletion: "true"` and the matching `deletion_reason` labels.

```bash
uv run ado edit TYPE ID -p "labels: {for_deletion: 'true', deletion_reason: <code>}"
```

## Step 3: Review metadata

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

## Step 4: Report

The report should have three sections

- Deleted: lists the resources deleted in Step 1
- Marked: lists the resources identified in Step 2 and why (table)
   - Include in this section how to unmark candidates
- Metadata: Lists any stale or missing metadata with potential fixes
   - Stale labels that are being used by resources
   - Resources that are missing labels
   - Descriptions that are stale and should be updated
   - Resources missing descriptions

## Label & condition definitions

Three labels make up the maintenance scheme.

- **`provisional: <reason>`**
  - Identifies that a resource is temporary for the given reason
  - Set at resource creation time or proactively by the user
- **`for_deletion: "true"`** —
  - Identifies that a resource is a candidate for deletion
- **`deletion_reason: <short-code>`**
  — Records why a resource is a candidate for deletion: one of
  `empty-space-stale`, `failed-no-entities`,
  `error-no-entities`, `superseeded-operator-minor-version`,
  `orphaned-datacontainer`, `superseeded-report`, `superseeded-project-report`,
  `provisional`,
  `prerelease-operator-version`.

### Conditions that qualify a resource for-deletion

1. **Stale empty spaces**: no measured entities, older than a week.

   ```bash
   uv run ado get spaces -o stats --details --output-file spaces-stats.txt
   ```

   Then filter rows where `MEASURED_ENTITIES == 0`.

   For each space use the value of `created` field
   client-side to filter those older than a week

   ```bash
   uv run ado get space SPACE_ID -o yaml --output-file SPACE_ID.yaml
   ```

   Deletion Reason: `empty-space-stale`.

2. **Failed operations that sampled no entities**: operation finished with
   `exit_state: fail` and made zero measurement requests.

   ```bash
   uv run ado show stats operation \
     --filter 'status=[{"event":"finished","exit_state":"fail"}]' \
     -o csv --output-file operations-fullstats.csv
   ```

   Filter rows where `TOTAL_REQUESTS == 0`.

   Deletion Reason: `failed-no-entities`.

3. **Error-state operations**: any operation that finished with
   `exit_state: error` and made zero measurement requests.

   ```bash
   uv run ado get operations \
     --filter 'status=[{"event":"finished","exit_state":"error"}]' --details
   ```

   Deletion Reason: `error-no-entities`.

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
   and mark the rest.  There is
   no single CLI filter that performs this grouping — do it as a
   script/manual pass over the YAML dump.

   Deletion Reason: `superseeded-operator-minor-version`.

5. **Orphaned datacontainers**: datacontainers belonging to an operation
   that qualifies under conditions 2-4.

   ```bash
   uv run ado show related operation OP_ID
   ```

   Mark any datacontainer returned.

   Deletion Reason: `orphaned-datacontainer`.

6. **Duplicate per-resource reports**: more than one `document` report
   related to the same space or operation.

   ```bash
   uv run ado get document -q 'config.relatedResources=RESOURCE_ID' --details
   ```

   Keep the newest by `created`, mark the rest (mirrors the replace-report
   pattern in
   [examining-ado-operations](../examining-ado-operations/SKILL.md) and
   [examining-discovery-spaces](../examining-discovery-spaces/SKILL.md)).

   Deletion Reason: `superseeded-report`.

7. **Duplicate project reports**: more than one `document` with
   `metadata.name: project_report`.

   ```bash
   uv run ado get document -q 'config.metadata.name=project_report' --details
   ```

   Keep the newest, mark the rest

   Deletion Reason: `superseeded-project-report`.

8. **Labeled `provisional`**: any resource carrying the `provisional` label

   Deletion Reason: `provisional`

9. **Pre-release/testing operator package version**: an operation whose
   `provenance.operators.<operatorIdentifier>.distributionVersion`
    contains a pre-release, dev, or local segment (`a`, `b`, `rc`, `.dev`, `+`).

   This field has no dedicated CLI filter so fetch operation YAML and
   inspect it client-side:

   ```bash
   uv run ado get operations -o yaml --output-file operations.yaml
   ```

   Deletion Reason: `prerelease-operator-version`.

## Related Skills

- [resource-yaml-creation](../resource-yaml-creation/SKILL.md) — `provisional`
  label definition and general metadata guidance
- [query-ado-data](../query-ado-data/SKILL.md) — `--filter`/`--label` syntax
  and resource statistics
- [using-ado-cli](../using-ado-cli/SKILL.md) — CLI command syntax and
  shortcuts
