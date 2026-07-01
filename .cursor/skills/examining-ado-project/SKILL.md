---
name: examining-ado-project
description: >-
  Builds a picture of work in an ado project: activity volume, spaces and
  operations created over time, experiments and operation configs  used etc. Use
  to create a project/context overview report, summarize what the team has been
  doing in an ado project, report trends across spaces/operations, or to onboard
  onto an ado project.
---

# Examining an ado Project

End-to-end workflow to summarize **all** discoveryspaces, operations, and
related metadata in the ado project associated to the active context.

- Run all commands from the **repository root** with `uv run` (see
  [using-ado-cli](../using-ado-cli/SKILL.md)).
- See
  [projects and contexts](../../../website/docs/resources/metastore.md#contexts-and-projects)
  for details on what projects and contexts are
- Users may refer to an ado project using either the term "project" or "context"

## Tips

- Prefer metastore listing and YAML dumps before heavy `uv run ado show` data
  pulls; see [query-ado-data](../query-ado-data/SKILL.md).
- For one space in depth:
  [examining-discovery-spaces](../examining-discovery-spaces/SKILL.md).
- For one operation in depth:
  [examining-ado-operations](../examining-ado-operations/SKILL.md).

## Pre-requisites - Check active context is associated with expected project

Context names and project names are identical by construction.

If asked to examine a specific named project:

1. Check the active context name: `uv run ado context`
   - If it has the correct name, continue to
     [next step](#1-overview-activity-and-types)
   - If not, run `uv run ado get contexts` to list all available contexts
     - If one matches, switch to it: `uv run ado context $NAME`
     - If none match, inform the user that a context for the specified project
       cannot be found

## 1. Overview: activity and types

Goal: volume of work, recency, and which spaces attract the most operations.

1. **Spaces (tabular, with metadata)**

   ```bash
   uv run ado get spaces --details
   ```

   Use **age** (list is age-sorted, most recent last), **name**,
   **description**, and **labels** to infer themes and activity.

2. **Operations (tabular, with metadata)**

   ```bash
   uv run ado get operations --details
   ```

   Relates operations to target spaces; age and labels summarize recent work.

   **Heuristic — operation IDs:** many ids encode the operator and a version
   segment, e.g. `OPERATOR_NAME-VERSION-...-UID` (exact shape varies). Use this
   together with `uv run ado get operator --details` to understand more about
   the operators used.

3. **Operations (with measurement statistics)**

   ```bash
   uv run ado get operations -o stats --output-file operations-stats.txt
   ```

   Adds `TOTAL_RESULTS`, `SUCCESSFUL_RESULTS`, `FAILED_RESULTS`, and
   `MEASURED_ENTITIES` columns. For explore operations, compare
   `MEASURED_ENTITIES` against the operator's `numberEntities` to check sampling
   completeness.

   For richer stats that also include request-level counts
   (`TOTAL_REQUESTS`, `FAILED_REQUESTS`, `SUCCESSFUL_REQUESTS`):

   ```bash
   uv run ado show stats operation --output-file operations-fullstats.csv -o csv
   ```

4. **Discovery Spaces (with statistics)**

   ```bash
   uv run ado get spaces -o stats --output-file spaces-stats.txt
   ```

   Adds `EXPERIMENTS`, `OPERATIONS`, `EXPLORE_OPERATIONS`, and
   `MEASURED_ENTITIES` columns.

   For richer stats that also include full entity-space coverage columns
   (`SIZE_OF_ENTITY_SPACE`, `UNMEASURED_ENTITIES`, `MATCHING_ENTITIES`, etc.):

   ```bash
   uv run ado show stats discoveryspace --output-file spaces-fullstats.csv -o csv
   ```

   > **Performance note**: `ado show stats discoveryspace` is slower than
   > `ado get spaces -o stats` as it instantiates each `DiscoverySpace`.
   > Prefer `ado get -o stats` for a quick overview across many spaces.

5. **Sample Stores (with statistics)**

   ```bash
   uv run ado get samplestores -o stats --output-file samplestores-stats.txt
   ```

   Adds `ENTITIES`, `RESULTS`, and `EXPERIMENTS` columns.

6. **Data Containers (with statistics)**

   ```bash
   uv run ado get datacontainers -o stats --output-file datacontainers-stats.txt
   ```

   Adds `TABLES`, `LOCATIONS`, `KEY_VALUES`, and `DATA_BYTES` columns.

**Synthesis:** cluster mentally (or in notes) by **creation time** to see bursts
of activity; count operations **per space** from the operations listing to see
which spaces are busiest.

## 2. Deeper pass: full YAML and experiments

Goal: experiments/actuators in use, operation parameters, and how much was
**submitted** for measurement (entities selected by explore-style operators—may
differ from completed measurements if runs failed).

Dump full resource documents for easier scripted or batched reading. Use
`--output-file` to ensure proper file handling:

```bash
uv run ado get spaces -o yaml --output-file spaces.yaml
uv run ado get operations -o yaml --output-file operations.yaml
```

On **large metastores**, these files can be very large—prefer
`uv run ado get … -q …` filters, scripts, or the SQL store API (see
[query-ado-data](../query-ado-data/SKILL.md)) before dumping everything.

When interpreting YAML fields, confirm paths against resource schemas:

```bash
uv run ado template discoveryspace --include-schema
uv run ado template operation --include-schema
```

From **space** YAML: note **experiment** and **actuator** identifiers referenced
by each space.

Gain further information on the experiments using

```bash
# Outputs description of actuators used
uv run ado get actuators --details
# Outputs description of experiments used
uv run ado get experiments --details
# Outputs detailed information on an experiment
# Use to drill down into most used experiments
uv run ado describe experiment $EXPERIMENT_ID
```

From **operation** YAML: note actuatorconfigurations used (if any), read
**parameters** (operator-specific), target **discoveryspace** references, and
any fields that indicate **entity count / batch / sample** configuration for
explore operations.

**Synthesize:**

- What is being **measured** and with which experiments (domain of the project).
- Which **explore** (or similar) operations drove the largest submitted entity
  sets and on **which spaces**.
- Whether **experiment definitions or parameters** shift over time (versions,
  configs).

## 3. Space relationships and entity-space shape

From step 2, pick a handful of **commonly operated on or recent** spaces. For
each, inspect its fragment in `spaces.yaml`, focusing on **entity space**
structure (dimensions, bounds, representation).

Find **Spaces that match these spaces** (refinement, expansion, or parallel
configurations)

```bash
uv run ado get spaces --matching-space-id SPACE_ID
```

Use the output to

- Group spaces that match
- Within each group identify subgroups whose spaces are identical
- If multiple subgroups identify the hierarchy between them (broadest to
  narrowest)
- Within subgroups identify what if measurement space is different between them
- Combine this information with the sequence of space creation to understand how
  researchers have been evolving the spaces

Optionally complement with one-hop links:

```bash
uv run ado show related space SPACE_ID
```

## 4. Report template

Write a concise markdown report

- Write the report to `reports/<ado_context_name>/` (create the directory if
  needed), where `ado_context_name` is the **active ado metastore context**
  (`uv run ado context`).
- Write the report as `project_<YYYY-MM-DD>_report.md`.
- Query the metastore for an existing project report document:

  ```bash
  uv run ado get document -q metadata.name=project_<YYYY-MM-DD>_report
  ```

  If a matching document is found, ask the user whether to replace it.
- If a report already exists, check whether there has been meaningful activity
  since it was written before replacing it:
  1. Run `uv run ado get spaces --details` and
     `uv run ado get operations --details` and note the age of the most recent
     resource.
  2. If the most recent resource is **younger** than the date of the existing
     report, there has been new activity — proceed to write a new report.
  3. If not, ask the user whether they want to replace it.
  4. If finer-grained confirmation is needed, fetch the YAML of the most recent
     space or operation
     (`uv run ado get space SPACE_ID -o yaml --output-file SPACE_ID.yaml`, or
     the same pattern for `operation`) and read its `creationTimestamp` field.

### Project summary

- Domains or problems implied by experiments, actuators, and space descriptions.
- Dominant **operation** and **experiment** patterns.

### Latest activity

- Most recent spaces and operations (from `--details` listings).
- What the latest work seems focused on (labels, names, target spaces).

### Spaces overview

- Which spaces are most used and most analyzed.
- How **entity spaces** and **matching-space** relationships evolve: expanding,
  narrowing, or shifting configuration.

### Operations overview

- Operator mix and whether **parameters** or **operator choice** evolve over
  time.
- Which operations **submitted** the most entities (from operation YAML/config)
  and which produced the most results (from the stats).
- What **analysis**-style operations ran (infer from operator names and
  parameters).
- Make a note of operations with failed measurements and highlight ones with
  abnormal failure rates (from the stats).

After writing the local report file, persist it as a document resource. Since a
project report spans many resources, set `relatedResources` to the identifiers
of the most recently active spaces and operations (for example the five most
recently created of each):

```yaml
# project_<YYYY-MM-DD>_document.yaml  (temp file, not committed)
metadata:
  name: "project_<YYYY-MM-DD>_report"
  description: "<one-line summary>"
content: |
  <full markdown report text>
relatedResources:
  - <recent space ids>
  - <recent operation ids>
```

Any charts or images generated during analysis should be base64-encoded and
included in the `attachments` section, with the markdown referencing them by
filename.

```bash
uv run ado create document -f project_<YYYY-MM-DD>_document.yaml
```
