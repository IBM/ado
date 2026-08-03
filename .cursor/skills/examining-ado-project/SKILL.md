---
name: examining-ado-project
description: >-
  Builds a picture of work in an ado project: activity volume, spaces and
  operations created over time, study documents, experiments and operation
  configs used etc. Use to create a project/context overview report, summarize
  what the team has been doing in an ado project, report trends across
  spaces/operations/studies, or to onboard onto an ado project.
---

# Examining an ado Project

End-to-end workflow to summarize **all** discoveryspaces, operations, and
related metadata in the ado project associated to the active context.

- Run all commands from the **repository root** with `uv run` (see
  [using-ado-cli](../using-ado-cli/SKILL.md)).
- The report produced by this skill is stored as a
  `document` resource in the active ado metastore context (see
  [Producing a report](#step-7-write-the-report)).
- A **project** is the namespace for all ado resources (spaces, operations,
  stores). A **context** is the local config pointing to a project's metastore
  (SQLite for local projects, MySQL for remote). Context name equals project
  name by construction. This definition is sufficient for applying this skill.
  Only consult the source if you need full schema details: read
  `docs/resources/metastore.md` if the source repo is available, otherwise see
  <https://ibm.github.io/ado/latest/resources/metastore/#contexts-and-projects>.
- Users may refer to an ado project using either the term "project" or "context"

## Tips

- Prefer metastore listing and YAML dumps before heavy `uv run ado show` data
  pulls; see [query-ado-data](../query-ado-data/SKILL.md).
- For creating document resources that store reports, see
  [resource-yaml-creation — Document](../resource-yaml-creation/SKILL.md#document).
- For study documents (`study-$ID`), see
  [create-research-study-document](../create-research-study-document/SKILL.md).
- For one space in depth:
  [examining-discovery-spaces](../examining-discovery-spaces/SKILL.md).
- For one operation in depth:
  [examining-ado-operations](../examining-ado-operations/SKILL.md).

## Pre-requisites

### Check active context is associated with expected project

Context names and project names are identical by construction.

If asked to examine a specific named project:

1. Check the active context name: `uv run ado context`
   - If it has the correct name, continue to
     [step one](#step-1-check-for-existing-report)
   - If not, run `uv run ado get contexts` to list all available contexts
     - If one matches, switch to it: `uv run ado context $NAME`
     - If none match, inform the user that a context for the specified project
       cannot be found

## Workflow

### Step 1. Check for existing report

The project report is stored as a `document` resource whose `metadata.name` is
 `project_report`.

Query the metastore:

```bash
uv run ado get document -q 'config.metadata.name=project_report' --details
```

There **may** be zero, one, or
many documents with that name — each has a unique identifier. If several match,
the **current** report is the most recently created one.

If matches exist

- identify the **current** document (newest by creation timestamp from
   `--details` or from document YAML `created` / status timestamps).
- Fetch its
   `content` (`uv run ado get document DOCUMENT_ID -o yaml`).
- Proceed to step 2

If no matches exist:

- Proceed to Step 3

### Step 2: Determine if there has been recent activity

Here we check whether there has been meaningful activity since the current report
was written:

1. Contextualizing Activity
   1. Run `uv run ado get docs --details`
   2. Find any study docs that were created after the last project report was written
   3. If there are, there has been recent activity in describing the motivations
      and scope of project
2. Operation Activity
   1. Run `uv run ado get operations --details -o stats`
   2. Find all the operations that were created after the last report was written
   3. Filter these for ones that are finished AND, if explore operations, that
      measured entities
   4. If there are, there has been recent experiment and analysis activity

If there has been no recent activity,
summarize the existing report for the user and point them to it. Do not continue.

If there has been recent activity, continue to next steps but focus on examining
the new activity in order to create an updated report based on the existing one.

### Step 3: Get overview of research

First check if there are study documents outlining
the research underway in the project

```bash
uv run ado get document --details | grep "study-"
```

For each study found, run

```bash
uv run ado get document DOCUMENT_ID -o yaml
```

to understand the study purpose and to find the labels
used to identify resources associated with the study

If there are study documents perform the next steps per study.

## Step 4: Examine research activities

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
which spaces are busiest; group activity under active **studies** when study
documents exist.

### Step 5: Deeper pass: full YAML and experiments

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

### Step 6: Space relationships and entity-space shape

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

### Step 7: Write the report

Write a concise markdown report. Store it as the `content` field of a
`document` resource (see
[resource-yaml-creation — Document](../resource-yaml-creation/SKILL.md#document)).

- Set `metadata.name` to `project_report`.
- Set `metadata.description` to include today's date, e.g.
  `Project overview as of YYYY-MM-DD`.
- Omit `relatedResources` (or leave empty).

### Project summary

- If study documents available:
  - Summary of the particular study or studies underway
- If no study documents available or many resource without associated study
  - Domains or problems implied by experiments, actuators, and space descriptions.
- Dominant **operation** and **experiment** patterns.

### Latest activity

Per-study when possible

- Most recent spaces and operations (from `--details` listings).
- What the latest work seems focused on (labels, names, target spaces).

### Spaces overview

Per-study when possible

- Which spaces are most used and most analyzed.
- How **entity spaces** and **matching-space** relationships evolve: expanding,
  narrowing, or shifting configuration.

### Operations overview

Per-study when possible

- Operator mix and whether **parameters** or **operator choice** evolve over
  time.
- Which operations **submitted** the most entities (from operation YAML/config)
  and which produced the most results (from the stats).
- What **analysis**-style operations ran (infer from operator names and
  parameters).
- Make a note of operations with failed measurements and highlight ones with
  abnormal failure rates (from the stats).
