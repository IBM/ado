---
name: examining-discovery-spaces
description: >-
  End-to-end workflow to examine and summarise an ado discoveryspace — fetch
  space YAML, describe entity and measurement space structure, assess sampling
  coverage, export measurement data, and find related resources. Use when the
  user asks to inspect, summarise, debug, or analyse a discoveryspace; wants to
  understand dimensions, experiments, or sampling coverage; provides a space ID
  or asks to use --use-latest for the current space.
---

# Examining ado Discovery Spaces

Structured workflow for understanding what a discoveryspace contains, how
covered its entity space is, and what data has been collected.

- Run all commands from the **repository root** with `uv run`.
- Write the report to `reports/<ado_context_name>/` (create the
directory if needed)
  - where `ado_context_name` is the
    **active ado metastore context** (`uv run ado context`)
- Write the report as `<SPACEID>_<YYYY-MM-DD>_report.md`

**Related skills**:

- For CLI verification and command spelling, see
  [using-ado-cli](../using-ado-cli/SKILL.md).
- For metastore filtering and schemas, see
  [query-ado-data](../query-ado-data/SKILL.md).
- For examining operations run on a space, see
  [examining-ado-operations](../examining-ado-operations/SKILL.md).

## Context

### Operations and DiscoverySpaces

- discoveryspaces (or spaces for short) define a set of points (entities) and
  how to measure them. They also contain the results of the measurements
- operations operate on discovery spaces either selecting or measuring points or
  analysing existing measurements

### Terminology: Distinguishing Entities in a DiscoverySpace

When working with the data from a discoveryspace the following distinctions are
important.

- Measured: These entities have been measured by an operation on the space
- Unmeasured: These entities have not been measured by an operation on the space

The samplestore used by a discovery space is shared. This means there may be
relevant measurement data in the samplestore for entities in the space but that
measurement has not been performed by an operation on the space (it was
performed on another).

- Matching: Data in the spaces samplestore that matches the space definition -
  it includes measured entities
- Missing: Entities that have no matching data in the samplestore.

Why is it useful to work with matching data?

1. Allows using the discoveryspace as a view to fetch particular data without
   having to perform operations on it
   - Concrete example: You create a discoveryspace that is a subspace of
     another sampled spaced to analyze it. You can perform analysis on existing
     data even though no operation has been run on the new discoveryspace.
2. Memoization: You can understand if there are
   [memoization opportunities](website/docs/core-concepts/data-sharing.md) that
   would speed up a operation on the space.

## Pre-requisites: The Space Identifier

To apply this skill you need either:

(a) a space id; (b) explicit instruction to examine the latest space

In the case of (b) get the actual identifier:

```bash
uv run ado show related space --use-latest
```

## Tips

### Avoiding refetching YAML

`ado get-o yaml` flag outputs YAML to console. It's often useful to redirect
this to a temporary file and work with that to avoid multiple `ado get` calls
for same YAML.

### Large output files

The files created by '-o/--output-format' can be very large e.g. from "show
entities".

When inspecting these files:

- Use wc to count the file size first before using head/tail/cat etc. on it.
- Use head -n1 to get column headers, this will not be large
- Avoid head -n > 1 unless you have a specific need e.g. checking if file is
  corrputed
- Avoid tail unless you have a specific need
- Prefer python e.g. pandas.read_csv for any detailed analysis on the file.

## Workflow

Run Step 2 and 3 first.
Then steps 4,5 and 6 can be run in parallel.

### Step 1: Get Space YAML

```bash
uv run ado get space SPACE_ID -o yaml
```

Extract and summarise:

- Resource **identifier** and **metadata** (name, description, labels)
- **sampleStoreIdentifier**: the sample store backing this space
- **entitySpace**: dimensions — property names, types (categorical / discrete /
  continuous), and their domains/values
- **experiments**: actuator and experiment identifiers that define what can be
  measured, and which target properties each experiment produces

### Step 2: Sampling coverage and related resources

Execute

```bash
uv run ado show details space SPACE_ID
```

This outputs two sections:

**DETAILS** — sampling coverage:

- Total entities in the space
- How many have been measured
- How many have failed measurements
- How many are unmeasured
- How many are matching

Compare measured vs total to understand exploration progress. Compare measured
vs matching to understand memoization opportunities. Also, a signal that other
overlapping spaces exist.

**RELATED RESOURCES** — all operations and stores linked to this space.

> **Performance note**: `ado show details space` is slow as it fetches and
> aggregates entity data. Use only when sampling coverage is needed.

### Step 3: Check for existing report

- Check if there is an existing report for this space in
  `reports/<ado_context_name>/`
- If yes, check if either of the following are true:
  - New operations have been run on space since report
  - The number of measured entities has increased
- If neither of above are true, ask the user if they want to write a
  new report or use existing
  - As nothing has changed, the only purpose of creating a new report is if a
    different agent is being used

### Step 4: Find Similar spaces

`ado get space --matching-space-id SPACE_ID --details` finds spaces with the
same entity structure. Use this to understand research progression and why this
space was created.

```bash
uv run ado get space --matching-space-id SPACE_ID --details
```

### Step 5: Export Measurement Data

Note: Keep in mind the [guidelines on large output files](#large-output-files)
for the following.

```bash
uv run ado show entities space SPACE_ID \
  --include measured \
  --property-format target \
  --output-format csv
```

This writes the data to `SPACE_ID_description_measured_target.csv`
automatically. If you find `SPACE_ID_description_measured_target.csv` already
exists do not use it, as data may be stale

You can also get lists of all unmeasured or missing entities, though this is not
typically required unless you want to analyse the unsampled portion.

Perform an analysis of the measurements, checking e.g. distributions of
metrics, metric outliers, correlations between metrics.
Take into account the domain of the experiment and meaning of metrics
when looking for patterns.

### Step 6: Examine Related Operations

For each related operation (output in step 2), use the
[examining-ado-operations](../examining-ado-operations/SKILL.md) skill to
understand what each operation did and what it produced.

Note: Do not analyze the data in the operations, or do detailed diagnoses. Just
enough for summary.

## Producing a Report

Structure the report as:

1. **Overview**: What the space represents. Infer from metadata, dimensions, and
   experiments. Short and narrative.
   - **Space summary** – ID, metadata, entity count, dimensions (parameters and
     their types/values)
   - **Measurement space** – experiments, target properties
2. **Related Spaces** (Optional): If there are related spaces, describe them and
   how they relate.
3. **Sampling coverage** – sampled vs unsampled vs missing counts; progress
   assessment
4. **Data summary** – distributions of measured properties, notable performers,
   outliers, correlations
5. **Related operations** – which operations ran on this space and their status
