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
- The report produced by this skill is stored as the `content` of a
  `document` resource in the active ado metastore context (see
  [Producing a report](#producing-a-report)).

**Related skills**:

- For CLI verification and command spelling, see
  [using-ado-cli](../using-ado-cli/SKILL.md).
- For metastore filtering and schemas, see
  [query-ado-data](../query-ado-data/SKILL.md).
- For creating document resources that store reports, see
  [resource-yaml-creation — Document](../resource-yaml-creation/SKILL.md#document).
- For examining operations run on a space, see
  [examining-ado-operations](../examining-ado-operations/SKILL.md).
- For a project/context wide view (all spaces and operations), see
  [examining-ado-project](../examining-ado-project/SKILL.md).

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
   - Concrete example: You create a discoveryspace that is a subspace of another
     sampled spaced to analyze it. You can perform analysis on existing data
     even though no operation has been run on the new discoveryspace.
2. Memoization: You can understand if there are
   [memoization opportunities](docs/concepts/data-sharing.md) that
   would speed up a operation on the space.

## Pre-requisites: The Space Identifier

To apply this skill you need either:

(a) a space id; (b) explicit instruction to examine “the latest” space

In the case of (b) get the actual identifier:

```bash
uv run ado show related space --use-latest
```

## Tips

### Avoiding refetching YAML

`ado get … -o yaml` writes YAML to stdout by default. Prefer
`--output-file PATH` (with the same `-o yaml`) to save it once and reuse the
file instead of calling `ado get` repeatedly for the same resource.

### Large output files

The output produced for a given `-o`/`--output` **format** can be very large
(for example from `show measurements`). Use `--output-file` with the path where
the output should be saved, and when inspecting these files:

- Use wc to count the file size first before using head/tail/cat etc. on it.
- Use head -n1 to get column headers, this will not be large
- Avoid head -n > 1 unless you have a specific need e.g. checking if file is
  corrupted
- Avoid tail unless you have a specific need
- Prefer python e.g. pandas.read_csv for any detailed analysis on the file.

## Workflow

Run Step 2 and 3 first. Then steps 4,5 and 6 can be run in parallel.

### Step 1: Get Space YAML

```bash
uv run ado get space SPACE_ID -o yaml --output-file SPACE_ID.yaml
```

Extract and summarise:

- Resource **identifier** and **metadata** (name, description, labels)
- **sampleStoreIdentifier**: the sample store backing this space
- **entitySpace**: dimensions — property names, types (categorical / discrete /
  continuous), and their domains/values
- **experiments**: actuator and experiment identifiers that define what can be
  measured, and which target properties each experiment produces

### Step 2: Sampling coverage and related resources

To get measured entities, experiments, operations, and full entity-space
coverage columns, use:

```bash
uv run ado show stats discoveryspace SPACE_ID
```

This outputs the base table columns plus full entity-space coverage columns:
`SIZE_OF_ENTITY_SPACE`, `UNMEASURED_ENTITIES`, `MATCHING_ENTITIES`,
`MATCHING_WITH_MEASUREMENTS`, `ENTITIES_WITH_ALL_MEASUREMENTS`,
`ENTITIES_WITH_PARTIAL_MEASUREMENTS`, `MATCHING_ENTITIES_WITH_ALL_MEASUREMENTS`.

Compare `MEASURED_ENTITIES` vs `SIZE_OF_ENTITY_SPACE` to understand exploration
progress. Compare `MEASURED_ENTITIES` vs `MATCHING_ENTITIES` to understand
memoization opportunities — a large gap signals other overlapping spaces exist.

For related resources (operations and stores linked to this space), execute:

```bash
uv run ado show related space SPACE_ID
```

> **Performance note**: `ado show stats discoveryspace` is slow as it fetches
> and aggregates entity data. Use only when sampling coverage is needed.

### Step 3: Check for existing report

- Query the metastore for an existing document linked to this space:

  ```bash
  uv run ado get document -q 'config.relatedResources=SPACE_ID'
  ```

  If a document is found, retrieve its metadata (name, created timestamp) and
  fetch its `content` (`uv run ado get document DOCUMENT_ID -o yaml`) to
  compare against current state.
- If a document is found, check if either of the following are true:
  - New operations have been run on the space since the document was created
  - The number of measured entities has increased
- If yes to either, ask the user whether to replace it with a new report. If
  they agree, delete the existing document (`uv run ado delete document
  DOCUMENT_ID`) once the new report has been created. See
  [resource-yaml-creation — Document](../resource-yaml-creation/SKILL.md#document).
- If neither of the above are true, ask the user if they want to write a new
  report or use the existing one
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
uv run ado show measurements space SPACE_ID \
  --include measured \
  --property-format target \
  -o csv --output-file SPACE_ID_entities.csv
```

This writes the data to `SPACE_ID_entities.csv`. If you find
`SPACE_ID_entities.csv` already exists do not use it, as data may be stale

You can also get lists of all unmeasured or missing entities, though this is not
typically required unless you want to analyse the unsampled portion.

Perform an analysis of the measurements, checking e.g. distributions of metrics,
metric outliers, correlations between metrics. Take into account the domain of
the experiment and meaning of metrics when looking for patterns.

### Step 6: Examine Related Operations

For each related operation (output in step 2), use the
[examining-ado-operations](../examining-ado-operations/SKILL.md) skill to
understand what each operation did and what it produced.

Note: Do not analyze the data in the operations, or do detailed diagnoses. Just
enough for summary.

## Producing a report

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

Store the report as a document resource (see
[resource-yaml-creation — Document](../resource-yaml-creation/SKILL.md#document)).
Set `relatedResources` to the space id and the related operation ids from
step 2.
