---
name: examining-ado-operations
description: >-
  End-to-end workflow to examine and summarize an ado operation — fetch
  operation and space YAML, summarise configuration, export
  entities/requests/results to CSV, perform simple analysis, and interpret
  failures and data quality. Use when the user asks to summarize, analyse,
  debug, or review an operation; wants insights from measurement data; or
  provides an operation ID or asks to use --use-latest for the current
  operation.
---

# Examining ADO operations

Structured workflow for understanding what an operation did, which space it ran
on, and whether measurements and results look healthy.

- Run all commands from the **repository root** with `uv run`.
- The report produced by this skill is stored as the `content` of a
  `document` resource in the active ado metastore context (see
  [Producing a report](#producing-a-report)).

**Related skills**:

- For CLI verification and command spelling, see
  [using-ado-cli](../using-ado-cli/SKILL.md).
- For metastore filtering, schemas see
  [query-ado-data](../query-ado-data/SKILL.md).
- For creating document resources that store reports, see
  [resource-yaml-creation — Document](../resource-yaml-creation/SKILL.md#document).
- For a project/context wide view (all spaces and operations), see
  [examining-ado-project](../examining-ado-project/SKILL.md).

## Context

Operations are applied to discoveryspaces. There are different types of
operation. The General Workflow can be applied to all types of operation.

In addition, the Explore Operation Workflow can be applied to
Explore operations.

- Read [operations](../../../docs/resources/operation.md) documentation
  for details

## Pre-requisites: The Operation Identifier

To apply this skill you need either:

(a) an operation id; (b) explicit instruction to examine the latest operation

In the case of (b) (latest) get the actual operation identifier as follows

```bash
uv run ado show related operation --use-latest
```

This will output the id of the latest operation created in the active ado
context.

## Tips

### Avoiding refetching YAML

`ado get … -o yaml` (or `json`) writes to stdout by default. Prefer
`--output-file PATH` with the same format flag, then work from that file to
avoid repeated `ado get` calls for the same resource.

In particular `ado get datacontainer … -o yaml` or `-o json` can be large; use
`--output-file` and load the file with Python (or another tool) instead of
re-fetching.

### Large output files

The output for a chosen `-o`/`--output` **format** can be very large (for
example from `show measurements` or `show trace`). Use
`--output-file` with the destination path and, when inspecting these files:

- Use wc to count the file size first before using head/tail/cat etc. on it.
- Use head -n1 to get column headers, this will not be large
- Avoid head -n > 1 unless you have a specific need e.g. checking if file is
  corrupted
- Avoid tail unless you have a specific need
- Prefer python e.g. pandas.read_csv for any detailed analysis on the file.

## General Workflow

- Run Steps 1 and 2 first
- Steps 3, 4 and 5 can be run in parallel
- Step 6 depends on Step 5

### Step 1: Get the operation YAML

```bash
uv run ado get operation OPERATION_ID -o yaml --output-file OPERATION_ID.yaml
```

Extract and summarise:

- Resource **identifier**, **operationType**, **operatorIdentifier**
- The identifiers of **input resources** to the operation:
  - discovery spaces from the spaces field
  - actuatorconfigurations from the actuatorConfigurationIdentifiers field
- **config**: operator-specific parameters (what the run was asked to do)
- **status**: latest **event** (e.g. started / finished) and **exit_state** when
  finished (success / fail / error)

Note anything in config that influences what operation does (thresholds,
objectives, stopping rules, etc.).

#### Identifying if an operation is still running

An operation which does not report finished is usually still running.

However, it is possible it failed in a way that meant it could not record the
failure. In this case:

1. Determine how long it has been running.
2. If it is many hours and the operationType is not explore flag that
   there may be a problem
3. If it is many hours and the operationType is explore proceed use
   specific techniques in Explore Operation Workflow to determine if its still
   running

### Step 2: Check for existing report

If the operation is finished,

- Query the metastore for an existing document linked to this operation:

  ```bash
  uv run ado get document -q 'config.relatedResources=OPERATION_ID'
  ```

  If a document is found, retrieve its metadata (name, created timestamp) and
  fetch its `content` (`uv run ado get document DOCUMENT_ID -o yaml`) to check
  if that report indicated the operation was finished.
  - If yes, ask the user whether to replace it with a new report. If they
    agree, delete the existing document (`uv run ado delete document
    DOCUMENT_ID`) once the new report has been created. See
    [resource-yaml-creation — Document](../resource-yaml-creation/SKILL.md#document).
  - If no, continue with creating a new report.

### Step 3: Review the operator

Each operation is run by an operator. The operator's name is retrieved in step
one, as the value of the field operatorIdentifier.

Note: `operatorIdentifer` is not the same as `operationIdentifier`

Execute

```bash
uv run ado get operator --details $OPERATOR_IDENTIFIER
```

#### Understanding operator parameters

To understand an operator's parameters, examine its schema:

```bash
uv run ado template operation --operator-name $OPERATOR_IDENTIFIER --include-schema
```

This will create a file called `operation_template_$UID_schema.yaml` containing
the schema.

### Step 4: Describe the space

Using the space id from step 1

```bash
uv run ado get space SPACE_ID -o yaml --output-file SPACE_ID.yaml
uv run ado describe space SPACE_ID
```

Summarise the: **dimensions** (parameters), **experiments** (actuators,
experiment types), **entity space** structure, and notable **constraints** or
metadata. For deeper context, read operator and experiment documentation under
`docs/user-guide/operators/` and actuator/experiment docs as needed (match
**operatorIdentifier** and experiment types from the space).

### Step 5: Get the output resources of the operation

Operations can create other resources. To identify these

```bash
uv run ado show related operation $OPERATION_IDENTIFIER
```

This will output the identifiers of the input and output resources related to
the operation.

From step 1 you know the input resource identifiers so you can work out the
output identifiers.

### Step 6: Examine the output resources of the operation (if any)

An operation can create the following resources

- discovery spaces: In this case examine the space as in step 3
- operations: In this case recursively examine the operations using this skill
- datacontainers: This contains non-ado resource outputs e.g. CSV data.

To retrieve contents of data container. Use `--output-file` to ensure proper
file handling:

```bash
uv run ado get datacontainer $DATACONTAINER_IDENTIFIER -o yaml --output-file datacontainer.yaml
```

For each output resource summarize what it is/contains.

## Explore Operation Workflow

The following assumes the General Workflow has been applied.

Explore operations sample entities from a discovery space and make
measurements on them.

Notes:

- If the data for the measurements exists it can be memoized (depends on
  operation parameters)
- The operation parameters will specify the number of entities to sample in some
  way.

Relevant Documentation

- [sample process](../../../docs/concepts/discovery-spaces.md#sampling-and-measurement)
- [memoization](../../../docs/concepts/data-sharing.md#memoization)

### Step 1: Get Details on what was Sampled and Measured

To get a numerical overview of results and requests before diving into the
trace, use:

```bash
uv run ado show stats operation $OPERATION_ID
```

This outputs the base table columns output by ado get plus `TOTAL_RESULTS`,
`SUCCESSFUL_RESULTS`, `FAILED_RESULTS`, `MEASURED_ENTITIES`, `TOTAL_REQUESTS`,
`FAILED_REQUESTS`, `SUCCESSFUL_REQUESTS`.

Compare this with the number of samples requested in the operator parameters.

If the state is finished, exit status was successful and all requested samples
were completed there are no issues ->
[examine entities](#step-3-get-entities-and-measurements)

If the state is not finished ->
[Use the diagnose if sampling operation running workflow](#diagnose-if-an-explore-operation-is-running-workflow).
For all other combinations ->
[Diagnose sampling issues](#step-2-optional-diagnose-sampling-issues)

### Step 2 (Optional): Diagnose sampling issues

First run these two commands to get the metadata on what was requested and
measured, noting the [guidelines on large files](#large-output-files):

```bash
uv run ado show trace operation OPERATION_ID \
  -o csv --output-file OPERATION_ID_trace.csv
```

Use `--unroll-entities` to include per-entity result metadata (validity,
reasons for invalidity, etc.) in the trace output:

```bash
uv run ado show trace operation OPERATION_ID --unroll-entities \
  -o csv --output-file OPERATION_ID_trace_entities.csv
```

- **trace**: This is metadata on what the sampling operation asked an
  actuator to measure. It includes the timestamp of when the request was
  created. Each row represents a measurement request.
- **trace --unroll-entities**: Each row in the output represents an entity
  result within a request, showing per-entity measurement metadata such as
  validity and reasons for invalidity.
  - InvalidMeasurementResult: The experiment failed for some reason

From the trace output, identify **failed** or **invalid** rows, **reasons**
for invalidity, and anomalies in **timing** or **ordering** if those columns
are present.

### Step 3: Get entities and measurements

To get the data on measurements execute (noting the
[guidelines on large files](#large-output-files)):

```bash
uv run ado show measurements operation OPERATION_ID \
  -o csv --output-file OPERATION_ID_entities.csv
```

### Step 4: Analyze the Measurement data

Perform an analysis of the measurements, checking e.g. distributions of metrics,
metric outliers, correlations between metrics. Take into account the domain of
the experiment and meaning of metrics when looking for patterns.

## Diagnose if an Explore Operation is Running Workflow

- Check if the operation is submitting experiments in batches
- Confirm if the operation uses continuous batching (new experiment requested
  once one has finished) or static batch (full batch finishes then next starts)
- Get the requests and results timeseries using `ado show trace`
- For continuous batching
  - Use the request time-series to determine the typical inter-request start
    time after the first batch i.e. this tells you how often after the first
    batch you should expect to see a new request
- For static batch
  - Use the request time-series to determine the typical inter-batch time i.e.
    how long between batches/how long a batch takes to execute on average
- Determine if the time since last recorded request is much greater than the
  expected inter request time e.g. 5x more. This indicates there may have been
  an issue.

## Producing a report

Structure the report as:

1. **Overview**: What the operation purpose was. Can be inferred from space and
   operation chosen. Short and narrative.
   - **Operation summary** – ID, operator, parameters, status
   - **Space summary** – dimensions, experiments, entity count
2. **Measurement overview** – sampled vs requested, success vs failure counts
3. **Findings** – notable patterns, best/worst performers, anomalies
4. **Unusual behaviour** – failures, timeouts, invalid results, unexpected
   distributions
5. **Next Steps**: A plan for the next research steps to take using ado.

Store the report as a document resource (see
[resource-yaml-creation — Document](../resource-yaml-creation/SKILL.md#document)).
Set `relatedResources` to the operation id and the input space ids from step 1.

## Troubleshooting

### Number points sampled is greater than the number of entities measured by operation

Some samplers can sample the same entity twice. In this case you may see
conflicting statistics about how many entities are measured. For example if an
operation is configured to sample two points, and it samples same point twice,
the additional number entities with measurements after the operation is 1, but
the number of points sampled by operation is 2.

Comparing the size of the set of entity identifiers to the timeseries length can
confirm this.

### Memoization on, but Entities measured twice

The requests which use memoized results for Entities are called "replayed
measurements". If the same entity is sampled twice in an operation, the second
should be replayed. If it is not, it means the sampling algorithm selected the
same point again before the first was stored to be reused. In this case it means
the same entity will be measured twice.
