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
- Write the report to `reports/<ado_context_name>/` (create the directory if
  needed)
  - where `ado_context_name` is the **active ado metastore context**
    (`uv run ado context`)
- Write the report as `<OPERATIONID>_<YYYY-MM-DD>_report.md`

**Related skills**:

- For CLI verification and command spelling, see
  [using-ado-cli](../using-ado-cli/SKILL.md).
- For metastore filtering, schemas see
  [query-ado-data](../query-ado-data/SKILL.md).

## Context

Operations are applied to discoveryspaces. There are different types of
operation. The General Workflow can be applied to all types of operation.

In addition, the Explore/Search Workflow operations can be applied to
Explore/Search operations.

- Read [operations](../../../website/docs/resources/operation.md) documentation
  for details

## Pre-requisites: The Operation Identifier

To apply this skill you need either:

(a) an operation id; (b) explicit instruction to examine the latest operation

In the case of (b) (latest) get the actual operation identifier as follows

```bash
uv run ado show related operation --use-latest
```

This will output the id of the latest operation.

## Tips

### Avoiding refetching YAML

`ado get -o yaml` flag outputs YAML to console. It's often useful to redirect
this to a temporary file and work with that to avoid multiple `ado get` calls
for same YAML.

In particular "get datacontainer -o yaml|json" can be large and should be
redirected to a file and loaded with python.

### Large output files

The files created by '-o/--output-format' can be very large e.g. from "show
entities", "show requests" or "show results".

When inspecting these files:

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
uv run ado get operation OPERATION_ID -o yaml
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
2. If it is many hours and the operationType is not search/explore flag that
   there may be a problem
3. If it is many hours and the operationType is search/explore proceed use
   specific techniques in Explore Operation Workflow to determine if its still
   running

### Step 2: Check for existing report

If the operation is finished,

- Check if there is an existing report for this operation in
  `reports/<ado_context_name>/`
- If yes, check if that report indicated the operation was finished
  - If yes, ask the user if they want to replace it with a new report
  - If no, continue with creating new report

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
uv run ado get space SPACE_ID -o yaml
uv run ado describe space SPACE_ID
```

Summarise the: **dimensions** (parameters), **experiments** (actuators,
experiment types), **entity space** structure, and notable **constraints** or
metadata. For deeper context, read operator and experiment documentation under
`website/docs/operators/` and actuator/experiment docs as needed (match
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

To retrieve contents of data container

```bash
uv run ado get datacontainer -o yaml $DATACONTAINER_IDENTIFIER > datacontainer.yaml
```

For each output resource summarize what it is/contains.

## Explore/Search Operation Workflow

The following assumes the General Workflow has been applied.

Explore/Search operations sample entities from a discovery space and make
measurements on them.

Notes:

- If the data for the measurements exists it can be memoized (depends on
  operation parameters)
- The operation parameters will specify the number of entities to sample in some
  way.

Relevant Documentation

- [sample process](../../../website/docs/core-concepts/discovery-spaces.md#sampling-and-measurement)
- [memoization](../../../website/docs/core-concepts/data-sharing.md#memoization)

### Step 1: Get Details on what was Sampled and Measured

```bash
uv run ado show details operation $OPERATION_ID
```

Compare this with the number of samples requested in the operator parameters.

If the state is finished, exit status was successful and all requested samples
were completed there are no issues ->
[examine entities](#step-3-get-entities-and-measurements)

If the state is not finished ->
[Use the diagnose if sampling operation running workflow](#diagnose-if-an-explore-or-search-operation-is-running-workflow).
For all other combinations ->
[Diagnose sampling issues](#step-2-optional-diagnose-sampling-issues)

### Step 2 (Optional): Diagnose sampling issues

First run these two commands to get the metadata on what was requested and
measured, noting the [guidelines on large files](#large-output-files):

```bash
uv run ado show requests operation OPERATION_ID \
  --output-format csv
uv run ado show results operation OPERATION_ID \
  --output-format csv
```

- **requests**: This is metadata on what the sampling operation asked an
  actuator to measure. It includes the timestamp of when the request was
  created - at the moment the completion time is not available. The requests
  contain the results of executing the request
- **results**: This is metadata on an actual measurement triggered by a request
  - ValidMeasurementResult: The experiment executed and return one or more
    observed property values
  - InvalidMeasurementResult: The experiment failed for some reason

From the output of `show requests` and `show results` identify **failed** or
**invalid** rows, **reasons** for invalidity, and anomalies in **timing** or
**ordering** if those columns are present.

### Step 3: Get entities and measurements

To get the data on measurements execute (noting the
[guidelines on large files](#large-output-files)):

```bash
uv run ado show entities operation OPERATION_ID \
  --output-format csv
```

### Step 4: Analyze the Measurement data

Perform an analysis of the measurements, checking e.g. distributions of
metrics, metric outliers, correlations between metrics.
Take into account the domain of the experiment and meaning of metrics
when looking for patterns.

## Diagnose if an Explore or Search Operation is Running Workflow

- Check if the operation is submitting experiments in batches
- Confirm if the operation uses continuous batching (new experiment requested
  once one has finished) or static batch (full batch finishes then next starts)
- Get the requests and results timeseries using `ado show requests` and
  `ado show results`
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
