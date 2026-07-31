---
name: query-ado-data
description:
  Query ado metadata and measurement data using CLI commands. Use when the user
  needs to find resources, filter by metadata, retrieve entities and
  measurements, or get resource schemas. Covers metastore queries (operations,
  discoveryspaces, samplestores, datacontainers, actuatorconfigurations) and
  samplestore queries (entities and measurements).
---

# Query ado Data

ado stores data in two places:

1. **Metastore**: Metadata about all resources (operations, discoveryspaces,
   samplestores, datacontainers, actuatorconfigurations)
2. **Samplestores**: Entities and measurements made on them

## Guidelines

- When getting a list of resources the output will always be tabular formatted
  string
- Do not change context to answer a query unless specifically requested -
  metadata and data is context specific

### Fast Querying

DOs:

- IMPORTANT Before deciding on what to query check the resource schema to
  confirm what is available in metadata - ado template RESOURCETYPE
  --include-schema
- Use Server side filtering
  - prefer --filter or --matching to fetching metadata and filtering on client
    side
- Fetch metadata over fetching data
  - if a query can be answered via metadata it is much faster
  - filter via metadata first if possible, before obtaining data
- Consider writing a script directly using SQLResourceStore API if the CLI is
  not expressive enough BEFORE fetching data
  - you can make batch requests e.g. getResources - much faster than one-by-one
    requests

DONTs

- Do not fetch discoveryspace or operation data for summary queries
  - Do not use: ado show measurements, ado show trace
  - Do not instantiating DiscoverySpace instances or SQLStore instance
- Only use these commands or classes when drilling down on a narrow set of
  resources

### Using Resource models

Each resource has a pydantic model. If working in code you can use these models

- discoveryspace, ado/core/discoveryspace/resource.py:
  DiscoverySpaceResource
- samplestore, ado/core/samplestore/resource.py: SampleStoreResource
- datacontainer, ado/core/datacontainer/resource.py:
  DataContainerResource
- operation, ado/core/operation/resource.py: OperationResource
- actuatorconfiguration, ado/core/actuatorconfiguration/resource.py:
  ActuatorConfigurationResource

## Querying Metadata

### Listing Resources

Get a general overview of what's present:

```bash
uv run ado get $RESOURCETYPE --details
```

Returns an age-sorted list (most recent last) of resources of the specified
type.

**Resource types**: `operations` (`op`), `discoveryspaces` (`space`),
`samplestores` (`store`), `datacontainers` (`dcr`), `actuatorconfigurations`
(`ac`)

### Resource Statistics

`-o stats` adds statistics columns to the table without fetching full resource
data. Supported for **operations**, **discovery spaces**, **sample stores**, and
**data containers**.

```bash
# Operations
uv run ado get operations -o stats --output-file operations-stats.txt
uv run ado get operation OPERATION_ID -o stats --no-trunc

# Discovery Spaces
uv run ado get spaces -o stats --output-file spaces-stats.txt
uv run ado get space SPACE_ID -o stats --no-trunc

# Sample Stores
uv run ado get samplestores -o stats --output-file samplestores-stats.txt
uv run ado get samplestore SAMPLESTORE_ID -o stats --no-trunc

# Data Containers
uv run ado get datacontainers -o stats --output-file datacontainers-stats.txt
uv run ado get datacontainer DATACONTAINER_ID -o stats --no-trunc
```

**Operations** extra columns: `TOTAL_RESULTS`, `SUCCESSFUL_RESULTS`,
`FAILED_RESULTS`, `MEASURED_ENTITIES` (distinct entities with at least one
result).

**Discovery Spaces** extra columns: `EXPERIMENTS`, `OPERATIONS`,
`EXPLORE_OPERATIONS`, `MEASURED_ENTITIES`.

**Sample Stores** extra columns: `ENTITIES`, `RESULTS`, `EXPERIMENTS`.

**Data Containers** extra columns: `TABLES`, `LOCATIONS`, `KEY_VALUES`,
`DATA_BYTES`.

### Filtering Resources

Filter resources based on metadata fields using MySQL JSON Path queries:

```bash
uv run ado get $RESOURCETYPE --filter 'path=candidate'
```

- Use single quotes around the candidate (required for strings, dictionaries,
  arrays)
- Path is dot-separated (e.g., `config.metadata.labels`)
- Candidate is a valid JSON value
- Can specify `--filter` multiple times (all filters must match)

**Examples:**

```bash
# Find operations using a specific operator
uv run ado get operations --filter 'config.operation.module.moduleClass=RayTune'

# Find spaces with a specific experiment
uv run ado get spaces --filter 'config.experiments={"experiments":{"identifier":"finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0"}}'

# Combine multiple filters
uv run ado get operations --filter 'config.operation.parameters.batchSize=1' \
  --filter 'status=[{"event": "finished", "exit_state": "success"}]'
```

For extensive examples, see `docs/resources/metastore.md`.

### Filtering by Labels

Filter resources by labels:

```bash
uv run ado get $RESOURCETYPE -l key=value
```

Can specify multiple times (all labels must match):

```bash
uv run ado get operations -l labelone=valueone -l label_two=value_two
```

### Matching Spaces

Find spaces matching a point or another space:

```bash
# Match spaces containing a specific entity point
uv run ado get space --matching-point point.yaml

# Match spaces similar to another space (by ID)
uv run ado get space --matching-space-id space-abc123-456def

# Match spaces similar to a space configuration (without creating it)
uv run ado get space --matching-space space.yaml
```

**Note**: `--matching-point`, `--matching-space`, and `--matching-space-id` are
exclusive to spaces and override `--filter` and `--label`.

### Related Resources

#### ado show related

Get IDs of all resources related to another resource (parent or child),
traversing the full relationship graph:

```bash
uv run ado show related $RESOURCETYPE [RESOURCE_ID] [--use-latest]
```

**Supported types**: `operation` (`op`), `samplestore` (`store`),
`discoveryspace` (`space`)

**Example:**

```bash
uv run ado show related space space-abc123-456def
```

#### ado get --related-to

Filter `ado get` results to resources related to a specific source resource,
including multi-hop relationships (e.g. operations linked to a space that is
linked to a store). Specify the source as `kind=id` (shorthand aliases supported):

```bash
uv run ado get $RESOURCETYPE --related-to kind=SOURCE_ID
```

Not supported for `actuator`, `experiment`, `operator`, or `context`. The source
kind must differ from the requested resource kind. Cannot be combined with a
direct resource ID or `--use-latest`. Can be combined with `--filter`,
`--label`, `--matching-point`, `--matching-space`, and `--matching-space-id`.

**Examples:**

```bash
# All operations related to a sample store
uv run ado get operations --related-to samplestore=STORE_ID

# All spaces related to a sample store (name only)
uv run ado get spaces --related-to samplestore=STORE_ID -o name

# Operations related to a space, narrowed by a metadata filter
uv run ado get operations --related-to discoveryspace=SPACE_ID \
  --filter config.metadata.name=my-op
```

## Querying Data

### Show Entities

Get entities and their measurements from a space or operation:

```bash
uv run ado show measurements RESOURCE_TYPE [RESOURCE_ID] \
                  [--use-latest] [--file | -f <file.yaml>]\
                  [--property-format {observed | target}] \
                  [--output | -o {csv | json | table}] \
                  [--output-file <path>] \
                  [--property <property-name>] \
                  [--include {sampled | matching | missing | unsampled}] \
                  [--aggregate {mean | median | variance | std | min | max}]
```

**Resource types**: `operation` (`op`), `discoveryspace` (`space`)

**Key options:**

- `--include` (spaces only): `sampled`, `unsampled`, `matching`, `missing`
- `--property-format`: `observed` (one row per entity) or `target` (one row per
  entity-experiment pair)
- `--output` (or `-o`): `csv`, `json`, or `table`
- `--output-file` specifies a file path to write the output to. If not provided,
  output is written to stdout.
- `--property`: Filter specific properties (can specify multiple times)
- `--aggregate`: Aggregate multiple values

**Examples:**

```bash
# Show matching entities in a space as CSV
uv run ado show measurements space space-abc123-456def --include matching \
                                             --property-format target \
                                             -o csv --output-file space-abc123-456def-entities.csv

# Show entities from an operation with specific properties
uv run ado show measurements operation randomwalk-0.5.0-123abc \
                  --property my-property-1 \
                  --property my-property-2 \
                  -o csv --output-file randomwalk-0.5.0-123abc.csv
```

### Show Trace

Get the trace of measurement requests made during an operation:

```bash
uv run ado show trace operation [RESOURCE_ID] [--use-latest] \
                         [--unroll-entities] \
                         [--output | -o <csv | json | table>] \
                         [--output-file <path>] \
                         [--filter <expr>] \
                         [--hide <field>]
```

<!-- markdownlint-disable line-length -->

| Flag                | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `--use-latest`      | Use the most recently created operation in the current context              |
| `--unroll-entities` | Include per-entity result metadata (validity, invalidity reasons, etc.)     |
| `-o` / `--output`   | Output format: `csv`, `json`, or `table`                                    |
| `--output-file`     | Write output to a file instead of stdout                                    |
| `--filter`          | Filter rows by expression                                                   |
| `--hide`            | Hide a field from the output                                                |

<!-- markdownlint-enable line-length -->

**Example — request-level view:**

```bash
uv run ado show trace operation randomwalk-0.5.0-123abc \
  -o csv --output-file randomwalk-0.5.0-123abc-trace.csv
```

**Example — per-entity result metadata:**

```bash
uv run ado show trace operation randomwalk-0.5.0-123abc --unroll-entities \
  -o csv --output-file randomwalk-0.5.0-123abc-trace-entities.csv
```

## Getting Schemas

Get JSON schemas for resource types:

```bash
uv run ado template $RESOURCETYPE --include-schema
```

**Example:**

```bash
# Get space template with schema
uv run ado template space --include-schema

# Get operation template with schema for a specific operator
uv run ado template operation --operator-name OPERATOR_NAME --include-schema
```

## Common Patterns

### Find operations that finished successfully

<!-- markdownlint-disable line-length -->
```bash
uv run ado get operations --filter 'status=[{"event": "finished", "exit_state": "success"}]'
```
<!-- markdownlint-enable line-length -->

### Find spaces containing a specific model

```bash
uv run ado get spaces --filter 'config.entitySpace={"propertyDomain":{"values":["mistral-7b-v0.1"]}}'
```

### Export operation entities to CSV

```bash
uv run ado show measurements operation OPERATION_ID -o csv --output-file OPERATION_ID_entities.csv
```

### Get all resources related to a space

```bash
uv run ado show related space SPACE_ID
```

## Advanced Filtering

The metastore class can provide more powerful querying via scripts. See
ado/metastore/sqlstore.py

## References

When modifying or creating code while using this skill, follow:

- [AGENTS.md](../../../AGENTS.md)
- [plugin-development.mdc](../../rules/plugin-development.mdc) (if working with
  plugins)
