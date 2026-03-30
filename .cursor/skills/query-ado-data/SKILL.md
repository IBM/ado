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
  - prefer --query or --matching to fetching metadata and filtering on client
    side
- Fetch metadata over fetching data
  - if a query can be answered via metadata it is much faster
  - filter via metadata first if possible, before obtaining data
  - IMPORTANT: ado show details space can be slow as it internally fetches
    spaces data to calculate - prefer using metadata
- Consider writing a script directly using SQLResourceStore API if the CLI is
  not expressive enough BEFORE fetching data
  - you can make batch requests e.g. getResources - much faster than one-by-one
    requests

DONTs

- Do not fetch discoveryspace or operation data for summary queries
  - Do not use: ado show entities, ado show requests, ado show results, ado show
    details)
  - Do not instantiating DiscoverySpace instances or SQLStore instance
- Only use these commands or classes when drilling down on a narrow set of
  resources

### Using Resource models

Each resource has a pydantic model. If working in code you can use these models

- discoveryspace, orchestrator/core/discoveryspace/resource.py:
  DiscoverySpaceResource
- samplestore, orchestrator/core/samplestore/resource.py: SampleStoreResource
- datacontainer, orchestrator/core/datacontainer/resource.py:
  DataContainerResource
- operation, orchestrator/core/operation/resource.py: OperationResource
- actuatorconfiguration, orchestrator/core/actuatorconfiguration/resource.py:
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

### Filtering Resources

Filter resources based on metadata fields using MySQL JSON Path queries:

```bash
uv run ado get $RESOURCETYPE --query 'path=candidate'
```

- Use single quotes around the candidate (required for strings, dictionaries,
  arrays)
- Path is dot-separated (e.g., `config.metadata.labels`)
- Candidate is a valid JSON value
- Can specify `--query` multiple times (all filters must match)

**Examples:**

```bash
# Find operations using a specific operator
uv run ado get operations -q 'config.operation.module.moduleClass=RayTune'

# Find spaces with a specific experiment
uv run ado get spaces -q 'config.experiments={"experiments":{"identifier":"finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0"}}'

# Combine multiple filters
uv run ado get operations -q 'config.operation.parameters.batchSize=1'
-q 'status=[{"event": "finished", "exit_state": "success"}]'
```

For extensive examples, see `website/docs/resources/metastore.md`.

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
exclusive to spaces and override `--query` and `--label`.

### Related Resources

Get IDs of resources related to another resource (parent or child):

```bash
uv run ado show related $RESOURCETYPE [RESOURCE_ID] [--use-latest]
```

**Supported types**: `operation` (`op`), `samplestore` (`store`),
`discoveryspace` (`space`)

**Example:**

```bash
uv run ado show related space space-abc123-456def
```

### Get Resource Details

View detailed information about a specific resource:

```bash
uv run ado show details $RESOURCETYPE [RESOURCE_ID] [--use-latest]
```

## Querying Data

### Show Entities

Get entities and their measurements from a space or operation:

```bash
uv run ado show entities $RESOURCETYPE [RESOURCE_ID] [--use-latest] \
                  [--include {sampled | matching | missing | unsampled}] \
                  [--property-format {observed | target}] \
                  [--output-format {console | csv | json}] \
                  [--property <property-name>] \
                  [--aggregate {mean | median | variance | std | min | max}]
```

**Resource types**: `operation` (`op`), `discoveryspace` (`space`)

**Key options:**

- `--include` (spaces only): `sampled`, `unsampled`, `matching`, `missing`
- `--property-format`: `observed` (one row per entity) or `target` (one row per
  entity-experiment pair)
- `--output-format`: `console`, `csv`, or `json`
- `--property`: Filter specific properties (can specify multiple times)
- `--aggregate`: Aggregate multiple values

**Examples:**

```bash
# Show matching entities in a space as CSV
uv run ado show entities space space-abc123-456def --include matching \
                                             --property-format target \
                                             --output-format csv

# Show entities from an operation with specific properties
uv run ado show entities operation randomwalk-0.5.0-123abc \
                  --property my-property-1 \
                  --property my-property-2 \
                  --output-format json
```

### Show Requests

Get measurement requests sent during an operation:

```bash
uv run ado show requests operation [RESOURCE_ID] [--use-latest] \
                            [--output-format {console | csv | json}] \
                            [--hide <field>]
```

**Example:**

```bash
uv run ado show requests operation randomwalk-0.5.0-123abc -o csv
```

### Show Results

Get measurement results metadata (valid/invalid status, etc.):

```bash
uv run ado show results operation [RESOURCE_ID] [--use-latest] \
                           [--output-format {console | csv | json}] \
                           [--hide <field>]
```

**Note**: This shows metadata about results (validity, reasons for invalidity),
not the actual measurement values.

**Example:**

```bash
uv run ado show results operation randomwalk-0.5.0-123abc -o csv
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

```bash
uv run ado get operations -q 'status=[{"event": "finished", "exit_state": "success"}]'
```

### Find spaces containing a specific model

```bash
uv run ado get spaces -q 'config.entitySpace={"propertyDomain":{"values":["mistral-7b-v0.1"]}}'
```

### Export operation entities to CSV

```bash
uv run ado show entities operation OPERATION_ID --output-format csv
```

### Get all resources related to a space

```bash
uv run ado show related space SPACE_ID
```

## Advanced Filtering

The metastore class can provide more powerful querying via scripts. See
orchestrator/metastore/sqlstore.py

## References

When modifying or creating code while using this skill, follow:

- [AGENTS.md](../../../AGENTS.md)
- [plugin-development.mdc](../../rules/plugin-development.mdc) (if working with
  plugins)
