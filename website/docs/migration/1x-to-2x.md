# Migrating from ado 1.x to ado 2.x

ado 2.x introduces a set of breaking changes that remove obsolete commands and
APIs. This guide describes each breaking change and shows how to update your
workflows.

## Breaking Changes

### Removed: `ado show requests` and `ado show results`

The `ado show requests` and `ado show results` commands have been removed. They
displayed `MeasurementRequest` and `MeasurementResult` metadata for an explore
operation in separate views. The `ado show trace` command supersedes both: it
provides the same information in a single, unified view with additional
capabilities such as field filtering, and YAML output.

#### Before (ado 1.x)

Inspect measurement requests for an operation:

```shell
ado show requests operation randomwalk-0.5.0-123abc -o csv --output-file requests.csv
```

Inspect measurement results metadata for an operation:

```shell
ado show results operation randomwalk-0.5.0-123abc -o csv --output-file results.csv
```

#### After (ado 2.x)

Use `ado show trace` to inspect the trace of measurement requests and optionally
metadata about the individual entity measurements made (the result metadata)

```shell
ado show trace operation randomwalk-0.5.0-123abc -o csv --output-file trace.csv
```

To see one row per entity (equivalent to the detail previously spread across the
two separate commands), add `--unroll-entities`:

<!-- markdownlint-disable line-length -->

```shell
ado show trace operation randomwalk-0.5.0-123abc --unroll-entities -o csv --output-file trace.csv
```

<!-- markdownlint-enable line-length -->

You can also filter by field or hide specific columns:

```shell
# Filter to failed requests only
ado show trace operation randomwalk-0.5.0-123abc --filter status=failed

# Hide a column
ado show trace operation randomwalk-0.5.0-123abc --hide uid
```

For the full `ado show trace` reference, see the
[ado CLI documentation](../getting-started/ado.md#ado-show-trace).

### Removed: `ado get measurementrequest`

The `ado get measurementrequest` command (and its `request` alias) has been
removed. The `ado show trace` command supersedes it: use the `--filter`
option to look up a specific measurement request by ID.

**Before (ado 1.x):**

```shell
ado get request <ID> --from-operation <OP-ID> -o yaml
ado get request <ID> --from-space <SPACE-ID> -o yaml
ado get request <ID> --from-sample-store <SAMPLE-STORE-ID> -o yaml
```

**After (ado 2.x):**

```shell
ado show trace operation <OP-ID> --filter requestid=<ID> -o yaml
ado show trace space <SPACE-ID> --filter requestid=<ID> -o yaml
ado show trace store <SAMPLE-STORE-ID> --filter requestid=<ID> -o yaml
```

### Removed: `ado show summary`

The `ado show summary` command has been removed entirely. The markdown
prose-report output format (`-o md-report`) it provided has no direct
replacement.

For numeric and tabular space statistics (EXPERIMENTS, OPERATIONS,
EXPLORE_OPERATIONS, MEASURED_ENTITIES, SIZE_OF_ENTITY_SPACE,
UNMEASURED_ENTITIES, MATCHING_ENTITIES, MATCHING_WITH_MEASUREMENTS,
ENTITIES_WITH_ALL_MEASUREMENTS, ENTITIES_WITH_PARTIAL_MEASUREMENTS,
MATCHING_ENTITIES_WITH_ALL_MEASUREMENTS), use `ado show stats discoveryspace`.

**Before (ado 1.x):**

```shell
ado show summary space space-abc123-456def
ado show summary space -l issue=123 -o csv --output-file summary.csv
```

**After (ado 2.x):**

```shell
ado show stats discoveryspace space-abc123-456def
ado show stats discoveryspace -l issue=123 -o csv --output-file summary.csv
```

### Removed: `ado show details`

The `ado show details` command has been removed entirely.

**`ado show details discoveryspace`** used to display aggregate statistics about
a space (number of experiments, measured entities, etc.). Use
`ado show stats discoveryspace` for the same numeric statistics, and
`ado show related discoveryspace` to list resources associated with the space.

**`ado show details operation`** used to display statistics stored in operation
metadata after an operation finished. Use `ado show stats operation` for numeric
operation statistics, and `ado show related operation` to list resources
associated with the operation.

**Before (ado 1.x):**

```shell
ado show details space space-abc123-456def
ado show details operation randomwalk-0.5.0-123abc
```

**After (ado 2.x):**

```shell
ado show stats discoveryspace space-abc123-456def
ado show related discoveryspace space-abc123-456def

ado show stats operation randomwalk-0.5.0-123abc
ado show related operation randomwalk-0.5.0-123abc
```
