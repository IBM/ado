# Migrating from ado 1.x to ado 2.x

ado 2.x introduces a set of breaking changes that remove obsolete commands and
APIs. There are no deprecation warnings or backwards-compatibility shims — the
removed functionality no longer exists at runtime. This guide describes each
breaking change and shows how to update your workflows.

## Breaking Changes

### Removed: `ado show requests` and `ado show results`

The `ado show requests` and `ado show results` commands have been removed.
They displayed `MeasurementRequest` and `MeasurementResult` metadata for an
operation in separate views. The `ado show trace` command supersedes both: it
provides the same information in a single, unified view with additional
capabilities such as entity unrolling, field filtering, and YAML output.

#### Before (ado 1.x)

Inspect measurement requests for an operation:

```shell
ado show requests operation randomwalk-0.5.0-123abc -o csv --output-file requests.csv
```

Inspect measurement results for an operation:

```shell
ado show results operation randomwalk-0.5.0-123abc -o csv --output-file results.csv
```

#### After (ado 2.x)

Use `ado show trace` to inspect both requests and results in a single command:

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
