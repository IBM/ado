# Migrating from ado 1.x to ado 2.x

ado 2.x introduces a set of breaking changes that remove obsolete commands and
APIs. This guide describes each breaking change and shows how to update your
workflows.

## Non-breaking changes

### Renamed: `ado show entities` → `ado show measurements`

The `ado show entities` command has been renamed to `ado show measurements`. All
options and behaviour are unchanged; only the command name differs.

**Before (ado 1.x):**

```shell
ado show entities space <space-id>
ado show entities operation <operation-id> --property-format target -o csv
```

**After (ado 2.x):**

```shell
ado show measurements space <space-id>
ado show measurements operation <operation-id> --property-format target -o csv
```

### Renamed: `operationType: search` → `operationType: explore`

The `explore` operation type was previously serialised as `"search"`. It is now
serialised as `"explore"` to match what is shown in `ado get operators`.

**Existing YAML files and stored operations continue to work without
modification.** The value `"search"` is transparently accepted and treated as
`"explore"` when loading any resource.

To update stored operation records to the new canonical value, run:

```shell
ado upgrade operations
```

New operations and YAML files should use `operationType: explore`.

## Breaking Changes

### Renamed: Python import package `orchestrator` → `ado`

The Python import package has been renamed from `orchestrator` to `ado`. This
is a breaking change for anyone who writes Python code that directly imports
from the package — custom actuators, operators, custom experiments, or scripts.

**Who is affected:** any code that contains `from orchestrator.` or
`import orchestrator.` statements, and any samplestore or operation YAML files
that contain `moduleName` values beginning with `orchestrator.`.

**Before (ado 1.x):**

```python
from orchestrator.schema.property import ConstitutiveProperty
from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.experiment import Experiment
```

**After (ado 2.x):**

```python
from ado.schema.property import ConstitutiveProperty
from ado.modules.actuators.custom_experiments import custom_experiment
from ado.schema.experiment import Experiment
```

**YAML files:** if you have samplestore or operation YAML files with a
`moduleName` field, update the value from `orchestrator.*` to `ado.*`:

```yaml
# Before
moduleName: orchestrator.modules.samplestores.my_store

# After
moduleName: ado.modules.samplestores.my_store
```

To find all affected imports and module names in your codebase, run:

<!-- markdownlint-disable line-length -->

```shell
grep -rn "from orchestrator\|import orchestrator" .
grep -rn "orchestrator\." --include="*.yaml" --include="*.json" .
```

<!-- markdownlint-enable line-length -->

### Renamed: `--query` → `--filter` in `ado get` and `ado show` commands

The `--query` long flag has been renamed to `--filter` across `ado get`,
`ado show trace`, and `ado show stats`. The short alias `-q` is unchanged.

**Before (ado 1.x):**

```shell
ado get operations --query labels.issue=123
ado show trace operation <op-id> --query status=failed
```

**After (ado 2.x):**

```shell
ado get operations --filter labels.issue=123
ado show trace operation <op-id> --filter status=failed
```

### Removed: legacy migrator system in `ado upgrade`

The legacy v1-era migrator system has been removed from `ado upgrade`. The
individual migration steps it provided (e.g. `entitysource_to_samplestore`,
`properties_field_removal`, `actuators_field_removal`, and several others) are
no longer available.

If you have ado resources that were created with a very early 1.x release and
have never been migrated, those resources can no longer be automatically
upgraded.

### Removed: `ado show requests` and `ado show results`

The `ado show requests` and `ado show results` commands have been removed. They
displayed `MeasurementRequest` and `MeasurementResult` metadata for an explore
operation in separate views. The `ado show trace` command supersedes both: it
provides the same information in a single, unified view with additional
capabilities such as field filtering, and YAML output.

**Before (ado 1.x):**

Inspect measurement requests for an operation:

```shell
ado show requests operation randomwalk-0.5.0-123abc -o csv --output-file requests.csv
```

Inspect measurement results metadata for an operation:

```shell
ado show results operation randomwalk-0.5.0-123abc -o csv --output-file results.csv
```

**After (ado 2.x):**

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
[ado CLI documentation](../cli-reference/index.md#ado-show-trace).

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
