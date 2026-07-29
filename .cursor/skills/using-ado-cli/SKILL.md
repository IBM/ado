---
name: using-ado-cli
description:
  "Reference for ado CLI command syntax, flags, and usage patterns — covers get,
  create, edit, show, and describe subcommands, output formatting with -o and
  --output-file, convenience flags (--use-latest, --set, --with), debugging with
  -l, and run_experiment for local point testing. Use when writing or verifying
  ado CLI commands, looking up correct command syntax or flags, debugging
  unexpected CLI output, or explaining ado command patterns."
---

# Using the ado CLI

## Command Verification

Always run `--help` before writing any ado command in documentation, comments,
or code. Do not rely on memory or analogy to other CLIs.

```bash
# Verify top-level command
uv run ado [COMMAND] --help

# Verify subcommands
uv run ado [COMMAND] [SUBCOMMAND] --help
uv run ado [COMMAND] [SUBCOMMAND1] [SUBCOMMAND2] --help
```

**Check**:

- Command and subcommand names are correct
- Options are spelled correctly (e.g., `--use-latest` not `--latest`)
- Required arguments are included
- Optional flags match actual CLI behavior

## Output Format and File Handling

For `ado get` and `ado show` subcommands:

- `-o` / `--output` selects the **output format** (for example `yaml`, `table`,
  `csv`, `json`, or `stats`; allowed values depend on the command — use
  `--help`).
- `--output-file PATH` writes formatted output to **PATH** instead of stdout.

Shell redirects (`>`) work for simple cases. Prefer `--output-file` when:

- **Pre-flight checks**: ado validates the path is writable before fetching,
  avoiding failure after a long data fetch.
- **Stdout pollution**: `--output-file` writes only formatted output to the
  file; logs stay on stderr. A redirect captures both.
- **Table truncation**: terminal-width column truncation applies to redirected
  output but not to `--output-file`.

```bash
uv run ado get space SPACE_ID -o yaml > space.yaml
uv run ado show measurements operation OPERATION_ID -o csv --output-file measurements.csv
```

## Commands That do not exist

These plausible-sounding commands do not exist in ado. Do not write them:

| ❌ Does not exist | ✅ Correct equivalent                    |
| ----------------- | ---------------------------------------- |
| `ado run`         | `ado create operation -f op.yaml`        |
| `ado start`       | `ado create operation -f op.yaml`        |
| `ado execute`     | `ado create operation -f op.yaml`        |
| `ado launch`      | `ado create operation -f op.yaml`        |
| `ado list`        | `ado get spaces` / `ado get operations`  |
| `ado status`      | `ado show stats discoveryspace SPACE_ID` |

**Key principle**: `ado create operation` both _defines_ and _starts_ the
operation in a single command. There is no separate "run" step.

## Point Testing with run_experiment

`run_experiment` is a **separate CLI entry point** (not an `ado` subcommand) for
running a single entity through an experiment locally, without creating a space
or operation. It is the correct tool for functional validation of custom
experiments.

```bash
uv run run_experiment PATH_TO_POINT_YAML
```

A point YAML has the form:

```yaml
entity:
  param_a: value_a
  param_b: value_b

experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: my_experiment
```

It prints the result as a pandas Series and exits. No metastore or Ray cluster
needed beyond a local Ray instance (started automatically).

## Core Commands

### ado get

Lists resources of a given type and gets resource YAML

```bash
# List all spaces
uv run ado get spaces

# Get the YAML for a space (console)
uv run ado get space SPACE_ID -o yaml

# Or write the same YAML to a file
uv run ado get space SPACE_ID -o yaml --output-file space.yaml

# Get the latest space as YAML
uv run ado get space --use-latest -o yaml

# Get the latest operation as YAML
uv run ado get operation --use-latest -o yaml

# Get measurement statistics for all operations
uv run ado get operations -o stats --output-file operations-stats.txt
uv run ado get operation OPERATION_ID -o stats --no-trunc

# Get statistics for all discovery spaces
uv run ado get spaces -o stats --output-file spaces-stats.txt
uv run ado get space SPACE_ID -o stats --no-trunc

# Get statistics for all sample stores
uv run ado get samplestores -o stats --output-file samplestores-stats.txt
uv run ado get samplestore SAMPLESTORE_ID -o stats --no-trunc

# Get statistics for all data containers
uv run ado get datacontainers -o stats --output-file datacontainers-stats.txt
uv run ado get datacontainer DATACONTAINER_ID -o stats --no-trunc

# Get all resources of a type reachable from an anchor resource (--related-to)
uv run ado get operations --related-to samplestore=STORE_ID
uv run ado get spaces --related-to samplestore=STORE_ID -o name
uv run ado get operations --related-to discoveryspace=SPACE_ID --filter config.metadata.name=OP_NAME
```

`--related-to` filters results to resources reachable from the given anchor
(`kind=id`). Shorthand aliases work (e.g. `store=STORE_ID`). Not supported for
`actuator`, `experiment`, `operator`, or `context`. Cannot be combined with a
direct resource ID or `--use-latest`. Can be combined with `--filter`,
`--label`, `--matching-point`, `--matching-space`, and `--matching-space-id`.

`-o stats` extends the table with statistics columns:

- **Operations**: `TOTAL_RESULTS`, `SUCCESSFUL_RESULTS`, `FAILED_RESULTS`,
  `MEASURED_ENTITIES`.
- **Discovery Spaces**: `EXPERIMENTS`, `OPERATIONS`, `EXPLORE_OPERATIONS`,
  `MEASURED_ENTITIES`.
- **Sample Stores**: `ENTITIES`, `RESULTS`, `EXPERIMENTS`.
- **Data Containers**: `TABLES`, `LOCATIONS`, `KEY_VALUES`, `DATA_BYTES`.

### ado create

Creates resources and starts operations.

```bash
# Create a discoveryspace
uv run ado create space -f space.yaml

# Create and start an operation
uv run ado create operation -f operation.yaml
```

**Key point**: `ado create` both defines AND initiates resources.

### ado edit

Updates **metadata** (name, description, labels, etc.) for metastore resources.

```bash
# Interactive (default editor: nano, or $ADO_EDITOR)
uv run ado edit space SPACE_ID

# Non-interactive: default is an inline YAML or JSON patch (-p / --patch),
# like oc
uv run ado edit space SPACE_ID -p "labels: { team: research }"

# Or merge from a file
uv run ado edit space SPACE_ID --patch-file meta.yaml
```

Always prefer a non-interative edit with `-p` / `--patch` or `--patch-file`. Use
`uv run ado edit --help` for current options.

### ado show

Retrieves details and data from resources.

```bash
# Inspect the trace of measurement requests for an operation
uv run ado show trace operation OPERATION_ID

# Get entities and measurements
uv run ado show measurements space SPACE_ID
uv run ado show measurements operation OPERATION_ID

# Show in-depth statistics (more columns than ado get -o stats)
# No IDs = all resources of that type
uv run ado show stats operation
uv run ado show stats operation --use-latest -o json
uv run ado show stats discoveryspace SPACE_ID
uv run ado show stats samplestore -l key=value
# Include DESCRIPTION and LABELS columns
uv run ado show stats operation --details
```

### ado describe

Outputs a human readable description

```bash
# Output a description of a space
# Dimensions, values, experiments
uv run ado describe space SPACE_ID

#Output a description of an experiment
# (input params, output params etc.)
uv run ado describe experiment EXPERIMENT_ID
```

## Debugging

If commands are not given expected output use the -l flag to activate different
log levels

e.g. for debug level logs

```bash
uv run ado -lDEBUG [COMMAND]
```

## show Commands Quick Reference

Entities are points in the discovery space — constitutive properties (inputs)
plus measured properties (outputs).

<!-- markdownlint-disable line-length -->

| Command                       | What It Shows                                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `show measurements operation` | Entities (inputs) and their measurements (outputs) from this operation                                                   |
| `show measurements space`     | All entities and measurements collected in this space                                                                    |
| `show trace operation`        | The trace of measurement requests made during an explore operation. Optionally can show per entity measurement metadata  |
| `show stats operation`        | In-depth stats: results (total/successful/failed), measured entities, plus request-level counts. No IDs = all operations |
| `show stats discoveryspace`   | In-depth stats: experiments, operations, measured entities, plus full entity-space coverage columns. No IDs = all spaces |
| `show stats samplestore`      | In-depth stats: entities, results, and experiments counts. No IDs = all sample stores                                    |
| `show stats datacontainer`    | In-depth stats: tables, locations, key-values, and data bytes. No IDs = all data containers                              |

<!-- markdownlint-enable line-length -->

```bash
# Measurement data
uv run ado show measurements operation op-123

# Inspect the trace of measurement requests for an operation
uv run ado show trace operation op-123
```

## Command-Line Shortcuts

### --use-latest

Queries the current context's metastore to find the most recently created
resource of the given type.

**Without --use-latest**:

```bash
# Step 1: Create space, note the ID from output
uv run ado create space -f space.yaml
# Output: Created space: space-abc123

# Step 2: Edit operation.yaml to add space-abc123
# Step 3: Create operation
uv run ado create operation -f operation.yaml
```

**With --use-latest**:

```bash
# Step 1: Create space
uv run ado create space -f space.yaml

# Step 2: Create operation using that space automatically
uv run ado create operation -f operation.yaml --use-latest space
```

The `--use-latest` flag automatically fills in the latest space ID.

### --set

Overrides individual fields in a resource YAML at creation time without editing
the file. Takes `path=JSON_document` pairs; can be used multiple times.

```bash
# Override the sample store used by a space
uv run ado create space -f space.yaml --set sampleStoreIdentifier=my_store

# Override a nested operation parameter
uv run ado create operation -f operation.yaml --set parameters.budget=100
```

### --with

Creates a resource from YAML inline and uses it in the current command.

**Without --with**:

```bash
# Create actuator configuration separately
uv run ado create actuatorconfiguration -f actuator.yaml

# Edit operation.yaml to reference the actuator config ID
uv run ado create operation -f operation.yaml
```

**With --with**:

```bash
# Create both in one command
uv run ado create operation -f operation.yaml \
  --with space=space.yaml \
  --with actuatorconfiguration=actuator.yaml
```

This creates the space and actuator configuration, then automatically references
them when creating the operation.

## Documentation Best Practices

When writing documentation with ado commands:

1. **Always verify** the command syntax with `--help`
2. **Use realistic IDs** in examples (e.g., `space-abc123` not `SPACE_ID` in
   code blocks where actual output is shown)
3. **Prefer shortcuts** (`--use-latest`, `--with`) in tutorials to reduce
   friction
4. **Explain terminology** the first time: "entities (the inputs and their
   measurements)"

## Common Patterns

### Query workflow

```bash
# List all operations
uv run ado get operations

# Get details on a specific operation (YAML to a file)
uv run ado get operation op-123 -o yaml --output-file op-123.yaml

# Get the entities and measurements
uv run ado show measurements operation op-123
```

### Create with dependencies

```bash
# Create everything in one command
uv run ado create operation -f operation.yaml \
  --with space=space.yaml \
  --with actuatorconfiguration=config.yaml
```

### Iterative development

```bash
# Create space
uv run ado create space -f space.yaml

# Validate with dry-run
uv run ado create operation -f operation.yaml --dry-run --use-latest space

# If dry-run fails: check error message, fix the YAML, re-run dry-run
# Common issues: missing experiment references, invalid parameter types,
# unresolvable --use-latest (no resource of that type exists yet)

# Once dry-run passes, actually create it
uv run ado create operation -f operation.yaml --use-latest space
```

## Related Resources

- For creating and structuring resource YAML files, see
  [resource-yaml-creation](../resource-yaml-creation/)
- For creating discoveryspace and operation YAML files, see
  [formulate-discovery-problem](../formulate-discovery-problem/)
- For general development guidelines, see [AGENTS.md](../../../AGENTS.md)
