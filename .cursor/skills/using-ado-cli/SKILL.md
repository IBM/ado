---
name: using-ado-cli
description:
  Guidelines for using ado CLI commands and documenting them correctly. Use when
  writing documentation that includes ado commands, verifying CLI syntax, or
  explaining ado CLI usage patterns to users.
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

## Commands That do not exist

These plausible-sounding commands do not exist in ado. Do not write them:

| ❌ Does not exist | ✅ Correct equivalent |
| --- | --- |
| `ado run` | `ado create operation -f op.yaml` |
| `ado start` | `ado create operation -f op.yaml` |
| `ado execute` | `ado create operation -f op.yaml` |
| `ado launch` | `ado create operation -f op.yaml` |
| `ado list` | `ado get spaces` / `ado get operations` |
| `ado status` | `ado show details space SPACE_ID` |

**Key principle**: `ado create operation` both *defines* and *starts* the operation
in a single command. There is no separate "run" step.

## Point Testing with run_experiment

`run_experiment` is a **separate CLI entry point** (not an `ado` subcommand) for
running a single entity through an experiment locally, without creating a space
or operation. It is the correct tool for functional validation of custom experiments.

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

Lists resources of a given and gets resource YAML

```bash
#List all spaces
uv run ado get spaces

#Get the YAML for a space
uv run ado get space SPACE_ID -o yaml
```

### ado create

Creates resources and starts operations.

```bash
# Create a discoveryspace
uv run ado create space -f space.yaml

# Create and start an operation
uv run ado create operation -f operation.yaml
```

**Key point**: `ado create` both defines AND initiates resources.

### ado show

Retrieves details and data from resources.

```bash
# Get a summary of what has been sampled from the space
uv run ado show details space SPACE_ID

# Get latest results
uv run ado show results operation OPERATION_ID

# Get entities and measurements
uv run ado show entities space SPACE_ID
uv run ado show entities operation OPERATION_ID
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

If commands are not given expected output use
the -l flag to activate different log levels

e.g. for debug level logs

```bash
uv run ado -lDEBUG [COMMAND]
```

## Terminology

### Entities

Entities represent points in the discovery space with:

- **Constitutive properties** (inputs/priors) - what defines the point
- **Measured properties** (outputs/posteriors) - what was observed

### Understanding show Commands

<!-- markdownlint-disable line-length -->

| Command                   | What It Shows                                                            |
| ------------------------- | ------------------------------------------------------------------------ |
| `show entities operation` | Entities (inputs) and their measurements (outputs) from this operation   |
| `show entities space`     | All entities and measurements collected in this space                    |
| `show results operation`  | Results **metadata** from this operation (not the full measurement data) |

<!-- markdownlint-enable line-length -->

**Example distinction**:

```bash
# Get the actual measurement data for entities
uv run ado show entities operation op-123

# Get metadata about the operation's results
uv run ado show results operation op-123
```

## Command-Line Shortcuts

### --use-latest

Uses the ID of the most recently created resource of the relevant type.

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
uv run ado create operation -f operation.yaml --use-latest
```

The `--use-latest` flag automatically fills in the space ID from the previous
`ado create space` command.

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
3. **Show expected output** when helpful for clarity
4. **Prefer shortcuts** (`--use-latest`, `--with`) in tutorials to reduce
   friction
5. **Explain terminology** the first time: "entities (the inputs and their
   measurements)"

### Example Documentation Pattern

```markdown
## Creating and Running an Operation

First, create your discovery space:

\`\`\`bash ado create space -f space.yaml \`\`\`

Then create and start the operation, automatically using the space you just
created:

\`\`\`bash ado create operation -f operation.yaml --use-latest space \`\`\`

View the entities (inputs) and their measurements (outputs):

\`\`\`bash ado show entities operation --use-latest \`\`\`
```

## Common Patterns

### Query workflow

```bash
# List all operations
uv run ado get operations

# Get details on a specific operation
uv run ado get operation -o yaml op-123

# Get the entities and measurements
uv run ado show entities operation op-123
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
uv run ado create operation -f operation.yaml --dry-run --use-latest

# Actually create it
uv run ado create operation -f operation.yaml --use-latest
```

## Related Resources

- For creating discoveryspace and operation YAML files, see
  [formulate-discovery-problem](../formulate-discovery-problem/)
- For general development guidelines, see [AGENTS.md](../../../AGENTS.md)
