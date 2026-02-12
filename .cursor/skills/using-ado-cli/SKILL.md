---
name: using-ado-cli
description:
  Guidelines for using ado CLI commands and documenting them correctly. Use when
  writing documentation that includes ado commands, verifying CLI syntax, or
  explaining ado CLI usage patterns to users.
---

# Using the ado CLI

## Command Verification

Before writing or committing documentation with ado CLI commands, verify the
syntax:

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

## Core Commands

### ado create

Creates resources and starts operations.

```bash
# Create a discoveryspace
ado create space -f space.yaml

# Create and start an operation
ado create operation -f operation.yaml
```

**Key point**: `ado create` both defines AND initiates resources.

### ado show

Retrieves details and data from resources.

```bash
# Get resource details
ado show space SPACE_ID

# Get latest results
ado show results operation OPERATION_ID

# Get entities and measurements
ado show entities space SPACE_ID
ado show entities operation OPERATION_ID
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
ado show entities operation op-123

# Get metadata about the operation's results
ado show results operation op-123
```

## Command-Line Shortcuts

### --use-latest

Uses the ID of the most recently created resource of the relevant type.

**Without --use-latest**:

```bash
# Step 1: Create space, note the ID from output
ado create space -f space.yaml
# Output: Created space: space-abc123

# Step 2: Edit operation.yaml to add space-abc123
# Step 3: Create operation
ado create operation -f operation.yaml
```

**With --use-latest**:

```bash
# Step 1: Create space
ado create space -f space.yaml

# Step 2: Create operation using that space automatically
ado create operation -f operation.yaml --use-latest
```

The `--use-latest` flag automatically fills in the space ID from the previous
`ado create space` command.

### --with

Creates a resource from YAML inline and uses it in the current command.

**Without --with**:

```bash
# Create actuator configuration separately
ado create actuatorconfiguration -f actuator.yaml

# Edit operation.yaml to reference the actuator config ID
ado create operation -f operation.yaml
```

**With --with**:

```bash
# Create both in one command
ado create operation -f operation.yaml \
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
ado show operation

# Get details on a specific operation
ado show operation op-123

# Get the entities and measurements
ado show entities operation op-123
```

### Create with dependencies

```bash
# Create everything in one command
ado create operation -f operation.yaml \
  --with space=space.yaml \
  --with actuatorconfiguration=config.yaml
```

### Iterative development

```bash
# Create space
ado create space -f space.yaml

# Validate with dry-run
ado create operation -f operation.yaml --dry-run --use-latest

# Actually create it
ado create operation -f operation.yaml --use-latest
```

## Related Resources

- For creating discoveryspace and operation YAML files, see
  [formulate-discovery-problem](../formulate-discovery-problem/)
- For general development guidelines, see [AGENTS.md](../../../AGENTS.md)
