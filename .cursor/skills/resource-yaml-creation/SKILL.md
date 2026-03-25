---
name: resource-yaml-creation
description: |
  Guidance for creating ado resource YAML files (discoveryspace, operation,
  actuatorconfiguration, samplestore). Covers metadata conventions, dynamic
  reference resolution with --use-latest/--with/--set, space design principles,
  avoiding duplicate resources, and validation. Use when creating or editing
  any ado resource YAML file.
---

# Creating ado Resource YAML Files

For CLI command syntax, see [using-ado-cli](../using-ado-cli/SKILL.md). For full
problem formulation workflow, see
[formulate-discovery-problem](../formulate-discovery-problem/SKILL.md).

## Metadata Fields

Every resource YAML should include a `metadata` block. The CLI uses these for
display (`ado get --details`) and filtering (`ado get --label`,
`ado get --query`).

```yaml
metadata:
  name: my_space # short human-readable identifier
  description: | # longer explanation of purpose
    Optimize learning rate and batch size for ResNet training.
  labels:
    project: my_project # arbitrary key=value pairs for filtering
    team: ml_team
```

- `name` and `description` are shown by `ado get --details`
- `labels` support filtering: `uv run ado get spaces --label project=my_project`
- `--query` supports path-based filtering across any field:
  `uv run ado get spaces --query "metadata.name=my_space"`

## Dynamic Reference Resolution

Resource YAMLs often reference other resources by ID. Leave these as
placeholders and resolve them at creation time — do not hard-code IDs.

### --use-latest

Fills in the most recently created resource ID of the given type.

```bash
# Create space, then operation that references it — no manual ID copy
uv run ado create space -f space.yaml
uv run ado create operation -f operation.yaml --use-latest space
```

> **Context scoping**: `--use-latest` resolves relative to the _execution
> context_, not the project. On a laptop, it finds the last resource created
> locally. When launched with `--remote`, it finds the last resource created on
> the remote cluster. Keep this in mind when reusing resources across
> environments.

### --with

Creates a dependency inline and injects its ID automatically.

```bash
# Create space + actuatorconfiguration + operation in one command
# Note: You cannot use --with store=store.yaml or store_id here
# The space must use default store or have a valid store_id in the YAML
uv run ado create operation -f operation.yaml \
  --with space=space.yaml \
  --with actuatorconfiguration=config.yaml
```

Note: Can also specify resources ids to --with

```bash
uv run ado create operation -f operation.yaml \
  --with space=space-abcd-1234
```

### --set

Overrides individual fields in the YAML at creation time without editing the
file. Useful for environment-specific values or quick one-off changes.

```bash
# Override the sample store identifier
uv run ado create space -f space.yaml --set sampleStoreIdentifier=my_store

# Override a nested field using dot notation
uv run ado create operation -f operation.yaml --set parameters.budget=100
```

`--set` takes `path=JSON_document` pairs and can be used multiple times.

## Validation

Always validate before creating:

```bash
uv run ado create RESOURCETYPE -f FILE --dry-run
```

`--dry-run` validates the YAML without creating the resource.

## Templates

Use `ado template` to generate a starter YAML for any resource type:

```bash
# Generic discoveryspace template
uv run ado template discoveryspace

# Space template pre-filled for a specific experiment
uv run ado template discoveryspace --from-experiment my_experiment

# Operation template for a specific operator
uv run ado template operation --operator-name ray_tune

# ActuatorConfiguration template for a specific actuator
uv run ado template actuatorconfiguration --actuator-identifier my_actuator
```

## Resource-Specific Guidance

### DiscoverySpace

**Before creating** (ado create space), check if a matching space already
exists:

```bash
# Match by space config (entity space + experiments)
uv run ado get spaces --matching-space space.yaml

# Match by space ID
uv run ado get spaces --matching-space-id space-abc123

# Filter by label
uv run ado get spaces --label project=my_project --details
```

Reuse an existing space rather than creating a new one — it means the new
operation benefits from measurements already collected.

**Constitutive properties vs. parameterization**:

- Declare a property as a constitutive property domain in the entity space when
  you want to **explore a range of values** for that property.
- Use **experiment parameterization** when you want to change the default value
  of an optional experiment property but keep it fixed across all entities. Do
  not add a single-valued domain to the entity space just to override a default.

```yaml
# Correct: parameterization overrides experiment default, keeps it out of space
experiments:
  - actuatorIdentifier: trainer
    experimentIdentifier: train_model
    parameterization:
      - property:
          identifier: optimizer
        value: adam # overrides default "sgd"

# Incorrect: single-valued domain should be parameterization instead
entitySpace:
  - identifier: optimizer
    propertyDomain:
      variableType: DISCRETE_VARIABLE_TYPE
      values: [adam] # single value — use parameterization instead
```

**Creating a space with a fresh samplestore**:

```bash
uv run ado create space -f space.yaml --new-sample-store
```

### ActuatorConfiguration

**Before creating**, check if a compatible configuration already exists:

```bash
uv run ado get actuatorconfigurations --details
uv run ado get actuatorconfigurations --label actuator=my_actuator
```

Reuse an existing actuator configuration when appropriate rather than creating
duplicates.

### SampleStore

You rarely need to create a samplestore explicitly. Every project comes with a
`default` samplestore that is suitable for most use cases.

Create a new samplestore only when you explicitly want a clean slate with no
shared measurement history.

```bash
uv run ado create samplestore -f samplestore.yaml
```

## Related Resources

- [using-ado-cli](../using-ado-cli/SKILL.md) — CLI command syntax and shortcuts
- [formulate-discovery-problem](../formulate-discovery-problem/SKILL.md) — full
  problem formulation workflow
- [AGENTS.md](../../../AGENTS.md) — YAML testing and linting guidance
