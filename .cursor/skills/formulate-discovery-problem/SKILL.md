---
name: formulate-discovery-problem
description: |
  Formulates problems for execution with ado by creating discoveryspace and
  operation YAML files. Guides through experiment selection, space creation,
  validation, operation configuration, and parameterization. Use when the user
  wants to create discoveryspace or operation YAML files, configure experiments,
  set up entity spaces, or formulate research, benchmarking or search problems.
---

# Formulating Problems for ado Execution

Create validated **discoveryspace** and **operation** YAML for systematic work
over an entity space: sampling and measuring entities, searching for entities
that meet objectives, or benchmarking configurations (including research and
benchmarking studies). Execution and analysis of results happen after resources
are created — see [conduct-empirical-study](../conduct-empirical-study/SKILL.md)
for the full workflow.

## Tips

- Unless directed otherwise place all YAML and .md files created in a
  subdirectory of examples/ dedicated to the given problem.
- If you want to change the default value of an optional property use experiment
  parameterization, rather than setting a single valued property in the entity
  space
- Before creating a space or actuator configuration, check if one already exists
  — see [resource-yaml-creation](../resource-yaml-creation/SKILL.md)
- Learn [ado CLI command-line construction and testing](../using-ado-cli/)
- For metadata conventions, dynamic references (--use-latest, --with, --set),
  and resource-specific guidance, see
  [resource-yaml-creation](../resource-yaml-creation/SKILL.md)

## Workflow Overview

The process has two main phases:

1. **Create DiscoverySpace YAML** - Define experiments and entity space
2. **Create Operation YAML** - Configure how to explore/analyze the space

Each phase follows a pattern: choose tool for task (experiment/operator) →
create YAML for task → validate YAML → iterate.

## Phase 1: Create DiscoverySpace YAML

### Step 1a: Choose Experiments

**List available experiments:**

```bash
uv run ado get experiments --details
```

**Describe a specific experiment:**

```bash
uv run ado describe experiment $EXPERIMENT_ID
```

**Key information to gather:**

- Required constitutive properties (must be in entity space)
- Optional properties (can use defaults or add to entity space)
- Target properties (what the experiment measures)

#### What to do if no experiment matching task available

1. Learn how to extend ado:
   [plugin-development.mdc](../../rules/plugin-development.mdc)
2. Propose a custom experiment or actuator to user that would provide missing
   functionality
3. Wait for user input

### Step 1b: Create DiscoverySpace YAML

**Generate initial template from experiment:**

```bash
uv run ado template space --from-experiment $EXPERIMENT_ID -o space.yaml
```

**Manual structure:**

See [skill-manual-structure.yaml](yaml-examples/skill-manual-structure.yaml).

### Step 1c: Validate DiscoverySpace YAML

```bash
uv run ado create space -f space.yaml --dry-run
```

### Step 1d: Iterate Until Valid

Fix validation errors and repeat validation until successful.

## Phase 2: Create Operation YAML

### Step 2a: Choose Operator

**List available operators:**

```bash
uv run ado get operators
```

**Get operator template:**

```bash
uv run ado template operation --operator-name $OPERATOR_NAME -o operation.yaml
```

### Step 2b: Decide Parameters

Review the template and configure parameters based on:

- User's query/goals
- Operator documentation
- Example operations in `examples/`

### Step 2c: Create Operation YAML

**Structure:**

See
[skill-operation-structure.yaml](yaml-examples/skill-operation-structure.yaml)
for an example structure.

### Step 2d: Validate Operation YAML

```bash
uv run ado create operation -f operation.yaml --dry-run
```

### Step 2e: Iterate Until Valid

Fix validation errors and repeat validation until successful.

## Critical Rules

### Experiment Selection Rules

1. **Choose experiments first** - Before defining entity space
2. **All required inputs must be in entity space** - Every `requiredProperties`
   (constitutive) from experiments must have a corresponding property in
   `entitySpace`
3. **Optional properties** - Only add to entity space if necessary to answer
   user's query. Explain why.
4. **Default values** - Only change default values of optional properties if
   necessary. Explain why.

### Entity Space Refinement Rules

1. **Refine domains to reduce size** - Narrow property domains based on user's
   query. Explain the refinement.
2. **No redundant dimensions** - All entity space properties should be required
   by at least one experiment (validation will catch this)
3. **Domain compatibility** - Entity space property domains must be compatible
   with experiment requirements (subdomain or equal)

### Property Domain Guidelines

**Discrete (categorical):**

See
[skill-property-domain-discrete-categorical.yaml](yaml-examples/skill-property-domain-discrete-categorical.yaml).

**Discrete (numeric):**

See
[skill-property-domain-discrete-numeric.yaml](yaml-examples/skill-property-domain-discrete-numeric.yaml).

**Continuous:**

See
[skill-property-domain-continuous.yaml](yaml-examples/skill-property-domain-continuous.yaml).

## Validation Checklist

Before finalizing, verify:

- All required experiment properties are in entity space
- Entity space domains are compatible with experiment requirements
- No redundant entity space dimensions
- Optional properties only added if necessary (with explanation)
- Default values only changed if necessary (with explanation)
- Domain refinements explained
- DiscoverySpace YAML validates (`--dry-run`)
- Operation YAML validates (`--dry-run`)
- All ado CLI commands and options are valid (uv run ado [COMMAND] --help)

## Common Issues and Solutions

**Issue:** Validation error "required property not in entity space"

- **Solution:** Add the missing property to `entitySpace` with appropriate
  domain

**Issue:** Validation error "domain incompatible"

- **Solution:** Ensure entity space domain is a subdomain of experiment's
  required domain

**Issue:** Validation error "redundant dimension"

- **Solution:** Remove properties from entity space that aren't required by any
  experiment

**Issue:** Operation validation fails

- **Solution:** Check operator parameters match schema. Use `--include-schema`
  flag with template command.

## Additional Resources

- For detailed schema information, see [reference.md](reference.md)
- For example workflows, see [examples.md](examples.md)
- For Pydantic model details, examine:
  - `orchestrator/schema/experiment.py`
  - `orchestrator/schema/measurementspace.py`
  - `orchestrator/schema/entityspace.py`
  - `orchestrator/core/discoveryspace/config.py`
  - `orchestrator/core/operation/operation.py`

## References

When modifying or creating code while using this skill, follow:

- [AGENTS.md](../../../AGENTS.md)
- [plugin-development.mdc](../../rules/plugin-development.mdc) (if working with
  plugins)
