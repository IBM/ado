# Reference: ado Problem Formulation

Detailed reference information for formulating problems in ado.

## DiscoverySpace Configuration Schema

### Core Fields

**sampleStoreIdentifier** (required):

- String identifier of the sample store
- Default: `"default"`
- Use existing store or create new one

**entitySpace** (optional):

- List of `ConstitutiveProperty` objects
- Defines dimensions of the space
- Required if space needs to generate new entities

**experiments** (optional):

- `MeasurementSpaceConfiguration` or list of `ExperimentReference`
- Defines what experiments to run
- Can be empty if only using existing entities

**metadata** (optional):

- `ConfigurationMetadata` object
- Name, description, labels, custom fields

### Experiment Reference Format

See
[reference-experiment-format.yaml](yaml-examples/reference-experiment-format.yaml).

## Entity Space Property Schema

### ConstitutiveProperty Structure

See
[reference-property-domain.yaml](yaml-examples/reference-property-domain.yaml).

### Variable Types

**DISCRETE_VARIABLE_TYPE:**

- Numeric values from a finite set
- Use `values` for explicit list: `[1, 2, 4, 8]`
- Use `domainRange` + `interval` for ranges: `domainRange: [1, 10], interval: 1`

**CONTINUOUS_VARIABLE_TYPE:**

- Real-valued range
- Requires `domainRange: [min, max]`
- Cannot enumerate all values

**CATEGORICAL_VARIABLE_TYPE:**

- String or other categorical values
- Requires `values: ["option1", "option2", ...]`

**BINARY_VARIABLE_TYPE:**

- Boolean or two-value categorical
- No domain specification needed

## Operation Configuration Schema

### Core Structure

See
[reference-operation-structure.yaml](yaml-examples/reference-operation-structure.yaml).

### Operation Types

- `search` - Exploration/optimization operations (e.g., random_walk, ray_tune)
- `modify` - Space modification operations
- `characterize` - Analysis/characterization operations
- `compare` - Comparison operations
- `fuse` - Space fusion operations
- `learn` - Learning operations

## Experiment Properties

### Required Properties

From `experiment.requiredProperties`:

- **ConstitutiveProperty**: Must be in entity space
- **ObservedProperty**: Must be measured by another experiment in the space

### Optional Properties

From `experiment.optionalProperties`:

- Have default values in `experiment.defaultParameterization`
- Can be:
  - Left as defaults (recommended unless user needs to vary them)
  - Added to entity space (if user wants to explore them)
  - Custom parameterized (if user needs specific non-default values)

### Target Properties

From `experiment.targetProperties`:

- What the experiment measures
- Become `ObservedProperty` instances after measurement
- Can be used as inputs to dependent experiments

## Domain Compatibility Rules

### Subdomain Relationship

Entity space property domain must be a **subdomain** of experiment's required
property domain:

- For discrete: All entity space values must be in experiment's domain values
- For continuous: Entity space range must be within experiment's range
- For categorical: Entity space values must be subset of experiment's values

### Validation

ado validates:

1. All required constitutive properties are in entity space
2. Entity space domains are compatible (subdomain check)
3. No redundant dimensions (all entity space properties required by at least one
   experiment)
4. Optional properties in entity space don't conflict with parameterization

## Common Patterns

### Pattern 1: Single Experiment, Simple Space

See
[reference-pattern1-simple-space.yaml](yaml-examples/reference-pattern1-simple-space.yaml).

### Pattern 2: Multiple Experiments with Dependencies

See
[reference-pattern2-multiple-experiments.yaml](yaml-examples/reference-pattern2-multiple-experiments.yaml).

### Pattern 3: Parameterized Experiment

See
[reference-pattern3-parameterized.yaml](yaml-examples/reference-pattern3-parameterized.yaml).

### Pattern 4: Optional Property in Entity Space

See
[reference-pattern4-optional-property.yaml](yaml-examples/reference-pattern4-optional-property.yaml).

## Validation Commands Reference

**DiscoverySpace:**

```bash
uv run ado create space -f FILE.yaml --dry-run
```

**Operation:**

```bash
uv run ado create operation -f FILE.yaml --dry-run
```

**With schema details:**

```bash
uv run ado template space --include-schema
uv run ado template operation --operator-name NAME --include-schema
```

## Template Commands Reference

**Space from experiment:**

```bash
uv run ado template space --from-experiment EXPERIMENT -o space.yaml
```

**Operation template:**

```bash
uv run ado template operation --operator-name OPERATOR_NAME -o operation.yaml
```

**List experiments:**

```bash
uv run ado get experiments --details
```

**Describe experiment:**

```bash
uv run ado describe experiment EXPERIMENT
```

**List operators:**

```bash
uv run ado get operators
```
