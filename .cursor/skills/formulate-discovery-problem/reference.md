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

### Domain range bounds (numeric)

`domainRange: [lower, upper]` is **half-open**: **lower inclusive, upper
exclusive** — valid values satisfy `lower <= value < upper`. The upper endpoint
itself is **not** in the domain.

> **The content below is sufficient for writing and validating YAML.**
> Only go deeper if you encounter an edge case not covered here:
> read `docs/concepts/properties-and-domains.md` if the source repo is available,
> otherwise see <https://ibm.github.io/ado/latest/concepts/properties-and-domains/>.

### Variable Types

**DISCRETE_VARIABLE_TYPE** — a finite set of numeric values.

```yaml
# explicit list
domain:
  values: [1, 2, 4, 8, 16, 32, 64, 128]

# range with interval (lower inclusive, upper exclusive)
domain:
  domainRange: [1, 129]
  interval: 1
```

**CONTINUOUS_VARIABLE_TYPE** — a continuous real-valued range.

```yaml
# bounded range (upper bound exclusive)
domain:
  domainRange: [0.0, 1.0]

# unbounded (any real number)
domain:
  variableType: CONTINUOUS_VARIABLE_TYPE
```

**CATEGORICAL_VARIABLE_TYPE** — a finite, named set of values (strings or
numbers).

```yaml
domain:
  values: [granite-3-8b, llama3-8b, mistral-7b-v0.1]
```

**BINARY_VARIABLE_TYPE** — exactly two values: `true` and `false`. No
`values` or `domainRange` needed.

```yaml
domain:
  variableType: BINARY_VARIABLE_TYPE
```

**OPEN_CATEGORICAL_VARIABLE_TYPE** — categorical values where the complete
set is not known in advance (e.g. molecule identifiers, AI model names). Must
be declared explicitly; an optional `values` field can seed known categories.

```yaml
domain:
  variableType: OPEN_CATEGORICAL_VARIABLE_TYPE

# with seed values
domain:
  variableType: OPEN_CATEGORICAL_VARIABLE_TYPE
  values: [pigeon-10.mps.gz]
```

### Auto-inference of variable type

When `variableType` is omitted, ado infers the type from other fields:

| Fields present | Inferred type |
| --- | --- |
| `values` with all numeric entries | `DISCRETE_VARIABLE_TYPE` |
| `values` with any non-numeric entry | `CATEGORICAL_VARIABLE_TYPE` |
| `domainRange` only (no `interval`) | `CONTINUOUS_VARIABLE_TYPE` |
| `domainRange` + `interval` | `DISCRETE_VARIABLE_TYPE` |
| `interval` only (no `domainRange`) | `DISCRETE_VARIABLE_TYPE` |

`BINARY_VARIABLE_TYPE` and `OPEN_CATEGORICAL_VARIABLE_TYPE` cannot be
inferred and must always be declared explicitly.

### probabilityFunction field

Each domain can optionally specify how values are sampled. Default is
**uniform** — every value equally likely.

```yaml
domain:
  values: [1, 2, 4, 8, 16]
  probabilityFunction:
    identifier: uniform
```

A **normal** distribution is available for continuous and discrete domains:

```yaml
domain:
  domainRange: [0.0, 1.0]
  probabilityFunction:
    identifier: normal
    parameters:
      mean: 0.5
      std: 0.1
```

## Operation Configuration Schema

### Core Structure

See
[reference-operation-structure.yaml](yaml-examples/reference-operation-structure.yaml).

### Operation Types

- `explore` - Exploration/optimization operations (e.g., random_walk, ray_tune)
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

### Compatible Subdomain Types

Not every combination of domain types is valid — the subdomain type must be
compatible with the parent type:

| Parent domain | Compatible sub-domain types |
| --- | --- |
| `CONTINUOUS` | `CONTINUOUS`, `DISCRETE` (finite), `BINARY` |
| `DISCRETE` | `DISCRETE`, `BINARY` |
| `CATEGORICAL` | `CATEGORICAL`, `DISCRETE` (finite), `BINARY` |
| `BINARY` | `BINARY`, `DISCRETE` (≤2 values) |
| `OPEN_CATEGORICAL` | `OPEN_CATEGORICAL`, `CATEGORICAL`, `DISCRETE` (finite), `BINARY` |

**Example** — narrowing an experiment's domains to a focused entity space:

```yaml
# Experiment input domains (maximum possible extent)
- identifier: model_name
  propertyDomain:
    values: [granite-3-8b, llama3-8b, mistral-7b-v0.1, granite-34b-code-base]
- identifier: batch_size
  propertyDomain:
    domainRange: [1, 4097]
    interval: 1
- identifier: temperature
  propertyDomain:
    domainRange: [0.0, 100.0]

# Valid entity space subdomains
- identifier: model_name
  propertyDomain:
    values: [granite-3-8b, llama3-8b]   # CATEGORICAL ⊆ CATEGORICAL ✓
- identifier: batch_size
  propertyDomain:
    values: [1, 2, 4, 8, 16]            # DISCRETE ⊆ DISCRETE ✓
- identifier: temperature
  propertyDomain:
    domainRange: [20.0, 40.0]           # CONTINUOUS ⊆ CONTINUOUS ✓
```

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
uv run ado template space --from-experiment EXPERIMENT --output-file space.yaml
```

**Operation template:**

```bash
uv run ado template operation --operator-name OPERATOR_NAME --output-file operation.yaml
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
