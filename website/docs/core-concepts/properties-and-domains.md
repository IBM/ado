# Properties and Domains

Properties and Property Domains are what `ado` uses to describe the
inputs and outputs of [Experiments](actuators.md),
and the dimensions of [Entity Spaces](entity-spaces.md).

## Property

A **Property** is a named concept — a string identifier such as:

* _gpu-model_
* _batch-size_
* _node-selection-method_
* _solve-time_

A Property may optionally carry metadata (a description) that explains what the
identifier represents.

Some Properties are also associated with a **Property Domain** that specifies
the set of values the Property is allowed to take:

* gpu-model → one of {A100, H100, MI300}
* batch-size → any integer between 1 and 1024
* node-selection-method → one of {round-robin, random, greedy}
* solve-time → a positive floating‑point number

## Property Types

In `ado` there are three roles a Property can play:

* **Constitutive properties** — the inputs to Experiments, and the dimensions
  of an Entity Space. They describe inherent or assumed characteristics of the
  Entity — the "givens". Constitutive properties usually have a Property Domain.
* **Target properties** — the properties an Experiment _intends_ to measure,
  e.g. `train_tokens_per_second`.
* **Observed properties** — the values actually recorded by a specific
  Experiment. Because many Experiments may target the same property, each
  observed property is namespaced to the Experiment that produced it
  (e.g. `finetune_lora-train_tokens_per_second`). See
  [Target and Observed Properties](actuators.md#target-and-observed-properties).

> [!NOTE]
>
> In ado, usually only constitutive properties have Property Domains.

## Property Domain Types

`ado` supports the following Property Domain types. Each is written under a
`domain:` key in ado YAML.

> [!NOTE]
>
> The different domain types are distinguished by a **Variable Type** field
> (`variableType`). In many cases this can be omitted and `ado` will infer it
> automatically — see [Auto-inference](#auto-inference-of-property-domain-types).

### Categorical

A finite, named set of values. Typically strings, though numeric values are
also allowed.

Used when the property can take one of a fixed list of labels.

```yaml
domain:
  values: [granite-3-8b, llama3-8b, mistral-7b-v0.1]
```

### Discrete

A finite set of numeric values, specified either as an explicit list or as a
range with a step interval. Both forms are equivalent.

Used when the property takes a countable set of numbers.

**Explicit list:**

```yaml
domain:
  values: [1, 2, 4, 8, 16, 32, 64, 128]
```

**Range with interval** (lower inclusive, upper exclusive):

```yaml
domain:
  domainRange: [1, 129]
  interval: 1
```

**Interval only** (unbounded discrete — any multiple of the interval):

```yaml
domain:
  interval: 1
```

### Continuous

A continuous numeric domain. Use for real-valued properties.

**Bounded range** — any real value within the bounds is valid:

```yaml
domain:
  domainRange: [0, 100]
```

**Unbounded** — any real number:

```yaml
domain:
  variableType: CONTINUOUS_VARIABLE_TYPE
```

### Binary

Exactly two values: `true` and `false`.

```yaml
domain:
  variableType: BINARY_VARIABLE_TYPE
```

### Open Categorical

Categorical values where the complete set of categories is not known in advance.
`variableType` must be set explicitly. An optional `values` field can seed a
known subset of categories.

Used for properties where new categories can appear at runtime, for example a
molecule identifier or an AI model name.

```yaml
domain:
  variableType: OPEN_CATEGORICAL_VARIABLE_TYPE
```

## Auto-inference of Property Domain Types

When `variableType` is omitted, `ado` infers it from the other fields:

| Fields present | Inferred type |
| --- | --- |
| `values` with all numeric entries | `DISCRETE_VARIABLE_TYPE` |
| `values` with any non-numeric entry | `CATEGORICAL_VARIABLE_TYPE` |
| `domainRange` only (no `interval`) | `CONTINUOUS_VARIABLE_TYPE` |
| `domainRange` + `interval` | `DISCRETE_VARIABLE_TYPE` |
| `interval` only (no `domainRange`) | `DISCRETE_VARIABLE_TYPE` |

`BINARY_VARIABLE_TYPE` and `OPEN_CATEGORICAL_VARIABLE_TYPE` cannot be inferred
and must always be declared explicitly.

## Probability Functions

Each domain can optionally specify a probability function that controls how
values are sampled. The default is **uniform** — every value in the domain is
equally likely.

```yaml
domain:
  values: [1, 2, 4, 8, 16]
  probabilityFunction:
    identifier: uniform
```

A **normal** distribution is also available for continuous and discrete domains:

```yaml
domain:
  domainRange: [0.0, 1.0]
  probabilityFunction:
    identifier: normal
    parameters:
      mean: 0.5
      std: 0.1
```

When no `probabilityFunction` is specified, uniform sampling is used.

## Property Subdomains

Domain A is a **subdomain** of domain B if every value in A is also a valid
value in B. A subdomain represents a narrowed or more specific version of a
parent domain.

The most common place this matters in `ado` is when defining an
[Entity Space](entity-spaces.md): the domain you assign to each entity space
dimension must be a subdomain of the corresponding experiment input domain.
This ensures that all entities in the space are valid inputs to the experiment.

### Compatible Subdomain Types

Not every combination of domain types is valid — the subdomain type must be
compatible with the parent type:

<!-- markdownlint-disable line-length -->
| Parent domain | Compatible sub-domain types | Notes |
| --- | --- | --- |
| `CONTINUOUS` | `CONTINUOUS`, `DISCRETE` (finite), `BINARY` | Sub-range must lie within the parent range; `BINARY` requires 0 and 1 to be within the range |
| `DISCRETE` | `DISCRETE`, `BINARY` | Sub-values must be a subset of the parent values; `BINARY` only valid if both 0 and 1 appear in the parent |
| `CATEGORICAL` | `CATEGORICAL`, `DISCRETE` (finite), `BINARY` | Sub-values must be a subset of the parent values |
| `BINARY` | `BINARY`, `DISCRETE` (≤2 values) | Values must be a subset of `{0, 1}` / `{false, true}` |
| `OPEN_CATEGORICAL` | `OPEN_CATEGORICAL`, `CATEGORICAL`, `DISCRETE` (finite), `BINARY` | The most permissive categorical parent |
<!-- markdownlint-enable line-length -->

### Example

Suppose an experiment declares the following required input domains:

```yaml
# Experiment input domains (the maximum possible extent)
model_name:
  values: [granite-3-8b, llama3-8b, mistral-7b-v0.1, granite-34b-code-base]

batch_size:
  domainRange: [1, 4097]
  interval: 1

temperature:
  domainRange: [0.0, 100.0]
```

A valid entity space could narrow each of these to a focused subdomain:

```yaml
# Entity space domains (subdomains of the experiment inputs above)
model_name:
  values: [granite-3-8b, llama3-8b]        # CATEGORICAL ⊆ CATEGORICAL ✓

batch_size:
  values: [1, 2, 4, 8, 16]                 # DISCRETE ⊆ DISCRETE ✓

temperature:
  domainRange: [20.0, 40.0]                # CONTINUOUS ⊆ CONTINUOUS ✓
```

The following entity space domains would be **invalid** because they are not
subdomains of the corresponding experiment inputs:

```yaml
batch_size:
  # Values above 4096 are not in the Experiment input domain for batch_size
  domainRange: [4096, 8124]
  interval: 1028

model_name:
  # granite-4-3b is not one of the allowed values
  domainRange: [granite-4-3b]
```
