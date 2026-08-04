# Examples: ado Problem Formulation

Concrete examples of formulating problems for ado execution.

## Example 1: Simple Hyperparameter Optimization

**User Query:** "I want to optimize learning rate and batch size for model
training"

### Step 1: Choose Experiments

```bash
uv run ado get experiments --details
# Find: TrainerActuator.train_model
uv run ado describe experiment TrainerActuator.train_model
```

**Experiment details:**

- Required: `learning_rate`, `batch_size`
- Optional: `optimizer` (default: "sgd"), `epochs` (default: 10)
- Targets: `accuracy`, `loss`

### Step 2: Create DiscoverySpace YAML

See [example1-space.yaml](yaml-examples/example1-space.yaml).

**Explanation:**

- Added `learning_rate` and `batch_size` to entity space (required properties)
- Left `optimizer` and `epochs` as defaults (not needed for user's query)
- Refined domains based on typical hyperparameter ranges

### Step 3: Validate

```bash
uv run ado create space -f space.yaml --dry-run
```

### Step 4: Create Operation YAML

See [example1-operation.yaml](yaml-examples/example1-operation.yaml).

### Step 5: Validate Operation

```bash
uv run ado create operation -f operation.yaml --dry-run
```

## Example 2: Multi-Experiment Pipeline

**User Query:** "I want to train models and then evaluate them on a test set"

### Example 2 Step 1: Choose Experiments

```bash
uv run ado describe experiment TrainerActuator.train_model
# Required: learning_rate, batch_size
# Targets: accuracy, loss

uv run ado describe experiment EvaluatorActuator.evaluate_model
# Required: accuracy (ObservedProperty from train_model)
# Targets: test_accuracy, test_loss
```

### Example 2 Step 2: Create DiscoverySpace YAML

See [example2-space.yaml](yaml-examples/example2-space.yaml).

**Explanation:**

- Added required constitutive properties (`learning_rate`, `batch_size`)
- Included both experiments - `evaluate_model` depends on `train_model`'s output
- Dependency automatically handled (evaluate_model requires accuracy from
  train_model)

### Example 2 Step 3: Validate

```bash
uv run ado create space -f space.yaml --dry-run
```

## Example 3: Exploring Optional Properties

**User Query:** "I want to compare different optimizers (adam, sgd, rmsprop)
with different learning rates"

### Example 3 Step 1: Choose Experiments

```bash
uv run ado describe experiment TrainerActuator.train_model
# Required: learning_rate, batch_size
# Optional: optimizer (default: "sgd")
```

### Example 3 Step 2: Create DiscoverySpace YAML

See [example3-space.yaml](yaml-examples/example3-space.yaml).

**Explanation:**

- Added `optimizer` to entity space (necessary to answer user's query about
  comparing optimizers)
- Reduced `batch_size` values to keep space manageable
- All three optimizers will be explored

### Example 3 Step 3: Validate

```bash
uv run ado create space -f space.yaml --dry-run
```

## Example 4: Custom Parameterization

**User Query:** "I want to train with adam optimizer specifically, but explore
learning rates"

### Example 4 Step 1: Choose Experiments

```bash
uv run ado describe experiment TrainerActuator.train_model
# Optional: optimizer (default: "sgd")
```

### Example 4 Step 2: Create DiscoverySpace YAML

See [example4-space.yaml](yaml-examples/example4-space.yaml).

**Explanation:**

- Used parameterization to set `optimizer` to "adam" (not exploring it, just
  fixing it)
- Only `learning_rate` and `batch_size` in entity space (required properties)
- Custom parameterization overrides default "sgd"

### Example 4 Step 3: Validate

```bash
uv run ado create space -f space.yaml --dry-run
```

## Example 5: Domain Refinement

**User Query:** "I want to fine-tune learning rate around 0.01"

### Example 5 Step 1: Choose Experiments

```bash
uv run ado describe experiment TrainerActuator.train_model
# Required: learning_rate (domain: [0.0001, 1.0])
```

### Example 5 Step 2: Create DiscoverySpace YAML

See [example5-space.yaml](yaml-examples/example5-space.yaml).

**Explanation:**

- Refined `learning_rate` domain from `[0.0001, 1.0]` to `[0.005, 0.02]`
  (narrower range around 0.01)
- This reduces the search space and focuses on the region of interest
- Domain is still compatible (subdomain of experiment's domain)

### Example 5 Step 3: Validate

```bash
uv run ado create space -f space.yaml --dry-run
```

## Example 6: Operation with Actuator Configuration

**User Query:** "Run random walk with GPU configuration"

### Step 1: Create DiscoverySpace (from previous examples)

### Step 2: Create Actuator Configuration

See
[example6-actuator-config.yaml](yaml-examples/example6-actuator-config.yaml).

```bash
uv run ado create actuatorconfiguration -f gpu_config.yaml
# Returns: actuatorconfiguration-xyz789
```

### Step 3: Create Operation YAML

See [example6-operation.yaml](yaml-examples/example6-operation.yaml).

**Explanation:**

- Referenced actuator configuration created in Step 2
- Operation will use GPU settings when running experiments

### Step 4: Validate

```bash
uv run ado create operation -f operation.yaml --dry-run
```

## Common Validation Errors and Fixes

### Error: "required property not in entity space"

**Error message:**

```text
ValueError: Identified a measurement space constitutive property not in
entity space: batch_size
```

**Fix:** Add missing property to entity space:

See [error-fix-batch-size.yaml](yaml-examples/error-fix-batch-size.yaml).

### Error: "domain incompatible"

**Error message:**

```text
ValueError: Identified an entity space dimension not compatible with the
measurement space requirements.
```

**Fix:** Ensure entity space domain is subdomain of experiment's domain:

```text
# Experiment requires: [0.0001, 1.0]
# Entity space: [0.001, 0.1] ✓ (subdomain)
# Entity space: [0.5, 2.0] ✗ (not subdomain)
```

### Error: "redundant dimension"

**Error message:**

```text
ValueError: Identified an entity space dimension that is not required for any experiment
```

**Fix:** Remove properties not required by any experiment, or add an experiment
that requires it.
