<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
A core task is sampling and measuring `entities` from a `discoveryspace` and
this is the objective of explore `operators`. In fact, other than copying data
from an external source, `operations` using _explore_ `operators` are the only
way new entities can be sampled and measurements performed.

Because `operations` of the explore operators do more than just modify or
analyse data, but result in measurements being executed and `entities` being
placed in the `samplestore`, it's worth diving into them in more detail.

## Timeseries

Every explore operation samples entities and performs measurements in some
sequence, hence there is an associated timeseries. This timeseries is recorded
for every `explore` operation. To see it via `ado` CLI use:

```commandline
ado show entities operation $OPERATION_IDENTIFIER
```

This will output a table of the entities in the order they were sampled during
the operation.

You can access this information programmatically by modifying the following
snippet:

<!-- markdownlint-disable line-length -->
```python
import yaml
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.metastore.project import ProjectContext

with open("my_context.yaml") as f:
    c = ProjectContext.model_validate(yaml.safe_load(f))

space = DiscoverySpace.from_stored_configuration(project_context=c, space_identifier='space_abc123')
# Get the timeseries of a property `someproperty` measured by `someexperiment from the space
# for operation "operation_abc123". You can also omit the limit_to_properties parameter to retrieve
# all the properties
space.complete_measurement_request_with_results_timeseries(operation_id="operation_abc123",
                                                           limit_to_properties=["someexperiment.someproperty"])
```
<!-- markdownlint-enable line-length -->

Importantly, the same entity can be visited by multiple different `operations`.
Looking at the `entity` will not show which `operation` measured which value.
However, this information is accessible via the timeseries.

## The core explore loop

Each explore operation will perform the following steps in some way:

- Sample 1 or more entities from the `discoveryspace`
- For each experiment in the `measurementspace`:
  - If it has been already executed on this entity AND the `discoveryspace` only
      permits one value per observed-property
    - replay the already measured value
  - Otherwise:
    - call the actuator to perform the measurement
- Wait until all measurements have completed
  - As each completes:
    - add the replayed/newly measured values to the sampling timeseries for this
          operation (if the measurement did not fail)
    - add the entity to the `samplestore` if it's not there
    - update the entity in the `samplestore` with the new measured property
          values (if the measurement did not fail)

### Replayed Measurements

A core goal of `ado` is transparent data-sharing. This is enabled via the
[common context provided by `samplestores`](../resources/sample-stores.md) and
the schema used to store `entities`.

To leverage this data-sharing capability explore operation will, be default, not
re-measure an entity they sample if it already has data for that measurement.
For example, an explore operation samples an entity from a space whose
`measurementspace` includes an experiment called "myexperiment-v1". If it sees
the entity has values for experiment `myexperiment-v1`, it won't execute it
again, instead it replays (aka "memoizes") it.

This means if a different user sampled and measured this entity with this
experiment on a different space we transparently reuse their results, saving
execution time.

Some operators will allow turning this replay behaviour on and off.

If the replay functionality is off, the entity will be re-measured with the
experiment and it will have two values for each observed property of that
experiment. Going back to our example above, if `myexperiment-v1` was executed
again, and it measured properties `prop1` and `prop2`, then the entity will have
two values for `myexperiment-v1.prop1` and `myexperiment-v1.prop2`, one from
each time the experiment was applied to that entity.

What if you switch it on but an entity has multiple measurements of the same
experiment? In this case _each existing measurement is replayed_. In our
example, this would mean if an entity has had `myexperiment-v1` applied twice,
and then is sampled again with replay on, two measurements will be replayed: the
first and the second.

### Failed measurements

If a measurement of an entity fails in a way not expected by the `actuator` i.e.
it could not measure any of its target properties, the `entity` will be added to
the `samplestore` (if it was not present already); the `operation` will proceed;
but this entity will have no measured values for this experiment.

## Exploration operation metadata

When an explore operation finishes, the system (top-level) metadata field of the
associated `operation` resource is updated with the following fields.

<!-- markdownlint-disable line-length -->
```python
entities_submitted: #The number of entities sampled from the space
experiments_requested: #The number of experiments requested - should be (number of experiments in measurement space)*entitiesSampled
```
<!-- markdownlint-enable line-length -->

Example from a completed random walk `operation`:

```yaml
config:
  metadata:
    description:
      Both single and multi GPU runs of GPTQ-LoRA experiments for first group of
      GPTQ-LoRA
    labels:
      group: "1"
      group_type: gptq-lora
      issue: "904"
  operation:
    module:
      moduleClass: RandomWalk
      moduleName: orchestrator.modules.operators.randomwalk
      modulePath: .
      moduleType: operation
    parameters:
      batchSize: 4
      singleMeasurement: false
      numberEntities: all
      samplerConfig:
        mode: sequential
        samplerType: generator
  spaces:
    - space-8f1cfb-91ecfb
created: "2024-10-07T06:46:08.176924Z"
identifier: randomwalk-0.6.4-1be83b
kind: operation
metadata:
  entities_submitted: 160
  experiments_requested: 160
operationType: search
operatorIdentifier: randomwalk-0.6.4
result: null
status: []
version: v1
```

## Viewing the `operation` and `discoveryspace` state as an operation runs

As an `operation` is running new measurements are being performed, entities
added, and some fraction of the requested entities will have been sampled. Some
`ado` commands reflect this changing state while other do not.

Commands that reflect changing state during an `operation`:

- `ado show entities space`
- `ado show entities operation`
- `ado show details space`

Commands that do not reflect changing state during an `operation`:

- `ado get operation $OPERATION_IDENTIFIER`
- `ado show details operation $OPERATION_IDENTIFIER`

The `operation` resource itself will be updated with metadata when the operation
finishes but not while it is running.
`ado show details operation $OPERATION_IDENTIFIER` uses this metadata, so it
will not be correct until the operation is finished.
