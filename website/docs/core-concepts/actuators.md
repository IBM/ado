<!-- markdownlint-disable-next-line first-line-h1 -->
## Experiments

To find the values of certain properties of Entities we need to perform
measurements on them. We use the term "experiment" to describe a particular type
of measurement. This is also referred to as an "experiment protocol".

An experiment will define its inputs - the set of constitutive and observed
properties it requires entities to have. It will also define the properties it
measures.

## Actuators

Experiments are provided by Actuators. An Actuator usually provides sets of
experiments that work on the same types of entities i.e. have the same or
similar input requirements. As such Actuators usually are related to a
particular domain e.g., computational chemistry, foundation model inference,
robotic biology lab.

`ado get actuators --details` lists the available actuators and experiments.
Below is a truncated example of the output:

<!-- markdownlint-disable line-length -->
```commandline
                  ACTUATOR ID                 CATALOG ID                                  EXPERIMENT ID  SUPPORTED
0         molecule-embeddings                 Embeddings                   calculate-morgan-fingerprint       True
1          molformer-toxicity         molformer-toxicity                               predict-toxicity       True
2                     mordred         Mordred Descriptor                  mordred-descriptor-calculator       True
3                       st4sd                      ST4SD                      toxicity-prediction-opera       True
4                       st4sd                      ST4SD                   band-gap-pm3-gamess-us:1.0.0       True
5   materials-model-evaluator  materials-model-evaluator                          evaluate_with_clintox       True
6   materials-model-evaluator  materials-model-evaluator                      evaluate_sider_for_target       True
7      caikit-config-explorer     caikit-config-explorer                                     fmaas-perf       True
8      caikit-config-explorer     caikit-config-explorer                   fmaas-perf-composable-gigaio       True
```
<!-- markdownlint-enable line-length -->

A primary way to extend `ado` is by developing new Actuators providing
the ability to do experiments on entities in a new domain.

### Example: Experiment from the SFTTrainer actuator

Here is an example description of an experiment from the SFTTrainer actuator.

<!-- markdownlint-disable line-length -->
```commandline
Identifier: SFTTrainer.finetune-full-fsdp-v1.6.0

Measures the performance of full-fine tuning a model with FSDP+flash-attention for a given (GPU model, number GPUS, batch_size, model_max_length) combination.

Inputs:
  Constitutive Properties:
      dataset_id
      Domain:
        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['news-chars-1024-entries-1024', 'news-chars-1024-entries-256', 'news-chars-1024-entries-4096', 'news-chars-2048-entries-1024', 'news-chars-2048-entries-256', 'news-chars-2048-entries-4096', 'news-chars-512-entries-1024', 'news-chars-512-entries-256', 'news-chars-512-entries-4096', 'news-tokens-128kplus-entries-320', 'news-tokens-128kplus-entries-4096', 'news-tokens-16384plus-entries-4096']


      model_name
      Domain:
        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['granite-13b-v2', 'granite-20b-v2', 'granite-34b-code-base', 'granite-3b-1.5', 'granite-3b-code-base-128k', 'granite-7b-base', 'granite-8b-code-base', 'granite-8b-code-base-128k', 'granite-8b-japanese', 'hf-tiny-model-private/tiny-random-BloomForCausalLM', 'llama-13b', 'llama-7b', 'llama2-70b', 'llama3-70b', 'llama3-8b', 'llama3.1-405b', 'llama3.1-70b', 'llama3.1-8b', 'mistral-7b-v0.1', 'mixtral-8x7b-instruct-v0.1']


      model_max_length
      Domain:
        Type: DISCRETE_VARIABLE_TYPE Interval: 1.0 Range: [1, 131073]

      torch_dtype
      Domain:
        Type: CATEGORICAL_VARIABLE_TYPE Values: ['bfloat16']

      number_gpus
      Domain:
        Type: DISCRETE_VARIABLE_TYPE Interval: 1.0 Range: [2, 9]

      gpu_model
      Domain:
        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['NVIDIA-A100-SXM4-80GB', 'NVIDIA-A100-80GB-PCIe', 'Tesla-T4', 'L40S', 'Tesla-V100-PCIE-16GB']


      batch_size
      Domain:
        Type: DISCRETE_VARIABLE_TYPE Interval: 1.0 Range: [1, 129]


Outputs:
  finetune-full-fsdp-v1.6.0-gpu_compute_utilization_min
  finetune-full-fsdp-v1.6.0-gpu_compute_utilization_avg
  finetune-full-fsdp-v1.6.0-gpu_compute_utilization_max
  finetune-full-fsdp-v1.6.0-gpu_memory_utilization_min
  finetune-full-fsdp-v1.6.0-gpu_memory_utilization_avg
  finetune-full-fsdp-v1.6.0-gpu_memory_utilization_max
  finetune-full-fsdp-v1.6.0-gpu_memory_utilization_peak
  finetune-full-fsdp-v1.6.0-gpu_power_watts_min
  finetune-full-fsdp-v1.6.0-gpu_power_watts_avg
  finetune-full-fsdp-v1.6.0-gpu_power_watts_max
  finetune-full-fsdp-v1.6.0-gpu_power_percent_min
  finetune-full-fsdp-v1.6.0-gpu_power_percent_avg
  finetune-full-fsdp-v1.6.0-gpu_power_percent_max
  finetune-full-fsdp-v1.6.0-cpu_compute_utilization
  finetune-full-fsdp-v1.6.0-cpu_memory_utilization
  finetune-full-fsdp-v1.6.0-train_runtime
  finetune-full-fsdp-v1.6.0-train_samples_per_second
  finetune-full-fsdp-v1.6.0-train_steps_per_second
  finetune-full-fsdp-v1.6.0-train_tokens_per_second
  finetune-full-fsdp-v1.6.0-train_tokens_per_gpu_per_second
  finetune-full-fsdp-v1.6.0-model_load_time
  finetune-full-fsdp-v1.6.0-dataset_tokens_per_second
  finetune-full-fsdp-v1.6.0-dataset_tokens_per_second_per_gpu
  finetune-full-fsdp-v1.6.0-is_valid
```
<!-- markdownlint-enable line-length -->

The SFTTrainer actuator provides experiments which measure the performance of
different fine-tuning techniques on a foundation model fine-tuning deployment
configuration. Therefore, the entities it takes as input represent fine-tuning
deployment configuration.

### Example: Experiment from the ST4SD actuator

Here is an example description of an experiment from the ST4SD actuator.

```commandline
Identifier: st4sd.band-gap-pm3-gamess-us:1.0.0

Required Inputs:
  Constitutive Properties:
      smiles
      Domain:
        Type: UNKNOWN_VARIABLE_TYPE



Outputs:
  band-gap-pm3-gamess-us:1.0.0-band-gap
  band-gap-pm3-gamess-us:1.0.0-homo
  band-gap-pm3-gamess-us:1.0.0-lumo
  band-gap-pm3-gamess-us:1.0.0-electric-moments
  band-gap-pm3-gamess-us:1.0.0-total-energy
```

The ST4SD actuator provides experiments that perform computational measurements
of entities, often molecules. Therefore, the entities it takes as input often
represent molecules.

## Experiment Inputs

Experiments define their inputs they require along with valid values for those
inputs.

### Required Inputs

Experiments can define required inputs. There are properties an Entity must have
values for, for it to be a valid input to the Experiment.

For example for `SFTTrainer.finetune-full-fsdp-v1.6.0` shown above we can see it
requires an Entity to have 7 constitutive properties defined: `dataset_id`,
`model_name`, `model_max_length`, `torch_dtype`, `number_gpus`, `gpu_model`, and
`batch_size`. Each one has a domain which defines the allowed values for that
property - if an Entity has a value for a property that is not in the defined
domain the experiment cannot run on it.

For example, the `number_gpu` property can only have the values 2,3,4,5,6,7 and
8 (range is exclusive of upper bound)

```commandline
      number_gpus
      Domain:
        Type: DISCRETE_VARIABLE_TYPE Interval: 1.0 Range: [2, 9]
```

All the required inputs in the examples above are
[constitutive properties](entity-spaces.md#entities). However, they can also be
observed properties (see next section) i.e. properties measured by other
experiments. If an Experiment, `B` has a required input that is an observed
property it means the experiment measuring that property has to be run on an
Entity before Experiment `B` can be run on it.

### Optional Properties

Experiments can also define optional properties. These are properties an Entity
can have but if they don't the Experiment will give it a default value. In
addition, the default values of optional properties can be overridden to create
**parameterized experiments**. This is described further in the
[`discoveryspace` resource documentation](../resources/discovery-spaces.md).

An example experiment with optional properties is

```terminaloutput
Identifier: robotic_lab.peptide_mineralization

Measures adsorption of peptide lanthanide combinations

Required Inputs:
  Constitutive Properties:
      peptide_identifier
      Domain:
        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['test_peptide', 'test_peptide_new']


      peptide_concentration
      Domain:
        Type: DISCRETE_VARIABLE_TYPE
        Values: [0.1, 0.4, 0.6, 0.8]
        Range: [0.1, 1.8]


      lanthanide_concentration
      Domain:
        Type: DISCRETE_VARIABLE_TYPE
        Values: [0.1, 0.4, 0.6, 0.8]
        Range: [0.1, 1.8]



Optional Inputs and Default Values:
  temperature
  Domain:
    Type: CONTINUOUS_VARIABLE_TYPE Range: [0, 100]

  Default value: 23.0

  replicas
  Domain:
    Type: DISCRETE_VARIABLE_TYPE Interval: 1.0 Range: [1, 4]

  Default value: 1.0

  robot_identifier
  Domain:
    Type: CATEGORICAL_VARIABLE_TYPE Values: ['harry', 'hermione']

  Default value: hermione


Outputs:
  peptide_mineralization-adsorption_timeseries
  peptide_mineralization-adsorption_plateau_value
```

Here you can see three optional properties, `temperature`, `replicas` and
`robot_identifier` that are given default values.

## Target and Observed Properties

Experiments define properties the properties they measure. However, there may be
many experiments that measure the same property in different ways so we need a
way to differentiate them.

The properties the experiment targets measuring are called `target properties`,
and the properties it actually measures `observed properties`. If experiment `A`
has target property `X`, then the observed property is `A-X` i.e. the value of
target property `X` measured by experiment `A`.

Looking at the definitions above for the `st4sd.band-gap-pm3-gamess-us:1.0.0`
experiment we can see a target is `band-gap` and the corresponding observed
property is `band-gap-pm3-gamess-us:1.0.0-band-gap`

## Measurement Space

A measurement space is simply a set of [experiments](actuators.md#experiments).

Since each experiment has a set of observed properties, a measurement space also
defines a set of observed properties.

Since each observed property is an observation of a target property, a
measurement space also defines a set of target properties.
