<!-- markdownlint-disable-next-line first-line-h1 -->
## Experiments

To find the values of certain properties of Entities we need to perform
measurements on them. We use the term "experiment" to describe a particular type
of measurement. This is also referred to as an "experiment protocol".

An experiment will define its inputs - the set of constitutive and observed
properties it requires entities to have. It will also define the properties it
measures.

You can list them with `ado get experiments --details`. The output will be
similar to:

<!-- markdownlint-disable line-length -->
```terminaloutput
┌────────────────────┬─────────────────────────────────────┬─────────────────────────────────────────────────────────┐
│ ACTUATOR ID        │ EXPERIMENT ID                       │ DESCRIPTION                                             │
├────────────────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ SFTTrainer         │ finetune_full_benchmark-v1.0.0      │ Measures the performance of full-finetuning a model for │
│                    │                                     │ a given (GPU model, number GPUS, batch_size,            │
│                    │                                     │ model_max_length, number nodes) combination.            │
│ SFTTrainer         │ finetune_full_stability-v1.0.0      │ Performs 5 full finetune runs of 5 steps each on a      │
│                    │                                     │ model and reports the fraction of those that resulted   │
│                    │                                     │ in GPU OOM, Other error, or No Error for a given (GPU   │
│                    │                                     │ model, number GPUS, batch_size, model_max_length)       │
│                    │                                     │ combination.                                            │
│ SFTTrainer         │ finetune_gptq-lora_benchmark-v1.0.0 │ Measures the performance of GPTQ-LORA tuning a model    │
│                    │                                     │ for a given (GPU model, number GPUS, batch_size,        │
│                    │                                     │ model_max_length, number nodes) combination.            │
│ SFTTrainer         │ finetune_lora_benchmark-v1.0.0      │ Measures the performance of LORA tuning a model for a   │
│                    │                                     │ given (GPU model, number GPUS, batch_size,              │
│                    │                                     │ model_max_length, number nodes) combination.            │
│ SFTTrainer         │ finetune_pt_benchmark-v1.0.0        │ Measures the performance of prompt-tuning a model for a │
│                    │                                     │ given (GPU model, number GPUS, batch_size,              │
│                    │                                     │ model_max_length, number nodes) combination.            │
│ custom_experiments │ acid_test                           │                                                         │
│ custom_experiments │ avoid_oom_recommender               │ An AutoConf recommender that suggests the minimum       │
│                    │                                     │ number of gpus per worker and number of workers         │
│                    │                                     │ necessary to execute a Tuning job whilekeeping the per  │
│                    │                                     │ GPU batch size constant                                 │
│ custom_experiments │ calculate_density                   │                                                         │
│ custom_experiments │ min_gpu_recommender                 │ An AutoConf plugin that suggests the minimum number of  │
│                    │                                     │ gpus per worker and number of workers necessary to      │
│                    │                                     │ execute a Tuning job                                    │
│ custom_experiments │ ml-multicloud-cost-v1.0             │                                                         │
│ custom_experiments │ nevergrad_opt_3d_test_func          │                                                         │
│ mock               │ test-experiment                     │                                                         │
│ mock               │ test-experiment-two                 │                                                         │
│ robotic_lab        │ peptide_mineralization              │ Measures adsorption of peptide lanthanide combinations  │
└────────────────────┴─────────────────────────────────────┴─────────────────────────────────────────────────────────┘
```
<!-- markdownlint-enable line-length -->

## Actuators

Experiments are provided by Actuators. An Actuator usually provides sets of
experiments that work on the same types of entities i.e. have the same or
similar input requirements. As such Actuators usually are related to a
particular domain e.g., computational chemistry, foundation model inference,
robotic biology lab.

`ado get actuators --details` lists the available actuators, the number of
experiments they provide, a description and their version. Below is an example
of the output:

<!-- markdownlint-disable line-length -->

```commandline
┌────────────────────┬─────────────┬─────────────────────────────────────────────────────┬───────────────────────────┐
│ ACTUATOR ID        │ EXPERIMENTS │ DESCRIPTION                                         │ VERSION                   │
├────────────────────┼─────────────┼─────────────────────────────────────────────────────┼───────────────────────────┤
│ SFTTrainer         │ 5           │ An actuator for benchmarking fine-tuning of         │ 1.5.1.dev13+ga1833142b    │
│                    │             │ foundation models                                   │                           │
│ custom_experiments │ 6           │ Actuator for applying user supplied custom          │ 1.5.1.dev8+531c6444.dirty │
│                    │             │ experiments                                         │                           │
│ mock               │ 2           │ A actuator class for testing                        │ 1.5.1.dev8+531c6444.dirty │
│ replay             │ 0           │ Special actuator for handling externally defined    │ 1.5.1.dev8+531c6444.dirty │
│                    │             │ experiments (experiments we don't have code for)    │                           │
│ robotic_lab        │ 1           │ A template for creating an actuator                 │ 1.5.1.dev13+ga1833142b    │
└────────────────────┴─────────────┴─────────────────────────────────────────────────────┴───────────────────────────┘
```

<!-- markdownlint-enable line-length -->

A primary way to extend `ado` is by developing new Actuators providing the
ability to do experiments on entities in a new domain.

### Example: Experiment from the SFTTrainer actuator

Here is an example (truncated) description of an experiment from the SFTTrainer
actuator.

<!-- markdownlint-disable line-length -->

```commandline
Identifier: SFTTrainer.finetune_pt_benchmark-v1.0.0
Description: Measures the performance of prompt-tuning a model for a given (GPU model, number GPUS, batch_size,
model_max_length, number nodes) combination.


Required Inputs:

   Constitutive Properties:
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: model_name
     Description: The huggingface name or path to the model

     Domain:

        Type: CATEGORICAL_VARIABLE_TYPE
        Values: [
            'allam-1-13b',
            'granite-13b-v2',
            'granite-20b-v2',
            'granite-3-8b',
            'granite-3.0-1b-a400m-base',
            'granite-3.1-2b',
            'granite-3.1-3b-a800m-instruct',
            'granite-3.1-8b-instruct',
            'granite-3.3-8b',
            'granite-34b-code-base',
            'granite-3b-1.5',
            'granite-3b-code-base-128k',
            'granite-4.0-1b',
            'granite-4.0-350m',
            'granite-4.0-h-1b',
            'granite-4.0-h-micro',
            'granite-4.0-h-small',
            'granite-4.0-h-tiny',
            'granite-4.0-micro',
            'granite-7b-base',
            'granite-8b-code-base',
            'granite-8b-code-base-128k',
            'granite-8b-code-instruct',
            'granite-8b-japanese',
            'granite-vision-3.2-2b',
            'hf-tiny-model-private/tiny-random-BloomForCausalLM',
            'llama-13b',
            'llama-7b',
            'llama2-70b',
            'llama3-70b',
            'llama3-8b',
            'llama3.1-405b',
            'llama3.1-70b',
            'llama3.1-8b',
            'llama3.2-1b',
            'llama3.2-3b',
            'llava-v1.6-mistral-7b',
            'mistral-123b-v2',
            'mistral-7b-v0.1',
            'mixtral-8x7b-instruct-v0.1',
            'smollm2-135m'
        ]

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: model_max_length
     Description: The maximum context size. Dataset entries with more tokens they are truncated. Entries with
     fewer are padded

     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Interval: 1
        Range: [1, 131073]

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: batch_size
     Description: The total batch size to use

     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Interval: 1
        Range: [1, 4097]

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: number_gpus
     Description: The total number of GPUs to use

     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Interval: 1
        Range: [0, 33]

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

Optional Inputs and Default Values:

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: max_steps
     Description: The number of optimization steps to perform. Set to -1 to respect num_train_epochs instead

     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Interval: 1
        Range: [-1, 10001]

     Default value: -1
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

Outputs:
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   finetune_pt_benchmark-v1.0.0-is_valid
   finetune_pt_benchmark-v1.0.0-dataset_tokens_per_second_per_gpu
   finetune_pt_benchmark-v1.0.0-train_runtime
   finetune_pt_benchmark-v1.0.0-dataset_tokens_per_second
   finetune_pt_benchmark-v1.0.0-train_samples_per_second
   finetune_pt_benchmark-v1.0.0-train_steps_per_second
   finetune_pt_benchmark-v1.0.0-train_tokens_per_second
   finetune_pt_benchmark-v1.0.0-train_tokens_per_gpu_per_second
   finetune_pt_benchmark-v1.0.0-cpu_compute_utilization
   finetune_pt_benchmark-v1.0.0-cpu_memory_utilization
   finetune_pt_benchmark-v1.0.0-gpu_compute_utilization_min
   finetune_pt_benchmark-v1.0.0-gpu_compute_utilization_avg
   finetune_pt_benchmark-v1.0.0-gpu_compute_utilization_max
   finetune_pt_benchmark-v1.0.0-gpu_memory_utilization_min
   finetune_pt_benchmark-v1.0.0-gpu_memory_utilization_avg
   finetune_pt_benchmark-v1.0.0-gpu_memory_utilization_max
   finetune_pt_benchmark-v1.0.0-gpu_memory_utilization_peak
   finetune_pt_benchmark-v1.0.0-gpu_power_watts_min
   finetune_pt_benchmark-v1.0.0-gpu_power_watts_avg
   finetune_pt_benchmark-v1.0.0-gpu_power_watts_max
   finetune_pt_benchmark-v1.0.0-gpu_power_percent_min
   finetune_pt_benchmark-v1.0.0-gpu_power_percent_avg
   finetune_pt_benchmark-v1.0.0-gpu_power_percent_max
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

<!-- markdownlint-enable line-length -->

The SFTTrainer actuator provides experiments which measure the performance of
different fine-tuning techniques on a foundation model fine-tuning deployment
configuration. Therefore, the entities it takes as input represent fine-tuning
deployment configuration.

## Experiment Inputs

Experiments define their inputs they require along with valid values for those
inputs.

### Required Inputs

Experiments can define required inputs. There are properties an Entity must have
values for, for it to be a valid input to the Experiment.

For example for `SFTTrainer.finetune_pt_benchmark-v1.0.0` shown above we can see
it requires an Entity to have 4 constitutive properties defined: `model_name`,
`model_max_length`, `batch_size` and `number_gpus`. Each one has a domain which
defines the allowed values for that property - if an Entity has a value for a
property that is not in the defined domain the experiment cannot run on it.

For example, the `number_gpu` property can only have the values from 0 to 32
(range is exclusive of upper bound)

<!-- markdownlint-disable line-length -->
```commandline
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: number_gpus
     Description: The total number of GPUs to use

     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Interval: 1
        Range: [0, 33]

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
<!-- markdownlint-enable line-length -->

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

<!-- markdownlint-disable line-length -->
```terminaloutput
Identifier: robotic_lab.peptide_mineralization
Description: Measures adsorption of peptide lanthanide combinations


Required Inputs:

   Constitutive Properties:
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: peptide_identifier
     Description: The identifier of the peptide to use

     Domain:

        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['test_peptide', 'test_peptide_new']

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: peptide_concentration
     Description: The concentration of the peptide

     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Values: [0.1, 0.4, 0.6, 0.8]

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: lanthanide_concentration
     Description: The concentration of lanthanide

     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Values: [0.1, 0.4, 0.6, 0.8]

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

Optional Inputs and Default Values:

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: temperature
     Description: The temperature at which to execute the experiment

     Domain:

        Type: CONTINUOUS_VARIABLE_TYPE
        Range: [0, 100]

     Default value: 23
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: replicas
     Description: How many replicas to average the adsorption_timeseries over

     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Interval: 1
        Range: [1, 4]

     Default value: 1
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: robot_identifier
     Description: The identifier of the robot to use to perform the experiment

     Domain:

        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['harry', 'hermione']

     Default value: 'hermione'
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

Outputs:
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   peptide_mineralization-adsorption_timeseries
   peptide_mineralization-adsorption_plateau_value
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
<!-- markdownlint-enable line-length -->

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

## Measurement Space

A measurement space is simply a set of [experiments](actuators.md#experiments).

Since each experiment has a set of observed properties, a measurement space also
defines a set of observed properties.

Since each observed property is an observation of a target property, a
measurement space also defines a set of target properties.
