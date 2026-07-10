<!-- markdownlint-disable-next-line first-line-h1 -->
## Entities

Entities represent the things you want to measure — for example, a molecule,
a fine-tuning deployment configuration, or a robotic experiment setup.

Every Entity is described by a set of
[**constitutive properties**](properties-and-domains.md#property-types), and
corresponding values, that uniquely identify it. For a fine-tuning deployment
configuration these
might be GPU model, number of GPUs, and batch size. For a molecule they might
be a SMILES string.

Once an Experiment has been run on an Entity, it also gains
[**observed properties**](actuators.md#target-and-observed-properties) — the
measured outputs produced by that Experiment.

### Example

Here is an Entity representing a fine-tuning deployment configuration:

<!-- markdownlint-disable line-length -->
```terminaloutput
Identifier: dataset_id.news-tokens-16384plus-entries-4096-model_name.llama3-8b-number_gpus.4.0-model_max_length.2048.0-torch_dtype.bfloat16-batch_size.16.0-gpu_model.NVIDIA-A100-80GB-PCIe

Constitutive properties:
                 name                               value
  0        dataset_id  news-tokens-16384plus-entries-4096
  1        model_name                           llama3-8b
  2       number_gpus                                 4.0
  3  model_max_length                              2048.0
  4       torch_dtype                            bfloat16
  5        batch_size                                16.0
  6         gpu_model               NVIDIA-A100-80GB-PCIe
```
<!-- markdownlint-enable line-length -->

The identifier is derived from the constitutive property values — two Entities
with the same values are the same Entity. Once Experiments have been run on
this Entity, observed properties (measured values such as
`train_tokens_per_second`) will also appear. See
[Target and Observed Properties](actuators.md#target-and-observed-properties)
for more.

>[!IMPORTANT] Measuring Entities with Experiments
>
> In order for an [Experiment](actuators.md#experiments) to measure an Entity,
> the Entity's constitutive property values must fall within the input domains
> declared by the Experiment.

## Entity Spaces

An individual Entity is a single point. An **Entity Space** defines the full
set of Entities you want to explore — all the points you could potentially
measure.

An Entity Space is a set of constitutive properties, each with a **Property
Domain** that constrains the values it can take. Each property is a dimension
of the space, and every combination of values across all dimensions is an
Entity in the space. That is,
the Entity Space is the cartesian product of the dimensions.

### Example: Fine-tuning Deployment Configuration

<!-- markdownlint-disable line-length -->
```commandline
Number entities: 80
  Categorical properties:
              name                                values
    0   dataset_id  [news-tokens-16384plus-entries-4096]
    1   model_name                [granite-8b-code-base]
    2  torch_dtype                            [bfloat16]
    3    gpu_model               [NVIDIA-A100-80GB-PCIe]

  Discrete properties:
                   name        range interval                         values
    0       number_gpus       [2, 5]     None                         [2, 4]
    1  model_max_length  [512, 8193]     None  [512, 1024, 2048, 4096, 8192]
    2        batch_size     [1, 129]     None  [1, 2, 4, 8, 16, 32, 64, 128]
```
<!-- markdownlint-enable line-length -->

This space has 7 dimensions: 4 categorical (each fixed to a single value) and
3 discrete. The total number of Entities is the product of the number of values
in each dimension:

```text
1 × 1 × 1 × 1 × 2 × 5 × 8 = 80 Entities
```

Each Property Domain constrains one dimension. The categorical properties list
their allowed values explicitly; the discrete properties specify a range and a
set of values within it. For the full list of domain types see
[Properties and Domains](properties-and-domains.md).
