<!-- markdownlint-disable-next-line first-line-h1 -->
## Entities

Entities represent things that can be measured. Examples are molecules or points
in an application configuration space.

Entities all have a set of constitutive properties which define them. A
molecule's constitutive properties might be a SMILES or INCHI string. The
constitutive properties of a fine-tuning deployment configuration might be GPU
model, number of GPUs and batch size.

An entity will also have observed properties. These are properties measured by
an experiment (or experiment protocol). For example, a molecule might have an
an observed property for its `band-gap` while a fine-tuning deployment
configuration might have an an observed property related to `tokens throughput`.

### Example: FM Fine-tuning Deployment Configuration

Here is an example of an entity that represents a FM fine-tuning deployment

<!-- markdownlint-disable line-length -->
```terminaloutput
Identifier: dataset_id.news-tokens-16384plus-entries-4096-model_name.llama3-8b-number_gpus.4.0-model_max_length.2048.0-torch_dtype.bfloat16-batch_size.16.0-gpu_model.NVIDIA-A100-80GB-PCIe
Generator: explicit_grid_sample_generator

Constitutive properties:
                 name                               value
  0        dataset_id  news-tokens-16384plus-entries-4096
  1        model_name                           llama3-8b
  2       number_gpus                                 4.0
  3  model_max_length                              2048.0
  4       torch_dtype                            bfloat16
  5        batch_size                                16.0
  6         gpu_model               NVIDIA-A100-80GB-PCIe

Observed properties:
                                                   name                                         experiment                    target-property                values
  0   finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...        gpu_compute_utilization_min   [98.14772727272727]
  1   finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...        gpu_compute_utilization_avg   [98.26988636363636]
  2   finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...        gpu_compute_utilization_max   [98.38636363636364]
  3   finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...         gpu_memory_utilization_min  [33.709723284090906]
  4   finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...         gpu_memory_utilization_avg  [33.709723284090906]
  5   finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...         gpu_memory_utilization_max  [33.709723284090906]
  6   finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...        gpu_memory_utilization_peak           [34.065475]
  7   finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...            cpu_compute_utilization   [98.94999999999999]
  8   finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...             cpu_memory_utilization  [6.3182326931818205]
  9   finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...                      train_runtime            [887.5672]
  10  finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...           train_samples_per_second               [4.615]
  11  finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...             train_steps_per_second               [0.072]
  12  finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...            train_tokens_per_second            [9451.236]
  13  finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...    train_tokens_per_gpu_per_second            [2362.809]
  14  finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...                    model_load_time                [-1.0]
  15  finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...          dataset_tokens_per_second   [9451.237044361262]
  16  finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...  dataset_tokens_per_second_per_gpu  [2362.8092610903154]
  17  finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0-...  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-defa...                           is_valid                 [1.0]

Associated experiments:

  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0
```
<!-- markdownlint-enable line-length -->

For more information about the meaning of `observed properties` see
[target & observed properties](actuators.md#target-and-observed-properties)

## Entity Spaces

An Entity Space describes a set of entities. The set could be discrete or
continuous, bounded or unbounded. In `ado` you normally define Entity Spaces and
then sample Entities from them.

### Example: Molecules

This space has a single dimension with type identifier. This is a property whose
values are a potentially very large set of unique-ids generated in some fashion.

```commandline
  Space with non-discrete dimensions. Cannot count entities
  Identifier properties:
         name
    0  smiles
```

### Example: Fine-tuning Deployment Configuration

This space has 7 dimensions, 4 categorical and 3 discrete. Each of the 4
categorical dimensions has only a single value. The discrete dimensions each
have a range of values they can take.

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

### Property Domains

Each property in an entity space can be associated with a domain. The domain is
the range of values the property can take and also the probability of those
values. In the `Fine-tuning Deployment Configuration` example we can see the
domains for each property. The categorical properties have a set of values and
the discrete properties a range and also a set of values.

In the `Molecules` example we see there is no domain, which means any value of
`smiles` is allowed. When there is no domain it also means the Entity Space
alone does not contain sufficient information by itself on how to sample the
entities.

By default, the probability is uniform, every value is equally likely, but it
could also be more complex.
