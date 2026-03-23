<!-- markdownlint-disable-next-line first-line-h1 -->
A Discovery Space combines an [Entity Space](entity-spaces.md) and a
[Measurement Space](actuators.md#measurement-space). The Entity Space defines
the Entities you want to measure; the Measurement Space defines how they are
measured. Results are stored in a [Sample Store](data-sharing.md).

A Discovery Space is a **view** rather than a container — data is fetched from
the Sample Store on demand. This means multiple Discovery Spaces can share
measurement results transparently, and any measurement made by anyone using the
same Sample Store becomes immediately available.

## Example: Fine-Tuning Deployment Configuration

<!-- markdownlint-disable descriptive-link-text -->
We can combine the
[Entity Space example](entity-spaces.md#example-fine-tuning-deployment-configuration)
with one of the Experiments from the [`SFTTrainer` Actuator](../actuators/sft-trainer.md)
to create the
following Discovery Space:
<!-- markdownlint-enable descriptive-link-text -->

<!-- markdownlint-disable line-length -->
```terminaloutput
Identifier: space-edf5e2-2351e8

Entity Space:

  Number entities: 80
  Categorical properties:
              name                                values
    0   dataset_id  [news-tokens-16384plus-entries-4096]
    1   model_name                           [llama3-8b]
    2  torch_dtype                            [bfloat16]
    3    gpu_model               [NVIDIA-A100-80GB-PCIe]

  Discrete properties:
                   name        range interval                         values
    0       number_gpus       [2, 5]     None                         [2, 4]
    1  model_max_length  [512, 8193]     None  [512, 1024, 2048, 4096, 8192]
    2        batch_size     [1, 129]     None  [1, 2, 4, 8, 16, 32, 64, 128]



Measurement Space:

                                                    experiment  supported                    target-property
  0   SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True        gpu_compute_utilization_min
  1   SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True        gpu_compute_utilization_avg
  2   SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True        gpu_compute_utilization_max
  3   SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True         gpu_memory_utilization_min
  4   SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True         gpu_memory_utilization_avg
  5   SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True         gpu_memory_utilization_max
  6   SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True        gpu_memory_utilization_peak
  7   SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True            cpu_compute_utilization
  8   SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True             cpu_memory_utilization
  9   SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True                      train_runtime
  10  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True           train_samples_per_second
  11  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True             train_steps_per_second
  12  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True            train_tokens_per_second
  13  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True    train_tokens_per_gpu_per_second
  14  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True                    model_load_time
  15  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True          dataset_tokens_per_second
  16  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True  dataset_tokens_per_second_per_gpu
  17  SFTTrainer.finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0       True                           is_valid



Sample Store identifier: '2351e8'
```
<!-- markdownlint-enable line-length -->

The output shows the unique Discovery Space identifier, the Entity Space (80
Entities across 7 dimensions), and the Measurement Space (one Experiment with
17 target properties). Together these define exactly what can be measured and
what the resulting data will look like.

## Measurement Space and Entity Space Compatibility

Since an [Experiment](actuators.md#experiments) declares the inputs it needs,
an Entity can only be measured by that Experiment if its
[constitutive property](properties-and-domains.md#property-types) values
satisfy those input requirements.

Since a [Measurement Space](actuators.md#measurement-space) is a set of
Experiments, it defines a set of required constitutive properties. An Entity
Space must therefore contain all those properties, and each Entity Space
[Property Domain](properties-and-domains.md#property-domain-types) must be a
**subdomain** of the corresponding Experiment's input domain.

In practice this means the Experiment's declared input domains define the
**maximum possible extent** of any Entity Space used with that Measurement
Space. Your Entity Space is always a focused subset within those bounds. For
example, if an Experiment accepts `batch_size` values from 1 to 4096, your
Entity Space can restrict that to `[1, 2, 4, 8, 16]` — but it cannot extend
beyond `[1, 4096]`.

| | Full Experiment extent | Focused Entity Space subset |
| --- | --- | --- |
| `batch_size` | `[1, 4097]` interval 1 | `[1, 2, 4, 8, 16, 32, 64, 128]` |
| `model_name` | 40 model names | `[granite-3-8b, llama3-8b]` |
| `number_gpus` | `[0, 33]` interval 1 | `[2, 4]` |

You can inspect the full extent of an Experiment's inputs with
`ado get experiments --details`.

## Sampling and Measurement

Data is added to a Discovery Space by running an **operation** on it, for
example a Random Walk or a Bayesian optimisation. The operation selects
Entities from the Entity Space, applies the Experiments in the Measurement
Space to them and stores the results in the Sample Store. Operations are
described in the [resources documentation](../resources/operation.md).

An Entity and its measurements only become **associated with a Discovery Space**
when an operation on that space has sampled them. Even if the underlying Sample
Store already contains compatible measurements from another Discovery Space,
those results are not automatically attributed to this one — attribution requires
an explicit operation. This prevents uncontrolled inheritance of data from other
spaces.

At any point in time a Discovery Space therefore has:

- Entities that have been sampled and successfully measured
- Entities that have been sampled but whose measurements failed
- Entities that have not yet been sampled

> [!NOTE]
> You can still query compatible data across spaces when needed
> — see [Shared Sample Stores](data-sharing.md).

## Discovery Space vs DataFrame

For users familiar with `pandas`, the table below summarises how a Discovery
Space relates to a DataFrame. The key difference is that a Discovery Space
*knows* its schema and how to fill it, and shares data from a common source
rather than holding a private copy.

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable MD060 -->
| | DataFrame | Discovery Space |
| --- | --- | --- |
| Column definition | Ad-hoc — defined when created; meaning communicated separately | Defined by the Discovery Space: Entity Space dimensions + Measurement Space target properties |
| How to fill missing data | Not defined — a DataFrame just holds data | Defined by the Measurement Space: run the Experiments |
| Data sharing | Not possible — a DataFrame is a static, private object | Yes — values are fetched from a shared Sample Store on demand |
<!-- markdownlint-enable MD060 -->
<!-- markdownlint-enable line-length -->
