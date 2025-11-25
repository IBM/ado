<!-- markdownlint-disable-next-line first-line-h1 -->
A Discovery Space is made up of an [`Entity Space`](entity-spaces.md) and a
[`Measurement Space`](actuators.md#measurement-space). The `Entity Space`
defines the things you want to measure and the `Measurement Space` how you want
to measure them.

A Discovery Space is also associated with a [Sample Store](data-sharing.md)
where measurement results and entities are recorded.

## Example: Fine-Tuning Deployment Configuration Discovery Space

<!-- markdownlint-disable descriptive-link-text -->
We can combine the Entity Space example for fine-tuning deployment configuration
[here](entity-spaces.md#example-fine-tuning-deployment-configuration) with one
of the experiments from the `SFTTrainer` actuator to create the following
Discovery Space:
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

Here we can see:

- A unique id for the discovery space
- The entity space
- For each experiment in the measurement space (in this case just one) the
  target properties it measures.

## Sampling and Measurement

A Discovery Space created with an empty Sample Store has no data associated with
it i.e. no sampled and measured entities. Adding data requires applying an
operation, like a Random Walk, to the Discovery Space. This operation samples
entities from the Entity Space, measures them according to the Measurement Space
experiments, and places the results into the Sample Store.

Therefore, at any given point in time a Discovery Space will have some number of

- sampled and measured entities
- sampled and unmeasured entities (because the measurements failed)
- unsampled entities

The first two will have corresponding data in the Sample Store.

## Comparison: Discovery Space and a DataFrame

Comparing a Discovery Space with a DataFrame can help clarify the concept and
also illustrate the benefits

### A Discovery Space defines a DataFrame schema

When you create a Discovery Space you can imagine you have created a DataFrame
schema where:

1. There are Columns for each entity space dimension
2. There are Columns for each measurement space property
3. Each row is an entity

If we were to look at the example fine-tuning deployment configuration Discovery
Space this would look like (the rows and columns are truncated)

<!-- markdownlint-disable line-length -->
| model_id | gpu_type              | batch_size | model_max_length | number_gpus | ... | finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0.dataset_tokens_per_second | finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0.gpu_memory_utilization_peak | ... |
| -------- | --------------------- | ---------- | ---------------- | ----------- | --- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------- | --- |
| lama3-8b | NVIDIA-A100-80GB-PCIe | 2          | 512              | 2           | ... | UNK                                                                     | UNK                                                                       | ... |
| lama3-8b | NVIDIA-A100-80GB-PCIe | 4          | 512              | 2           | ... | UNK                                                                     | UNK                                                                       | ... |
| lama3-8b | NVIDIA-A100-80GB-PCIe | 8          | 512              | 2           | ... | UNK                                                                     | UNK                                                                       | ... |
| ...      | ...                   | ...        | ...              | ...         | ... | ...                                                                     | ...                                                                       | ... |
<!-- markdownlint-enable line-length -->

This DataFrame has 80 rows, one for each entity, and (4+3+17) columns, one for
each of the 7 constitutive properties and the 17 target properties of
`finetune-lora-fsdp-r-4-a-16-tm-default-v1.2.0.`

We can fill all the entity space columns for all the rows as we know the full
space. No measurements have taken place so all the measurement values are
unknown

### A Discovery Space defines how to fill all the data in the DataFrame

In the above example the columns associated with the measurement space have no
data. However, the Discovery Space specifies exactly how to obtain this data, as
it defines the actual experiments, supplied by actuators, that you can execute
to get it.

Using the Discovery Space at any point we can choose a row (entity) with no
measurement and get the measurements

### A Discovery Space populates the schema from a shared external source

A Discovery Space is a view rather than a container.

This means when you generate a DataFrame from a Discovery Space the data in the
rows is fetched from a shared-source. If someone else measured an entity that
corresponds to one of the rows in your DataFrame it will be automatically
populated.

As operations are run on a Discovery Space the rows in the table become filled
in. You can choose to look at:

1. Rows filled in by operations on this space (Entities sampled and measured via
   this Discovery Space)
2. Rows filled in by operations on other spaces (Entities sampled and
   measured via any Discovery Space using same Sample Store)
3. Rows not filled in at all (Unmeasured entities)

### Summary

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable MD060 -->
| Method          | Column Definition                                                                                                                   | Defines how to acquire missing data?       | Data Sharing                                                  |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| DataFrame       | Ad-Hoc. The data-frame creator defines the columns when it is created. The meaning of the columns must be communicated separately, | Not defined. The DataFrame just holds data | Not possible. A DataFrame is a static object                  |
| Discovery Space | Defined by the discovery space. A set of Entity Space columns and Measurement Space columns.                                        | Yes ,defined by the MeasurementSpace       | Yes, values are loaded from a distributed shared db on demand |
<!-- markdownlint-enable MD060 -->
<!-- markdownlint-enable line-length -->