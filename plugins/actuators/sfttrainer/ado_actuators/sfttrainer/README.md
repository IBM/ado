# Currently supported experiments

Table of contents

- [Full Fine-Tuning Experiments](#full-fine-tuning-experiments)
- [Full Fine-Tuning Experiments for exploring GPU Out Of Memory and Transient Errors](#full-fine-tuning-experiments-for-exploring-gpu-out-of-memory-and-transient-errors)
- [LORA Fine-Tuning Experiments](#lora-fine-tuning-experiments)
- [GPTQ-LORA Fine-Tuning Experiments](#gptq-lora-fine-tuning-experiments)
- [PT Fine-Tuning Experiments](#pt-fine-tuning-experiments)

## Overview

The `SFTTrainer` actuator provides a flexible and scalable interface for running
supervised fine-tuning (SFT) experiments on large language and vision-language
models. It supports a variety of fine-tuning strategies including full
fine-tuning, LoRA, QPTQ-LoRA, and prompt-tuning across both text-to-text and
image-to-text datasets.

Designed for high-performance and distributed environments, `SFTTrainer`
supports:

- **Single-GPU**, **multi-GPU**, and **multi-node** training
- **Distributed Data Parallel (DDP)** and **Fully Sharded Data Parallel (FSDP)**
  strategies
- **RDMA over Converged Ethernet (RoCE)** for optimized multi-node communication
- **Ray-based task scheduling**, enabling execution on both Kubernetes clusters
  and bare-metal infrastructure

Under the hood, this actuator wraps the
[fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning)
library, which itself builds on the
[`SFTTrainer` API from Hugging Face Transformers](https://huggingface.co/docs/trl/sft_trainer).
This layered design allows users to leverage the robustness of the Hugging Face
ecosystem while benefiting from ado’s orchestration and reproducibility
features.

<!-- markdownlint-disable-next-line no-duplicate-heading -->
### Requirements

The SFTTrainer actuator currently **supports only Python 3.10, 3.11, 3.12**.

[fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning) imports
packages like `flash-attn` and `mamba-ssm`, which import `torch` during their  
build phase. This means the base virtual environment of your Ray workers must  
already include the appropriate version of `torch`:

<!-- markdownlint-disable line-length -->
- **`fms-hf-tuning <= 2.8.2`**  
  - Install `torch==2.4.1`  
  - For RayClusters on Kubernetes, use: `quay.io/ado/ado:1.0.1-py310-cu121-ofed2410v1140`

- **`fms-hf-tuning > 2.8.2`**  
  - Install `torch==2.6.0`  
    - Requires Python 3.11  
  - For RayClusters on Kubernetes, use: `quay.io/ado/ado:c6ba952ad79a2d86d1174fd9aaebddd8953c78cf-py311-cu121-ofed2410v1140`
<!-- markdownlint-enable line-length -->

## Full Fine-Tuning Experiments

### finetune_full_benchmark-v1.0.0

An experiment instance:

- performs full fine tuning
  - You may notice that even large-memory GPUs like the 80GB variant of the
    NVIDIA A100 chip need at least 2 GPUs to train models as big as 13B
    parameters.
- the training data is artificial
- `use_flash_attn` is set to True
- `packing` is set to False
- `torch_dtype` is set to `bfloat16` by default, can also be float16
- uses the `FSDP` distributed backend for multi-gpu runs by default, can also be
  `DDP`
- multi-gpu runs with FSDP and DDP backends use 1 process per GPU (via
  `accelerate`)
- runs 1 epoch by default, can also run a custom number of steps
- does not save checkpoint
- loads weights from a PVC
- request 2 CPU cores per GPU device (with a minimum of 2 cores)

For FSDP runs we use the following `accelerate_config.yml` YAML file:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: "no"
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: ${fsdp_sharding_strategy}
  fsdp_state_dict_type: ${fsdp_state_dict_type}
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: ${accelerate_config_fsdp_transformer_layer_cls_to_wrap}
machine_rank: { $THE MACHINE RANK - always 0 for single-node runs }
main_training_function: main
mixed_precision: ${accelerate_config_mixed_precision}
num_machines: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: ${SOME_PORT}
num_processes: ${NUM_GPUS}
```

For DDP runs we use this instead:

```yaml
compute_environment: LOCAL_MACHINE
debug: False
downcast_bf16: no
distributed_type: MULTI_GPU
machine_rank: { $THE MACHINE RANK - always 0 for single-node runs }
main_training_function: main
mixed_precision: ${accelerate_config_mixed_precision}
num_machines: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: ${SOME_PORT}
num_processes: ${NUM_GPUS}
```

Commandline:

<!-- markdownlint-disable line-length -->
```commandline
accelerate launch --config_file ${PATH_ACCELERATE_CONFIG} --num_processes ${NUMBER_GPUS} \
  ${PATH_TO_OUR_WRAPPER_OF_FMS_HF_TUNING_SFT_TRAINER} --model_name_or_path ${MODEL} \
  --torch_dtype bfloat16 --use_flash_attn True --training_data_path ${DATASET_PATH} \
  --response_template "\n### Response:" --dataset_text_field output --log_level debug \
  --num_train_epochs 1 --per_device_train_batch_size ${BATCH_SIZE/NUM_GPUS} \
  --max_seq_length ${MODEL_MAX_LENGTH} --eval_strategy no --output_dir ${RANDOM_DIR} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --save_strategy no \
  --learning_rate 1e-05 --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --logging_steps 1 --include_tokens_per_second True --gradient_checkpointing True \
  --packing False --peft_method none --optim ${OPTIM} --bf16 ${BF16} \
  --gradient_checkpointing_kwargs='{"use_reentrant": ${GRADIENT_CHECKPOINTING_USE_REENTRANT}}' \
  --fast_moe ${FAST_MOE}
```
<!-- markdownlint-enable line-length -->

**Note**: `--fast_moe` is only supported for fms-hf-tuning v2.4.0+

We use a thin wrapper of `sft_trainer.py` which injects a custom Callback that
exports the metrics collected by AIM. You can repeat our experiments by just
pointing the above command-line to `sft_trainer.py` from the `fms-hf-tuning`
package.

Versioning:

- Actuator version: `2.1.0`
- fms-hf-tuning versions:
  - 3.1.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.1.0_9aca2139f4244f500cf2f5b1a0fe2ef3f8251a82.txt](packages/fms-hf-tuning_v3.1.0_9aca2139f4244f500cf2f5b1a0fe2ef3f8251a82.txt)
  - 3.0.0.1 (this is a phony release)
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.0.0.1_51875160343064a1056e0105b7971ed8d9f26854.txt](packages/fms-hf-tuning_v3.0.0.1_51875160343064a1056e0105b7971ed8d9f26854.txt)
  - 3.0.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.0.0_d8cb1cbacfbab7ed23e91151f59516766ab339e2.txt](packages/fms-hf-tuning_v3.0.0_d8cb1cbacfbab7ed23e91151f59516766ab339e2.txt)
  - 2.8.2
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.8.2_ad594c7270e934679d48286aa87c5ade7bfc54e2.txt](packages/fms-hf-tuning_v2.8.2_ad594c7270e934679d48286aa87c5ade7bfc54e2.txt)
  - 2.7.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.7.1_456fe2a840e7f0b2d8d48c04cefe24faddbf261a.txt](packages/fms-hf-tuning_v2.7.1_456fe2a840e7f0b2d8d48c04cefe24faddbf261a.txt)
  - 2.6.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.6.0_53f2babaddf07c85f5274167af9aaa947f19faf3.txt](packages/fms-hf-tuning_v2.6.0_53f2babaddf07c85f5274167af9aaa947f19faf3.txt)
  - 2.5.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.5.0_6f9bab223987732826f625fc7a522a78b58697fb.txt](packages/fms-hf-tuning_v2.5.0_6f9bab223987732826f625fc7a522a78b58697fb.txt)
  - 2.4.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.4.0_76bd76d0cfef0852e8490c344b791a35a1080ead.txt](packages/fms-hf-tuning_v2.4.0_76bd76d0cfef0852e8490c344b791a35a1080ead.txt)
  - 2.3.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.3.1_3ec30a0f9c47b0b6b9f43ce9200ab4ff24ed01e8.txt](packages/fms-hf-tuning_v2.3.1_3ec30a0f9c47b0b6b9f43ce9200ab4ff24ed01e8.txt)
  - 2.2.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.2.1_e6f7a2205c06e703c6b22bdcc5e1f248823c2a2e.txt](packages/fms-hf-tuning_v2.2.1_e6f7a2205c06e703c6b22bdcc5e1f248823c2a2e.txt)
  - 2.1.2 (default)
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.2_1e82e020f64d5a53acf98eecccb33c3597881b5e.txt](packages/fms-hf-tuning_v2.1.2_1e82e020f64d5a53acf98eecccb33c3597881b5e.txt)
  - 2.1.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.1_e2ac09183d8ba29084e110fd16b6b6c872e4a267.txt](packages/fms-hf-tuning_v2.1.1_e2ac09183d8ba29084e110fd16b6b6c872e4a267.txt)
  - 2.1.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.0_8f168183a70b41cb66902f438ecba7734144138c.txt](packages/fms-hf-tuning_v2.1.0_8f168183a70b41cb66902f438ecba7734144138c.txt)
  - 2.0.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.0.1_9b8245e74144f7ee73b7241a1687b6c77f0eb2e4.txt](packages/fms-hf-tuning_v2.0.1_9b8245e74144f7ee73b7241a1687b6c77f0eb2e4.txt)

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Requirements

- The S3 bucket `watson.runtime.wisdom.model.us-south` mounted under
  `/ibm-research-models`
  ([instructions](../../../../examples/fms-hf-tuning/README.md)).
- The PVC `hf-models-pvc` mounted under `/hf-models-pvc` - should contain the
  models:
  - LLaMa/models/hf/13B/
  - LLaMa/models/hf/7B/
  - LLaMa/models/hf/llama2-70b/
  - LLaMa/models/hf/llama3-70b/
  - LLaMa/models/hf/llama3-8b/
  - LLaMa/models/hf/llama3.1-405b/
  - LLaMa/models/hf/llama3.1-70b/
  - LLaMa/models/hf/llama3.1-8b/
  - Mixtral-8x7B-Instruct-v0.1/
  - allam-1-13b-instruct-20240607/
  - granite-13b-base-v2/step_300000_ckpt/
  - granite-20b-code-base-v2/step_280000_ckpt/
  - granite-34b-code-base/
  - granite-8b-code-base/
  - granite-8b-japanese-base-v1-llama/
  - mistralai-mistral-7b-v0.1/
  - mistral-large/fp16_240620
- The PVC `ray-disorch-storage` mounted under `/data` with the preprocessed
  `artificial-dataset` files
  (<https://github.ibm.com/ai-foundation/watson-fm-stack-tracker/issues/550>)
  under `/data/fms-hf-tuning/artificial-dataset`

#### Entity space

Required:

- model_name: Supported models:
  <!-- markdownlint-disable-next-line line-length -->
  `["granite-3b-1.5", "hf-tiny-model-private/tiny-random-BloomForCausalLM", "llama-7b", "granite-13b-v2", "llama-13b", "granite-20b-v2", "granite-7b-base", "granite-8b-japanese", "granite-8b-code-base", "granite-34b-code-base", "mistral-7b-v0.1", "llama3-8b", "llama3-70b", "mixtral-8x7b-instruct-v0.1", "llama2-70b", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "granite-3b-code-base-128k", "granite-8b-code-base-128k", "allam-1-13b", "granite-3-8b", "granite-3.1-2b", "granite-3.1-8b-instruct", "mistral-123b-v2", "granite-3.1-3b-a800m-instruct", "granite-vision-3.2-2b", "smollm2-135m", "llava-v1.6-mistral-7b", "granite-4.0-micro", "granite-4.0-h-1b", "granite-4.0-350m", "granite-4.0-h-small", "granite-4.0-h-micro", "granite-4.0-h-tiny", "granite-3.3-8b"]`
- model_max_length: Maximum sequence length. Sequences will be right padded (and
  possibly truncated)
- number_gpus: The effective number of GPUs (to be evenly distributed to
  `number_nodes` machines)
- batch_size: the effective batch_size (will be evenly distributed to max(1,
  number_gpus) devices)
- gpu_model: The value of the kubernetes node label `nvidia.com/gpu.prod` for
  example
  - `NVIDIA-A100-80GB-PCIe`
  - `NVIDIA-A100-SXM4-80GB`
  - `NVIDIA-H100-PCIe`

Optional:

- dataset_id: Default is `news-tokens-16384plus-entries-4096`. Available options
  are:
  - `news-chars-512-entries-4096`: 4096 entries with samples of 512 + 127
    (prompt) + 512 characters
  - `news-chars-1024-entries-4096`: 4096 entries with samples of 1024 + 127
    (prompt) + 1024 characters
  - `news-chars-2048-entries-4096`: 4096 entries with samples of 2048 + 127
    (prompt) + 2048 characters
  - `news-tokens-16384plus-entries-4096`: 4096 entries, each entry has least
    16384 tokens when tokenized with any of the granite-13b-v2, llama-13b-v2,
    llama-7b, or granite-20b-v2 tokenizers
  - `vision-384x384-16384plus-entries-4096`: A vision dataset containing 4096
    entries. Each entry includes at least 16384 tokens when tokenized with
    `granite-vision-3.2-2b`, and consists of repeated copies of a single image
    with dimensions 384×384.
  - `vision-384x768-16384plus-entries-4096`: Similar to the above, this dataset
    also contains 4096 entries with a minimum of 16384 tokens per entry
    (tokenized using `granite-vision-3.2-2b`). Each entry uses repeated copies
    of a single image sized 384×768.
- gradient_checkpointing: Default is `True`. If `True`, use gradient
  checkpointing to save memory (i.e. higher batchsizes) at the expense of slower
  backward pass
- gradient_accumulation_steps: Default is 4. Number of update steps to
  accumulate before performing a backward/update pass. Only takes effect when
  gradient_checkpointing is True
- torch_dtype: Default is `bfloat16`. One of `bfloat16`, `float32`, `float16`
- max_steps: Default is `-1`. The number of optimization steps to perform. Set
  to -1 to respect num_train_epochs instead.
- num_train_epochs: Default is `1.0`. How many epochs to run. Ignored if
  max_steps is greater than 0.
- stop_after_seconds: Default is `-1.0`. If set, the optimizer will be asked to
  stop after the specified time elapses. The check is performed after the end of
  each training step.
- auto_stop_method: The default value is `None`. This parameter defines the
  method used to automatically stop the fine-tuning job. Supported values are
  `WARMUP_60S_STABLE_120S_OR_10_STEPS` and `None`. If set to
  `WARMUP_60S_STABLE_120S_OR_10_STEPS`, the job stops after spending at least 60
  seconds in the warmup phase plus the longer of 120 seconds or the duration of
  10 optimization steps. This method excludes the first 60 seconds of training
  when calculating throughput and system metrics.
- distributed_backend: Default is `FSDP` for multi-gpu measurements, `None`
  (i.e. Data Parallel (DP)) for single-gpu measurements. Which pytorch backend
  to use when training with multiple GPU devices.
- number_nodes: Default is `1`. If set, actuator distributes tasks on multiple
  nodes. Each Node will use number_gpus/number_nodes GPUs. Each Node will use 1
  process for each GPU it uses
- fms_hf_tuning_version: Default is `2.1.2`. Which version of fms-hf-tuning to
  use. Available options are: `3.1.0`, `3.0.0.1`, `3.0.0`, `2.8.2`, `2.7.1`,
  `2.6.0`, `2.5.0`, `2.4.0`, `2.3.1`, `2.2.1`, `2.1.2`, `2.1.0`, `2.0.1`
- enable_roce: Default is `False`. This setting is only in effect for multi-node
  runs. It controls whether RDMA over Converged Ethernet (RoCE) is switched on
  or not.
- fast_moe: Default is `0`. Configures the amount of expert parallel sharding.
  number_gpus must be divisible by it
- fast_kernels: Default is `None`. Switches on fast kernels, the value is a list
  with strings of boolean values for
  `[fast_loss, fast_rms_layernorm, fast_rope_embeddings]`
- optim: Default is `adamw_torch`. The optimizer to use. Available options are
  `adamw_hf`, `adamw_torch`, `adamw_torch_fused`, `adamw_torch_xla`,
  `adamw_torch_npu_fused`, `adamw_apex_fused`, `adafactor`,
  `adamw_anyprecision`, `adamw_torch_4bit`, `ademamix`, `sgd`, `adagrad`,
  `adamw_bnb_8bit`, `adamw_8bit`, `ademamix_8bit`, `lion_8bit`, `lion_32bit`,
  `paged_adamw_32bit`, `paged_adamw_8bit`, `paged_ademamix_32bit`,
  `paged_ademamix_8bit`, `paged_lion_32bit`, `paged_lion_8bit`, `rmsprop`,
  `rmsprop_bnb`, `rmsprop_bnb_8bit`, `rmsprop_bnb_32bit`, `galore_adamw`,
  `galore_adamw_8bit`, `galore_adafactor`, `galore_adamw_layerwise`,
  `galore_adamw_8bit_layerwise`, `galore_adafactor_layerwise`, `lomo`,
  `adalomo`, `grokadamw`, `schedule_free_adamw`, `schedule_free_sgd`
- bf16: Default is `False`. Whether to use bf16 (mixed) precision instead of
  32-bit. Requires Ampere or higher NVIDIA add bf16 mixed precision support for
  NPU architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental
  API and it may change. Can be `True`, `False`.
- gradient_checkpointing_use_reentrant: Default is `False` Specify whether to
  use the activation checkpoint variant that requires reentrant autograd. This
  parameter should be passed explicitly. Torch version 2.5 will raise an
  exception if use_reentrant is not passed. If use_reentrant=False, checkpoint
  will use an implementation that does not require reentrant autograd. This
  allows checkpoint to support additional functionality, such as working as
  expected with torch.autograd.grad and support for keyword arguments input into
  the checkpointed function. Can be `True`, `False`.
- fsdp_sharding_strategy: Default is `FULL_SHARD`. [1] FULL_SHARD (shards
  optimizer states, gradients and parameters), " [2] SHARD_GRAD_OP (shards
  optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD
  (shards optimizer states, gradients and parameters within each node while
  each node has full copy - equivalent to FULL_SHARD for single-node runs),
  [5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients
  within each node while each node has full copy). For more information, please
  refer the official PyTorch docs.
- fsdp_state_dict_type: Default is `FULL_STATE_DICT`. [1] FULL_STATE_DICT, [2]
  LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT
- fsdp_use_orig_params: Default is `True`. If True, allows non-uniform
  `requires_grad` during init, which means support for interspersed frozen and
  trainable parameters. (useful only when `use_fsdp` flag is passed).
- accelerate_config_mixed_precision: Default is `no`. Whether to use mixed
  precision training or not. Choose from `no`,`fp16`,`bf16` or `fp8`. `fp8`
  requires the installation of transformers-engine.
- accelerate_config_fsdp_transformer_layer_cls_to_wrap: Default is None. List of
  transformer layer class names (case-sensitive) to wrap, e.g,
  `GraniteDecoderLayer`, `LlamaDecoderLayer`, `MistralDecoderLayer`,
  `BertLayer`, `GPTJBlock`, `T5Block` ... (useful only when using FSDP)
- dataset_text_field: Default is None. Training dataset text field containing
  single sequence. Either the dataset_text_field or data_formatter_template need
  to be supplied. For running vision language model tuning pass the column name
  for text data.
- dataset_image_field: Default is None. For running vision language model tuning
  pass the column name of the image data in the dataset.
- remove_unused_columns: Default is True. Remove columns not required by the
  model when using an nlp.Dataset.
- dataset_kwargs_skip_prepare_dataset: Default is False. When True, configures
  trl to skip preparing the dataset

> **NOTE**: Because running `accelerate` with a single gpu is unsupported, when
> setting `number_gpus` to 1 this experiment actually runs the
> `tuning.sft_trainer` script directly (i.e. a DataParallel (DP) run).

#### Measured properties

We use AIM to collect profiling metadata. Then we convert the timeseries that
AIM collects into the metrics below.

- gpu_compute_utilization_min (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_compute_utilization_avg (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_compute_utilization_max (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_memory_utilization_min (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_memory_utilization_avg (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_memory_utilization_max (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_power_percent_min (0.0 if not using any GPUs): Measured in Percentages (0
  to 100) (see note 1)
- gpu_power_percent_avg (0.0 if not using any GPUs): Measured in Percentages (0
  to 100) (see note 1)
- gpu_power_percent_max (0.0 if not using any GPUs): Measured in Percentages (0
  to 100) (see note 1)
- gpu_power_watts_min (0.0 if not using any GPUs): Measured in Watts (see
  Note 1)
- gpu_power_watts_avg (0.0 if not using any GPUs): Measured in Watts (see
  Note 1)
- gpu_power_watts_max (0.0 if not using any GPUs): Measured in Watts (see
  Note 1)
- gpu_memory_utilization_peak (0.0 if not using any GPUs): peak GPU memory
  utilization percentage across all devices
- cpu_compute_utilization: Measured in Percentages (0 to 100 where 100 means 1
  full core) (see note 2)
- cpu_memory_utilization: Measured in Percentages (0 to 100) taken from AIM (see
  note 2)
- train_runtime: Measured in seconds
- train_samples_per_second
- train_steps_per_second
- train_tokens_per_second: How many tokens (including padding tokens) the run
  processed every second (for FSDP this is estimated from num_gpus \*
  rank_0_train_tokens_per_second). Omitted when stop_after_seconds is greater
  than 0
- train_tokens_per_gpu_per_second (will be equal to train_tokens_per_second when
  number_gpus <= 1, for FSDP this is reported using just rank 0). Omitted when
  stop_after_seconds is greater than 0
- dataset_tokens_per_second: How many tokens from the dataset the run processed
  every second (see note 3)
- dataset_tokens_per_second_per_gpu: How many tokens from the dataset the run
  processed every second per GPU (see note 3)
- is_valid (see `is_valid logic`)

Notes:

- (1) They are reported as the min/max/avg of the average in the timeseries
  metrics that AIM collects for the in-use GPUs of a run.
- (2) CPU compute and memory utilization are percentages. They are reported as
  the min/max/avg of the metrics that AIM collects for the sft_trainer.py
  process. The total memory capacity of the nodes varies from 100 GB to 400 GB.
  Currently, we do not store this information in our database.
- (3) `dataset_tokens_per_second` and `dataset_tokens_per_second_per_gpu` take
  into account the `tokenizer.model_max_length` and `max_seq_length` (i.e. for
  each entry, we report
  `min(len(tokens(entry["output"])), tokenizer.model_max_length, max_seq_length))`.

#### is_valid logic

A run for an entity is invalid if:

1. `batch_size` cannot be evenly divided by `number_gpus` (i.e.
   `batch_size % number_gpus != 0`)
2. `number_gpus` cannot be evenly divided by `number_nodes` (i.e.
   `number_gpus % number_nodes != 0`)
3. `number_nodes` must be greater than 0
4. `batch_size` must be greater than 0
5. if `number_gpus` is greater than 0 then `gpu_model` must be a non-empty
   string
6. if `fast_moe` is set and `number_gpus` is not divisible by it
7. if `fast_moe` is set and the `num_local_experts` of the Mixture of Experts
   (MoE) model is not divisible by `fast_moe` (which is interpreted as
   `ep_degrees` by fms-hf-tuning)

Runs raising the following errors are considered invalid due to running out of
GPU memory:

- `torch.cuda.OutOfMemoryError`
- `RuntimeError: CUDA error: an illegal memory access was encountered`

Measurements raising any other exception (including for example `RuntimeError`
containing the string `NCCL Error`) are considered to have `Failed`. They will
not contain the `is_valid` measured property, or any other property for that
matter. `Failed` measurements do not record any properties and can be repeated.

## Full Fine-Tuning Experiments for exploring GPU Out Of Memory and Transient Errors

### finetune_full_stability-v1.0.0

An experiment instance:

- performs full fine-tuning 5 times and reports the fraction of tasks that ran
  out of GPU memory, exhibited some unknown error, or completed successfully
  - You may notice that even large-memory GPUs like the 80GB variant of the
    NVIDIA A100 chip need at least 2 GPUs to train models as big as 13B
    parameters.
- the training data is artificial
- `use_flash_attn` is set to True
- `packing` is set to False
- `torch_dtype` is set to `bfloat16`
- uses the `FSDP` distributed backend
- runs 5 optimization steps
- does not save checkpoint
- loads weights from a PVC
- request 2 CPU cores per GPU device (with a minimum of 2 cores)

We use the following `accelerate_config.yml` YAML file for all models:

```yaml
compute_environment: LOCAL_MACHINE
debug: False
distributed_type: FSDP
downcast_bf16: "no"
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
machine_rank: 0
main_training_function: main
mixed_precision: "no"
num_machines: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: { $SOME_PORT }
num_processes: { $NUM_GPUS }
```

Commandline:

<!-- markdownlint-disable line-length -->
```commandline
accelerate launch --config_file ${PATH_ACCELERATE_CONFIG} --num_processes ${NUMBER_GPUS} \
  ${PATH_TO_OUR_WRAPPER_OF_FMS_HF_TUNING_SFT_TRAINER} --model_name_or_path ${MODEL} \
  --torch_dtype bfloat16 --use_flash_attn True --training_data_path ${DATASET_PATH} \
  --response_template "\n### Response:" --dataset_text_field output --log_level debug \
  --max_steps -1 --per_device_train_batch_size ${BATCH_SIZE/NUM_GPUS} \
  --max_seq_length ${MODEL_MAX_LENGTH} --eval_strategy no --output_dir ${RANDOM_DIR} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --save_strategy no \
  --learning_rate 1e-05 --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --logging_steps 1 --include_tokens_per_second True --gradient_checkpointing True \
  --packing False --peft_method none --optim ${OPTIM} --bf16 ${BF16} \
  --gradient_checkpointing_kwargs='{"use_reentrant": ${GRADIENT_CHECKPOINTING_USE_REENTRANT}}' \
  --fast_moe ${FAST_MOE}
```
<!-- markdownlint-enable line-length -->

**Note**: `--fast_moe` is only supported for fms-hf-tuning v2.4.0+

We use a thin wrapper of `sft_trainer.py` which injects a custom Callback that
exports the metrics collected by AIM. You can repeat our experiments by just
pointing the above command-line to `sft_trainer.py` from the `fms-hf-tuning`
package.

Versioning:

- Actuator version: `2.1.0`
- fms-hf-tuning versions:
  - 3.1.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.1.0_9aca2139f4244f500cf2f5b1a0fe2ef3f8251a82.txt](packages/fms-hf-tuning_v3.1.0_9aca2139f4244f500cf2f5b1a0fe2ef3f8251a82.txt)
  - 3.0.0.1 (this is a phony release)
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.0.0.1_51875160343064a1056e0105b7971ed8d9f26854.txt](packages/fms-hf-tuning_v3.0.0.1_51875160343064a1056e0105b7971ed8d9f26854.txt)
  - 3.0.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.0.0_d8cb1cbacfbab7ed23e91151f59516766ab339e2.txt](packages/fms-hf-tuning_v3.0.0_d8cb1cbacfbab7ed23e91151f59516766ab339e2.txt)
  - 2.8.2
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.8.2_ad594c7270e934679d48286aa87c5ade7bfc54e2.txt](packages/fms-hf-tuning_v2.8.2_ad594c7270e934679d48286aa87c5ade7bfc54e2.txt)
  - 2.7.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.7.1_456fe2a840e7f0b2d8d48c04cefe24faddbf261a.txt](packages/fms-hf-tuning_v2.7.1_456fe2a840e7f0b2d8d48c04cefe24faddbf261a.txt)
  - 2.6.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.6.0_53f2babaddf07c85f5274167af9aaa947f19faf3.txt](packages/fms-hf-tuning_v2.6.0_53f2babaddf07c85f5274167af9aaa947f19faf3.txt)
  - 2.5.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.5.0_6f9bab223987732826f625fc7a522a78b58697fb.txt](packages/fms-hf-tuning_v2.5.0_6f9bab223987732826f625fc7a522a78b58697fb.txt)
  - 2.4.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.4.0_76bd76d0cfef0852e8490c344b791a35a1080ead.txt](packages/fms-hf-tuning_v2.4.0_76bd76d0cfef0852e8490c344b791a35a1080ead.txt)
  - 2.3.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.3.1_3ec30a0f9c47b0b6b9f43ce9200ab4ff24ed01e8.txt](packages/fms-hf-tuning_v2.3.1_3ec30a0f9c47b0b6b9f43ce9200ab4ff24ed01e8.txt)
  - 2.2.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.2.1_e6f7a2205c06e703c6b22bdcc5e1f248823c2a2e.txt](packages/fms-hf-tuning_v2.2.1_e6f7a2205c06e703c6b22bdcc5e1f248823c2a2e.txt)
  - 2.1.2 (default)
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.2_1e82e020f64d5a53acf98eecccb33c3597881b5e.txt](packages/fms-hf-tuning_v2.1.2_1e82e020f64d5a53acf98eecccb33c3597881b5e.txt)
  - 2.1.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.1_e2ac09183d8ba29084e110fd16b6b6c872e4a267.txt](packages/fms-hf-tuning_v2.1.1_e2ac09183d8ba29084e110fd16b6b6c872e4a267.txt)
  - 2.1.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.0_8f168183a70b41cb66902f438ecba7734144138c.txt](packages/fms-hf-tuning_v2.1.0_8f168183a70b41cb66902f438ecba7734144138c.txt)
  - 2.0.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.0.1_9b8245e74144f7ee73b7241a1687b6c77f0eb2e4.txt](packages/fms-hf-tuning_v2.0.1_9b8245e74144f7ee73b7241a1687b6c77f0eb2e4.txt)

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Requirements

- The S3 bucket `watson.runtime.wisdom.model.us-south` mounted under
  `/ibm-research-models`
  ([instructions](../../../../examples/fms-hf-tuning/README.md)).
- The PVC `hf-models-pvc` mounted under `/hf-models-pvc` - should contain the
  models:
  - LLaMa/models/hf/13B/
  - LLaMa/models/hf/7B/
  - LLaMa/models/hf/llama2-70b/
  - LLaMa/models/hf/llama3-70b/
  - LLaMa/models/hf/llama3-8b/
  - LLaMa/models/hf/llama3.1-405b/
  - LLaMa/models/hf/llama3.1-70b/
  - LLaMa/models/hf/llama3.1-8b/
  - Mixtral-8x7B-Instruct-v0.1/
  - allam-1-13b-instruct-20240607/
  - granite-13b-base-v2/step_300000_ckpt/
  - granite-20b-code-base-v2/step_280000_ckpt/
  - granite-34b-code-base/
  - granite-8b-code-base/
  - granite-8b-japanese-base-v1-llama/
  - mistralai-mistral-7b-v0.1/
  - mistral-large/fp16_240620
- The PVC `ray-disorch-storage` mounted under `/data` with the preprocessed
  `artificial-dataset` files
  (<https://github.ibm.com/ai-foundation/watson-fm-stack-tracker/issues/550>)
  under `/data/fms-hf-tuning/artificial-dataset`

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Entity space

- model_name: Supported models:
  <!-- markdownlint-disable-next-line line-length -->
  `["granite-3b-1.5", "hf-tiny-model-private/tiny-random-BloomForCausalLM", "llama-7b", "granite-13b-v2", "llama-13b", "granite-20b-v2", "granite-7b-base", "granite-8b-japanese", "granite-8b-code-base", "granite-34b-code-base", "mistral-7b-v0.1", "llama3-8b", "llama3-70b", "mixtral-8x7b-instruct-v0.1", "llama2-70b", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "granite-3b-code-base-128k", "granite-8b-code-base-128k", "allam-1-13b", "granite-3-8b", "granite-3.1-2b", "granite-3.1-8b-instruct", "mistral-123b-v2", "granite-3.1-3b-a800m-instruct", "granite-vision-3.2-2b", "smollm2-135m", "llava-v1.6-mistral-7b", "granite-4.0-micro", "granite-4.0-h-1b", "granite-4.0-350m", "granite-4.0-h-small", "granite-4.0-h-micro", "granite-4.0-h-tiny", "granite-3.3-8b"]`
- dataset_id: One of
  - `news-chars-512-entries-4096`: 4096 entries with samples of 512 + 127
    (prompt) + 512 characters
  - `news-chars-1024-entries-4096`: 4096 entries with samples of 1024 + 127
    (prompt) + 1024 characters
  - `news-chars-2048-entries-4096`: 4096 entries with samples of 2048 + 127
    (prompt) + 2048 characters
  - `news-tokens-16384plus-entries-4096`: 4096 entries, each entry has least
    16384 tokens when tokenized with any of the granite-13b-v2, llama-13b-v2,
    llama-7b, or granite-20b-v2 tokenizers
  - `news-tokens-128kplus-entries-320` : 320 entries, each entry has at least
    128\*1024 tokens
  - `vision-384x384-16384plus-entries-4096`: A vision dataset containing 4096
    entries. Each entry includes at least 16384 tokens when tokenized with
    `granite-vision-3.2-2b`, and consists of repeated copies of a single image
    with dimensions 384×384.
  - `vision-384x768-16384plus-entries-4096`: Similar to the above, this dataset
    also contains 4096 entries with a minimum of 16384 tokens per entry
    (tokenized using `granite-vision-3.2-2b`). Each entry uses repeated copies
    of a single image sized 384×768.
- number_gpus: Can be 0 or more - no support for multi-node runs
- model_max_length: Maximum sequence length. Sequences will be right padded (and
  possibly truncated)
- torch_dtype: Here you can use any valid `torch_dtype` value e.g. `float32`,
  `bfloat16`, `float16`, etc
- batch_size: the effective batch_size (will be evenly distributed to max(1,
  number_gpus) devices)
- gpu_model: The value of the kubernetes node label `nvidia.com/gpu.prod` for
  example
  - `NVIDIA-A100-80GB-PCIe`
  - `NVIDIA-A100-SXM4-80GB`
  - `NVIDIA-H100-PCIe`
- gradient_accumulation_steps: Number of update steps to accumulate before
  performing a backward/update pass. Defaults to 4 when not set.

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Measured properties

- f_gpu_oom: fraction of tasks that ran out of GPU memory
- f_other_error: fraction of tasks that ran into an unknown error
- f_no_error: fraction of tasks that completed successfully
- is_valid: whether this collection of tasks is a valid point to investigate

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### is_valid logic

A run for an entity is invalid if:

1. `batch_size` cannot be evenly divided by `number_gpus` (i.e.
   `batch_size % number_gpus != 0`)
2. `number_gpus` cannot be evenly divided by `number_nodes` (i.e.
   `number_gpus % number_nodes != 0`)
3. `number_nodes` must be greater than 0
4. `batch_size` must be greater than 0
5. if `number_gpus` is greater than 0 then `gpu_model` must be a non-empty
   string
6. if `fast_moe` is set and `number_gpus` is not divisible by it
7. if `fast_moe` is set and the `num_local_experts` of the Mixture of Experts
   (MoE) model is not divisible by `fast_moe` (which is interpreted as
   `ep_degrees` by fms-hf-tuning)

Runs raising the following errors are considered invalid due to running out of
GPU memory:

- `torch.cuda.OutOfMemoryError`
- `RuntimeError: CUDA error: an illegal memory access was encountered`

Measurements raising any other exception (including for example `RuntimeError`
containing the string `NCCL Error`) are considered to have `Failed`. They will
not contain the `is_valid` measured property, or any other property for that
matter. `Failed` measurements do not record any properties and can be repeated.

## LORA Fine-Tuning Experiments

### finetune_lora_benchmark-v1.0.0

An experiment instance:

- performs LORA fine tuning
- the training data is artificial
- `use_flash_attn` is set to True
- `packing` is set to False
- `torch_dtype` is set to `bfloat16` by default, can also be float16
- uses the `FSDP` distributed backend for multi-gpu runs by default, can also be
  `DDP`
- multi-gpu runs with FSDP and DDP backends use 1 process per GPU (via
  `accelerate`)
- runs 1 epoch by default, can also run a custom number of steps
- does not save checkpoint
- loads weights from a PVC
- request 2 CPU cores per GPU device (with a minimum of 2 cores)

For FSDP runs we use the following `accelerate_config.yml` YAML file:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: "no"
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: ${fsdp_sharding_strategy}
  fsdp_state_dict_type: ${fsdp_state_dict_type}
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: ${accelerate_config_fsdp_transformer_layer_cls_to_wrap}
machine_rank: { $THE MACHINE RANK - always 0 for single-node runs }
main_training_function: main
mixed_precision: ${accelerate_config_mixed_precision}
num_machines: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: { $SOME_PORT }
num_processes: { $NUM_GPUS }
```

For DDP runs we use this instead:

```yaml
compute_environment: LOCAL_MACHINE
debug: False
downcast_bf16: no
distributed_type: MULTI_GPU
machine_rank: { $THE MACHINE RANK - always 0 for single-node runs }
main_training_function: main
mixed_precision: ${accelerate_config_mixed_precision}
num_machines: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: { $SOME_PORT }
num_processes: { $NUM_GPUS }
```

Commandline:

<!-- markdownlint-disable line-length -->
```commandline
accelerate launch --config_file ${PATH_ACCELERATE_CONFIG} --num_processes ${NUMBER_GPUS} \
  ${PATH_TO_OUR_WRAPPER_OF_FMS_HF_TUNING_SFT_TRAINER} --model_name_or_path ${MODEL} \
  --torch_dtype bfloat16 --use_flash_attn True --training_data_path ${DATASET_PATH} \
  --response_template "\n### Response:" --dataset_text_field output --log_level debug \
  --num_train_epochs 1 --per_device_train_batch_size ${BATCH_SIZE/NUM_GPUS} \
  --max_seq_length ${MODEL_MAX_LENGTH} --eval_strategy no --output_dir ${RANDOM_DIR} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --save_strategy no \
  --learning_rate 1e-05 --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --logging_steps 1 --include_tokens_per_second True --gradient_checkpointing True \
  --packing False --peft_method lora --target_modules ${SPACE SEPARATED LAYER NAMES} \
  --optim ${OPTIM} --bf16 ${BF16} \
  --gradient_checkpointing_kwargs='{"use_reentrant": ${GRADIENT_CHECKPOINTING_USE_REENTRANT}}' \
  --fast_moe ${FAST_MOE}
```
<!-- markdownlint-enable line-length -->

**Note**: `--fast_moe` is only supported for fms-hf-tuning v2.4.0+

We use a thin wrapper of `sft_trainer.py` which injects a custom Callback that
exports the metrics collected by AIM. You can repeat our experiments by just
pointing the above command-line to `sft_trainer.py` from the `fms-hf-tuning`
package.

Versioning:

- Actuator version: `2.1.0`
- fms-hf-tuning versions:
  - 3.1.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.1.0_9aca2139f4244f500cf2f5b1a0fe2ef3f8251a82.txt](packages/fms-hf-tuning_v3.1.0_9aca2139f4244f500cf2f5b1a0fe2ef3f8251a82.txt)
  - 3.0.0.1 (this is a phony release)
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.0.0.1_51875160343064a1056e0105b7971ed8d9f26854.txt](packages/fms-hf-tuning_v3.0.0.1_51875160343064a1056e0105b7971ed8d9f26854.txt)
  - 3.0.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.0.0_d8cb1cbacfbab7ed23e91151f59516766ab339e2.txt](packages/fms-hf-tuning_v3.0.0_d8cb1cbacfbab7ed23e91151f59516766ab339e2.txt)
  - 2.8.2
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.8.2_ad594c7270e934679d48286aa87c5ade7bfc54e2.txt](packages/fms-hf-tuning_v2.8.2_ad594c7270e934679d48286aa87c5ade7bfc54e2.txt)
  - 2.7.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.7.1_456fe2a840e7f0b2d8d48c04cefe24faddbf261a.txt](packages/fms-hf-tuning_v2.7.1_456fe2a840e7f0b2d8d48c04cefe24faddbf261a.txt)
  - 2.6.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.6.0_53f2babaddf07c85f5274167af9aaa947f19faf3.txt](packages/fms-hf-tuning_v2.6.0_53f2babaddf07c85f5274167af9aaa947f19faf3.txt)
  - 2.5.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.5.0_6f9bab223987732826f625fc7a522a78b58697fb.txt](packages/fms-hf-tuning_v2.5.0_6f9bab223987732826f625fc7a522a78b58697fb.txt)
  - 2.4.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.4.0_76bd76d0cfef0852e8490c344b791a35a1080ead.txt](packages/fms-hf-tuning_v2.4.0_76bd76d0cfef0852e8490c344b791a35a1080ead.txt)
  - 2.3.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.3.1_3ec30a0f9c47b0b6b9f43ce9200ab4ff24ed01e8.txt](packages/fms-hf-tuning_v2.3.1_3ec30a0f9c47b0b6b9f43ce9200ab4ff24ed01e8.txt)
  - 2.2.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.2.1_e6f7a2205c06e703c6b22bdcc5e1f248823c2a2e.txt](packages/fms-hf-tuning_v2.2.1_e6f7a2205c06e703c6b22bdcc5e1f248823c2a2e.txt)
  - 2.1.2 (default)
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.2_1e82e020f64d5a53acf98eecccb33c3597881b5e.txt](packages/fms-hf-tuning_v2.1.2_1e82e020f64d5a53acf98eecccb33c3597881b5e.txt)
  - 2.1.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.1_e2ac09183d8ba29084e110fd16b6b6c872e4a267.txt](packages/fms-hf-tuning_v2.1.1_e2ac09183d8ba29084e110fd16b6b6c872e4a267.txt)
  - 2.1.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.0_8f168183a70b41cb66902f438ecba7734144138c.txt](packages/fms-hf-tuning_v2.1.0_8f168183a70b41cb66902f438ecba7734144138c.txt)
  - 2.0.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.0.1_9b8245e74144f7ee73b7241a1687b6c77f0eb2e4.txt](packages/fms-hf-tuning_v2.0.1_9b8245e74144f7ee73b7241a1687b6c77f0eb2e4.txt)

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Requirements

- The S3 bucket `watson.runtime.wisdom.model.us-south` mounted under
  `/ibm-research-models`
  ([instructions](../../../../examples/fms-hf-tuning/README.md)).
- The PVC `hf-models-pvc` mounted under `/hf-models-pvc` - should contain the
  models:
  - LLaMa/models/hf/13B/
  - LLaMa/models/hf/7B/
  - LLaMa/models/hf/llama2-70b/
  - LLaMa/models/hf/llama3-70b/
  - LLaMa/models/hf/llama3-8b/
  - LLaMa/models/hf/llama3.1-405b/
  - LLaMa/models/hf/llama3.1-70b/
  - LLaMa/models/hf/llama3.1-8b/
  - Mixtral-8x7B-Instruct-v0.1/
  - allam-1-13b-instruct-20240607/
  - granite-13b-base-v2/step_300000_ckpt/
  - granite-20b-code-base-v2/step_280000_ckpt/
  - granite-34b-code-base/
  - granite-8b-code-base/
  - granite-8b-japanese-base-v1-llama/
  - mistralai-mistral-7b-v0.1/
  - mistral-large/fp16_240620
- The PVC `ray-disorch-storage` mounted under `/data` with the preprocessed
  `artificial-dataset` files
  (<https://github.ibm.com/ai-foundation/watson-fm-stack-tracker/issues/550>)
  under `/data/fms-hf-tuning/artificial-dataset`

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Entity space

Required:

- model_name: Supported models:
  <!-- markdownlint-disable-next-line line-length -->
  `["granite-3b-1.5", "hf-tiny-model-private/tiny-random-BloomForCausalLM", "llama-7b", "granite-13b-v2", "llama-13b", "granite-20b-v2", "granite-7b-base", "granite-8b-japanese", "granite-8b-code-base", "granite-34b-code-base", "mistral-7b-v0.1", "llama3-8b", "llama3-70b", "mixtral-8x7b-instruct-v0.1", "llama2-70b", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "granite-3b-code-base-128k", "granite-8b-code-base-128k", "allam-1-13b", "granite-3-8b", "granite-3.1-2b", "granite-3.1-8b-instruct", "mistral-123b-v2", "granite-3.1-3b-a800m-instruct", "granite-vision-3.2-2b", "smollm2-135m", "llava-v1.6-mistral-7b", "granite-4.0-micro", "granite-4.0-h-1b", "granite-4.0-350m", "granite-4.0-h-small", "granite-4.0-h-micro", "granite-4.0-h-tiny", "granite-3.3-8b"]`
- model_max_length: Maximum sequence length. Sequences will be right padded (and
  possibly truncated)
- number_gpus: The effective number of GPUs (to be evenly distributed to
  `number_nodes` machines)
- batch_size: the effective batch_size (will be evenly distributed to max(1,
  number_gpus) devices)
- gpu_model: The value of the kubernetes node label `nvidia.com/gpu.prod` for
  example
  - `NVIDIA-A100-80GB-PCIe`
  - `NVIDIA-A100-SXM4-80GB`
  - `NVIDIA-H100-PCIe`

Optional:

- dataset_id: Default is `news-tokens-16384plus-entries-4096`. Available options
  are:
  - `news-chars-512-entries-4096`: 4096 entries with samples of 512 + 127
    (prompt) + 512 characters
  - `news-chars-1024-entries-4096`: 4096 entries with samples of 1024 + 127
    (prompt) + 1024 characters
  - `news-chars-2048-entries-4096`: 4096 entries with samples of 2048 + 127
    (prompt) + 2048 characters
  - `news-tokens-16384plus-entries-4096`: 4096 entries, each entry has least
    16384 tokens when tokenized with any of the granite-13b-v2, llama-13b-v2,
    llama-7b, or granite-20b-v2 tokenizers
  - `vision-384x384-16384plus-entries-4096`: A vision dataset containing 4096
    entries. Each entry includes at least 16384 tokens when tokenized with
    `granite-vision-3.2-2b`, and consists of repeated copies of a single image
    with dimensions 384×384.
  - `vision-384x768-16384plus-entries-4096`: Similar to the above, this dataset
    also contains 4096 entries with a minimum of 16384 tokens per entry
    (tokenized using `granite-vision-3.2-2b`). Each entry uses repeated copies
    of a single image sized 384×768.
- gradient_checkpointing: Default is `True`. If `True`, use gradient
  checkpointing to save memory (i.e. higher batchsizes) at the expense of slower
  backward pass
- gradient_accumulation_steps: Default is 4. Number of update steps to
  accumulate before performing a backward/update pass. Only takes effect when
  gradient_checkpointing is True
- torch_dtype: Default is `bfloat16`. One of `bfloat16`, `float32`, `float16`
- max_steps: Default is `-1`. The number of optimization steps to perform. Set
  to -1 to respect num_train_epochs instead.
- num_train_epochs: Default is `1.0`. How many epochs to run. Ignored if
  max_steps is greater than 0.
- stop_after_seconds: Default is `-1.0`. If set, the optimizer will be asked to
  stop after the specified time elapses. The check is performed after the end of
  each training step.
- auto_stop_method: The default value is `None`. This parameter defines the
  method used to automatically stop the fine-tuning job. Supported values are
  `WARMUP_60S_STABLE_120S_OR_10_STEPS` and `None`. If set to
  `WARMUP_60S_STABLE_120S_OR_10_STEPS`, the job stops after spending at least 60
  seconds in the warmup phase plus the longer of 120 seconds or the duration of
  10 optimization steps. This method excludes the first 60 seconds of training
  when calculating throughput and system metrics.
- distributed_backend: Default is `FSDP` for multi-gpu measurements, `None`
  (i.e. Data Parallel (DP)) for single-gpu measurements. Which pytorch backend
  to use when training with multiple GPU devices.
- number_nodes: Default is `1`. If set, actuator distributes tasks on multiple
  nodes. Each Node will use number_gpus/number_nodes GPUs. Each Node will use 1
  process for each GPU it uses
- fms_hf_tuning_version: Default is `2.1.2`. Which version of fms-hf-tuning to
  use. Available options are: `3.1.0`, `3.0.0.1`, `3.0.0`, `2.8.2`, `2.7.1`,
  `2.6.0`, `2.5.0`, `2.4.0`, `2.3.1`, `2.2.1`, `2.1.2`, `2.1.0`, `2.0.1`
- enable_roce: Default is `False`. This setting is only in effect for multi-node
  runs. It controls whether RDMA over Converged Ethernet (RoCE) is switched on
  or not.
- fast_moe: Default is `0`. Configures the amount of expert parallel sharding.
  number_gpus must be divisible by it
- fast_kernels: Default is `None`. Switches on fast kernels, the value is a list
  with strings of boolean values for
  `[fast_loss, fast_rms_layernorm, fast_rope_embeddings]`
- r: Default is `4`. The LORA rank
- lora_alpha: Default is `16`. Scales the learning weights.
- optim: Default is `adamw_torch`. The optimizer to use. Available options are
  `adamw_hf`, `adamw_torch`, `adamw_torch_fused`, `adamw_torch_xla`,
  `adamw_torch_npu_fused`, `adamw_apex_fused`, `adafactor`,
  `adamw_anyprecision`, `adamw_torch_4bit`, `ademamix`, `sgd`, `adagrad`,
  `adamw_bnb_8bit`, `adamw_8bit`, `ademamix_8bit`, `lion_8bit`, `lion_32bit`,
  `paged_adamw_32bit`, `paged_adamw_8bit`, `paged_ademamix_32bit`,
  `paged_ademamix_8bit`, `paged_lion_32bit`, `paged_lion_8bit`, `rmsprop`,
  `rmsprop_bnb`, `rmsprop_bnb_8bit`, `rmsprop_bnb_32bit`, `galore_adamw`,
  `galore_adamw_8bit`, `galore_adafactor`, `galore_adamw_layerwise`,
  `galore_adamw_8bit_layerwise`, `galore_adafactor_layerwise`, `lomo`,
  `adalomo`, `grokadamw`, `schedule_free_adamw`, `schedule_free_sgd`
- bf16: Default is `False`. Whether to use bf16 (mixed) precision instead of
  32-bit. Requires Ampere or higher NVIDIA add bf16 mixed precision support for
  NPU architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental
  API and it may change. Can be `True`, `False`.
- gradient_checkpointing_use_reentrant: Default is `False` Specify whether to
  use the activation checkpoint variant that requires reentrant autograd. This
  parameter should be passed explicitly. Torch version 2.5 will raise an
  exception if use_reentrant is not passed. If use_reentrant=False, checkpoint
  will use an implementation that does not require reentrant autograd. This
  allows checkpoint to support additional functionality, such as working as
  expected with torch.autograd.grad and support for keyword arguments input into
  the checkpointed function. Can be `True`, `False`.
- fsdp_sharding_strategy: Default is `FULL_SHARD`. [1] FULL_SHARD (shards
  optimizer states, gradients and parameters), " [2] SHARD_GRAD_OP (shards
  optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD
  (shards optimizer states, gradients and parameters within each node while
  each node has full copy - equivalent to FULL_SHARD for single-node runs),
  [5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients
  within each node while each node has full copy). For more information, please
  refer the official PyTorch docs.
- fsdp_state_dict_type: Default is `FULL_STATE_DICT`. [1] FULL_STATE_DICT, [2]
  LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT
- fsdp_use_orig_params: Default is `True`. If True, allows non-uniform
  `requires_grad` during init, which means support for interspersed frozen and
  trainable parameters. (useful only when `use_fsdp` flag is passed).
- accelerate_config_mixed_precision: Default is `no`. Whether to use mixed
  precision training or not. Choose from `no`,`fp16`,`bf16` or `fp8`. `fp8`
  requires the installation of transformers-engine.
- accelerate_config_fsdp_transformer_layer_cls_to_wrap: Default is None. List of
  transformer layer class names (case-sensitive) to wrap, e.g,
  `GraniteDecoderLayer`, `LlamaDecoderLayer`, `MistralDecoderLayer`,
  `BertLayer`, `GPTJBlock`, `T5Block` ... (useful only when using FSDP)
- dataset_text_field: Default is None. Training dataset text field containing
  single sequence. Either the dataset_text_field or data_formatter_template need
  to be supplied. For running vision language model tuning pass the column name
  for text data.
- dataset_image_field: Default is None. For running vision language model tuning
  pass the column name of the image data in the dataset.
- remove_unused_columns: Default is True. Remove columns not required by the
  model when using an nlp.Dataset.
- dataset_kwargs_skip_prepare_dataset: Default is False. When True, configures
  trl to skip preparing the dataset

Hardcoded:

Sets the `--target_modules` layer names based on the `model_name`:

- `llama3.2-1b`: `["q_proj", "v_proj"]`
- `llama3.2-3b`: `["q_proj", "v_proj"]`
- `smollm2-135m`: `["q_proj", "v_proj"]`
- `granite-3.0-1b-a400m-base`: `["q_proj", "v_proj"]`
- `granite-3.1-3b-a800m-instruct`: `["q_proj", "v_proj"]`
- `granite-vision-3.2-2b`: `["q_proj", "v_proj"]`
- `granite-3b-code-base-128k`: `["q_proj", "v_proj"]`
- `granite-7b-base`: `["q_proj", "v_proj"]`
- `granite-8b-code-base-128k`: `["q_proj", "v_proj"]`
- `granite-8b-code-base`: `["q_proj", "v_proj"]`
- `granite-8b-japanese`: `["q_proj", "v_proj"]`
- `granite-13b-v2`: `["c_attn", "c_proj"]`
- `granite-20b-v2`: `["c_attn", "c_proj"]`
- `granite-34b-code-base`: `["c_attn", "c_proj"]`
- `llama-7b`: `["q_proj", "k_proj"]`
- `llama-13b`: `["q_proj", "k_proj"]`
- `llama2-70b`: `["q_proj", "v_proj"]`
- `llama3-8b`: `["q_proj", "k_proj"]`
- `llama3-70b`: `["q_proj", "v_proj"]`
- `llama3.1-8b`: `["q_proj", "v_proj"]`
- `llama3.1-70b`: `["q_proj", "v_proj"]`
- `llama3.1-405b`: `["q_proj", "v_proj"]`
- `granite-4.0-micro`: `["q_proj", "v_proj"]`
- `granite-4.0-h-1b`: `["q_proj", "v_proj"]`
- `granite-4.0-350m`: `["q_proj", "v_proj"]`
- `granite-4.0-h-small`: `["q_proj", "v_proj"]`
- `granite-4.0-h-micro`: `["q_proj", "v_proj"]`
- `granite-4.0-h-tiny`: `["q_proj", "v_proj"]`
- `allam-1-13b`: `["q_proj", "v_proj"]`
- `hf-tiny-model-private/tiny-random-BloomForCausalLM`:
  `["dense_h_to_4h", "dense_4h_to_4h"]`
- `mistral-7b-v0.1`: `["q_proj", "v_proj"]`
- `mistral-123b-v2`: `["q_proj", "v_proj"]`
- `mixtral-8x7b-instruct-v0.1`: `["q_proj", "v_proj"]`
- `granite-3-8b`: `["q_proj", "v_proj"]`
- `granite-3.3-8b`: `["q_proj", "v_proj"]`
- `granite-3.1-2b`: `["q_proj", "v_proj"]`
- `granite-3.1-8b-instruct`: `["q_proj", "v_proj"]`
- `llava-v1.6-mistral-7b`: `["q_proj", "v_proj"]`

> **NOTE**: Because running `accelerate` with a single gpu is unsupported, when
> setting `number_gpus` to 1 this experiment actually runs the
> `tuning.sft_trainer` script directly (i.e. a DataParallel (DP) run).

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Measured properties

We use AIM to collect profiling metadata. Then we convert the timeseries that
AIM collects into the metrics below.

- gpu_compute_utilization_min (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_compute_utilization_avg (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_compute_utilization_max (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_memory_utilization_min (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_memory_utilization_avg (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_memory_utilization_max (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_power_percent_min (0.0 if not using any GPUs): Measured in Percentages (0
  to 100) (see note 1)
- gpu_power_percent_avg (0.0 if not using any GPUs): Measured in Percentages (0
  to 100) (see note 1)
- gpu_power_percent_max (0.0 if not using any GPUs): Measured in Percentages (0
  to 100) (see note 1)
- gpu_power_watts_min (0.0 if not using any GPUs): Measured in Watts (see
  Note 1)
- gpu_power_watts_avg (0.0 if not using any GPUs): Measured in Watts (see
  Note 1)
- gpu_power_watts_max (0.0 if not using any GPUs): Measured in Watts (see
  Note 1)
- gpu_memory_utilization_peak (0.0 if not using any GPUs): peak GPU memory
  utilization percentage across all devices
- cpu_compute_utilization: Measured in Percentages (0 to 100 where 100 means 1
  full core) (see note 2)
- cpu_memory_utilization: Measured in Percentages (0 to 100) taken from AIM (see
  note 2)
- train_runtime: Measured in seconds
- train_samples_per_second
- train_steps_per_second
- train_tokens_per_second: How many tokens (including padding tokens) the run
  processed every second (for FSDP this is estimated from num_gpus \*
  rank_0_train_tokens_per_second). Omitted when stop_after_seconds is greater
  than 0
- train_tokens_per_gpu_per_second (will be equal to train_tokens_per_second when
  number_gpus <= 1, for FSDP this is reported using just rank 0). Omitted when
  stop_after_seconds is greater than 0
- dataset_tokens_per_second: How many tokens from the dataset the run processed
  every second (see note 3)
- dataset_tokens_per_second_per_gpu: How many tokens from the dataset the run
  processed every second per GPU (see note 3)
- is_valid (see `is_valid logic`)

Notes:

- (1) They are reported as the min/max/avg of the average in the timeseries
  metrics that AIM collects for the in-use GPUs of a run.
- (2) CPU compute and memory utilization are percentages. They are reported as
  the min/max/avg of the metrics that AIM collects for the sft_trainer.py
  process. The total memory capacity of the nodes varies from 100 GB to 400 GB.
  Currently, we do not store this information in our database.
- (3) `dataset_tokens_per_second` and `dataset_tokens_per_second_per_gpu` take
  into account the `tokenizer.model_max_length` and `max_seq_length` (i.e. for
  each entry, we report
  `min(len(tokens(entry["output"])), tokenizer.model_max_length, max_seq_length))`.

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### is_valid logic

A run for an entity is invalid if:

1. `batch_size` cannot be evenly divided by `number_gpus` (i.e.
   `batch_size % number_gpus != 0`)
2. `number_gpus` cannot be evenly divided by `number_nodes` (i.e.
   `number_gpus % number_nodes != 0`)
3. `number_nodes` must be greater than 0
4. `batch_size` must be greater than 0
5. if `number_gpus` is greater than 0 then `gpu_model` must be a non-empty
   string
6. if `fast_moe` is set and `number_gpus` is not divisible by it
7. if `fast_moe` is set and the `num_local_experts` of the Mixture of Experts
   (MoE) model is not divisible by `fast_moe` (which is interpreted as
   `ep_degrees` by fms-hf-tuning)

Runs raising the following errors are considered invalid due to running out of
GPU memory:

- `torch.cuda.OutOfMemoryError`
- `RuntimeError: CUDA error: an illegal memory access was encountered`

Measurements raising any other exception (including for example `RuntimeError`
containing the string `NCCL Error`) are considered to have `Failed`. They will
not contain the `is_valid` measured property, or any other property for that
matter. `Failed` measurements do not record any properties and can be repeated.

## GPTQ-LORA Fine-Tuning Experiments

### finetune_gtpq-lora_benchmark-v1.0.0

An experiment instance:

- performs LORA fine tuning
- the training data is artificial
- `use_flash_attn` is set to True
- `packing` is set to False
- `torch_dtype` is set to `float16`, cannot be a different value
- uses the `FSDP` distributed backend for multi-gpu runs by default, can also be
  `DDP`
- multi-gpu runs with FSDP and DDP backends use 1 process per GPU (via
  `accelerate`)
- runs 1 epoch by default, can also run a custom number of steps
- does not save checkpoint
- loads weights from a PVC
- request 2 CPU cores per GPU device (with a minimum of 2 cores)
- uses fms-acceleration plugins to perform GPTQ LoRA. Specifically:
  - `auto_gptq` is set to `triton_v2`
  - `fast_kernels` is set to `True True True`
  - `fused_lora` is set to `auto_gptq True`
  - `torch_dtype` is set to `float16`
  - loads GPTQ compatible pre-quantized weights from a PVC

For FSDP runs we use the following `accelerate_config.yml` YAML file:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: "no"
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: ${fsdp_sharding_strategy}
  fsdp_state_dict_type: ${fsdp_state_dict_type}
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: ${accelerate_config_fsdp_transformer_layer_cls_to_wrap}
machine_rank: { $THE MACHINE RANK - always 0 for single-node runs }
main_training_function: main
mixed_precision: ${accelerate_config_mixed_precision}
num_machines: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: { $SOME_PORT }
num_processes: { $NUM_GPUS }
```

For DDP runs we use this instead:

```yaml
compute_environment: LOCAL_MACHINE
debug: False
downcast_bf16: no
distributed_type: MULTI_GPU
machine_rank: { $THE MACHINE RANK - always 0 for single-node runs }
main_training_function: main
mixed_precision: ${accelerate_config_mixed_precision}
num_machines: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: { $SOME_PORT }
num_processes: { $NUM_GPUS }
```

Commandline:

<!-- markdownlint-disable line-length -->
```commandline
accelerate launch --config_file ${PATH_ACCELERATE_CONFIG} --num_processes ${NUMBER_GPUS} \
  ${PATH_TO_OUR_WRAPPER_OF_FMS_HF_TUNING_SFT_TRAINER} --model_name_or_path ${MODEL} \
  --torch_dtype float16 --use_flash_attn True --training_data_path ${DATASET_PATH} \
  --response_template "\n### Response:" --dataset_text_field output --log_level debug \
  --num_train_epochs 1 --per_device_train_batch_size ${BATCH_SIZE/NUM_GPUS} \
  --max_seq_length ${MODEL_MAX_LENGTH} --eval_strategy no --output_dir ${RANDOM_DIR} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --save_strategy no \
  --learning_rate 1e-05 --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --logging_steps 1 --include_tokens_per_second True --gradient_checkpointing True \
  --packing False --peft_method lora --target_modules ${SPACE SEPARATED LAYER NAMES} \
  --fp16 true --fast_kernels true true true --fused_lora auto_gptq true --auto_gptq triton_v2 \
  --optim ${OPTIM} --bf16 ${BF16} \
  --gradient_checkpointing_kwargs='{"use_reentrant": ${GRADIENT_CHECKPOINTING_USE_REENTRANT}}' \
  --fast_moe ${FAST_MOE}
```
<!-- markdownlint-enable line-length -->

**Note**: `--fast_moe` is only supported for fms-hf-tuning v2.4.0+

We use a thin wrapper of `sft_trainer.py` which injects a custom Callback that
exports the metrics collected by AIM. You can repeat our experiments by just
pointing the above command-line to `sft_trainer.py` from the `fms-hf-tuning`
package.

Versioning:

- Actuator version: `2.1.0`
- fms-hf-tuning versions:
  - 3.1.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.1.0_9aca2139f4244f500cf2f5b1a0fe2ef3f8251a82.txt](packages/fms-hf-tuning_v3.1.0_9aca2139f4244f500cf2f5b1a0fe2ef3f8251a82.txt)
  - 3.0.0.1 (this is a phony release)
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.0.0.1_51875160343064a1056e0105b7971ed8d9f26854.txt](packages/fms-hf-tuning_v3.0.0.1_51875160343064a1056e0105b7971ed8d9f26854.txt)
  - 3.0.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.0.0_d8cb1cbacfbab7ed23e91151f59516766ab339e2.txt](packages/fms-hf-tuning_v3.0.0_d8cb1cbacfbab7ed23e91151f59516766ab339e2.txt)
  - 2.8.2
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.8.2_ad594c7270e934679d48286aa87c5ade7bfc54e2.txt](packages/fms-hf-tuning_v2.8.2_ad594c7270e934679d48286aa87c5ade7bfc54e2.txt)
  - 2.7.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.7.1_456fe2a840e7f0b2d8d48c04cefe24faddbf261a.txt](packages/fms-hf-tuning_v2.7.1_456fe2a840e7f0b2d8d48c04cefe24faddbf261a.txt)
  - 2.6.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.6.0_53f2babaddf07c85f5274167af9aaa947f19faf3.txt](packages/fms-hf-tuning_v2.6.0_53f2babaddf07c85f5274167af9aaa947f19faf3.txt)
  - 2.5.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.5.0_6f9bab223987732826f625fc7a522a78b58697fb.txt](packages/fms-hf-tuning_v2.5.0_6f9bab223987732826f625fc7a522a78b58697fb.txt)
  - 2.4.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.4.0_76bd76d0cfef0852e8490c344b791a35a1080ead.txt](packages/fms-hf-tuning_v2.4.0_76bd76d0cfef0852e8490c344b791a35a1080ead.txt)
  - 2.3.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.3.1_3ec30a0f9c47b0b6b9f43ce9200ab4ff24ed01e8.txt](packages/fms-hf-tuning_v2.3.1_3ec30a0f9c47b0b6b9f43ce9200ab4ff24ed01e8.txt)
  - 2.2.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.2.1_e6f7a2205c06e703c6b22bdcc5e1f248823c2a2e.txt](packages/fms-hf-tuning_v2.2.1_e6f7a2205c06e703c6b22bdcc5e1f248823c2a2e.txt)
  - 2.1.2 (default)
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.2_1e82e020f64d5a53acf98eecccb33c3597881b5e.txt](packages/fms-hf-tuning_v2.1.2_1e82e020f64d5a53acf98eecccb33c3597881b5e.txt)
  - 2.1.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.1_e2ac09183d8ba29084e110fd16b6b6c872e4a267.txt](packages/fms-hf-tuning_v2.1.1_e2ac09183d8ba29084e110fd16b6b6c872e4a267.txt)
  - 2.1.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.0_8f168183a70b41cb66902f438ecba7734144138c.txt](packages/fms-hf-tuning_v2.1.0_8f168183a70b41cb66902f438ecba7734144138c.txt)
  - 2.0.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.0.1_9b8245e74144f7ee73b7241a1687b6c77f0eb2e4.txt](packages/fms-hf-tuning_v2.0.1_9b8245e74144f7ee73b7241a1687b6c77f0eb2e4.txt)

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Requirements

- The S3 bucket `watson.runtime.wisdom.model.us-south` mounted under
  `/ibm-research-models`
  ([instructions](../../../../examples/fms-hf-tuning/README.md)).
- The PVC `hf-models-pvc` mounted under `/hf-models-pvc` - should contain the
  models:
  - LLaMa/models/hf/7B-gptq/
  - LLaMa/models/hf/llama3-70b-gptq/
  - LLaMa/models/hf/llama3.1-405b-gptq/
  - granite-20b-code-base-v2/step_280000_ckpt-gptq/
  - granite-34b-gptq/
  - granite-7b-base-gtpq/
  - granite-8b-code-instruct-gptq/
  - mistral-7B-v0.3-gptq/
  - mixtral_8x7b_instruct_v0.1_gptq/
- The PVC `ray-disorch-storage` mounted under `/data` with the preprocessed
  `artificial-dataset` files
  (<https://github.ibm.com/ai-foundation/watson-fm-stack-tracker/issues/550>)
  under `/data/fms-hf-tuning/artificial-dataset`

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Entity space

Required:

- model_name: Supported models:
  <!-- markdownlint-disable-next-line line-length -->
  `["llama-7b", "granite-20b-v2", "granite-7b-base", "granite-8b-code-instruct", "granite-34b-code-base", "mistral-7b-v0.1", "llama3-70b", "mixtral-8x7b-instruct-v0.1", "llama3.1-405b"]`
- model_max_length: Maximum sequence length. Sequences will be right padded (and
  possibly truncated)
- number_gpus: The effective number of GPUs (to be evenly distributed to
  `number_nodes` machines)
- batch_size: the effective batch_size (will be evenly distributed to max(1,
  number_gpus) devices)
- gpu_model: The value of the kubernetes node label `nvidia.com/gpu.prod` for
  example
  - `NVIDIA-A100-80GB-PCIe`
  - `NVIDIA-A100-SXM4-80GB`
  - `NVIDIA-H100-PCIe`

Optional:

- dataset_id: Default is `news-tokens-16384plus-entries-4096`. Available options
  are:
  - `news-chars-512-entries-4096`: 4096 entries with samples of 512 + 127
    (prompt) + 512 characters
  - `news-chars-1024-entries-4096`: 4096 entries with samples of 1024 + 127
    (prompt) + 1024 characters
  - `news-chars-2048-entries-4096`: 4096 entries with samples of 2048 + 127
    (prompt) + 2048 characters
  - `news-tokens-16384plus-entries-4096`: 4096 entries, each entry has least
    16384 tokens when tokenized with any of the granite-13b-v2, llama-13b-v2,
    llama-7b, or granite-20b-v2 tokenizers
  - `vision-384x384-16384plus-entries-4096`: A vision dataset containing 4096
    entries. Each entry includes at least 16384 tokens when tokenized with
    `granite-vision-3.2-2b`, and consists of repeated copies of a single image
    with dimensions 384×384.
  - `vision-384x768-16384plus-entries-4096`: Similar to the above, this dataset
    also contains 4096 entries with a minimum of 16384 tokens per entry
    (tokenized using `granite-vision-3.2-2b`). Each entry uses repeated copies
    of a single image sized 384×768.
- gradient_checkpointing: Default is `True`. If `True`, use gradient
  checkpointing to save memory (i.e. higher batchsizes) at the expense of slower
  backward pass
- gradient_accumulation_steps: Default is 4. Number of update steps to
  accumulate before performing a backward/update pass. Only takes effect when
  gradient_checkpointing is True
- torch_dtype: Default is `float16`. One of `float16`
- max_steps: Default is `-1`. The number of optimization steps to perform. Set
  to -1 to respect num_train_epochs instead.
- num_train_epochs: Default is `1.0`. How many epochs to run. Ignored if
  max_steps is greater than 0.
- stop_after_seconds: Default is `-1.0`. If set, the optimizer will be asked to
  stop after the specified time elapses. The check is performed after the end of
  each training step.
- auto_stop_method: The default value is `None`. This parameter defines the
  method used to automatically stop the fine-tuning job. Supported values are
  `WARMUP_60S_STABLE_120S_OR_10_STEPS` and `None`. If set to
  `WARMUP_60S_STABLE_120S_OR_10_STEPS`, the job stops after spending at least 60
  seconds in the warmup phase plus the longer of 120 seconds or the duration of
  10 optimization steps. This method excludes the first 60 seconds of training
  when calculating throughput and system metrics.
- distributed_backend: Default is `FSDP` for multi-gpu measurements, `None`
  (i.e. Data Parallel (DP)) for single-gpu measurements. Which pytorch backend
  to use when training with multiple GPU devices.
- number_nodes: Default is `1`. If set, actuator distributes tasks on multiple
  nodes. Each Node will use number_gpus/number_nodes GPUs. Each Node will use 1
  process for each GPU it uses
- fms_hf_tuning_version: Default is `2.1.2`. Which version of fms-hf-tuning to
  use. Available options are: `3.1.0`, `3.0.0.1`, `3.0.0`, 2.8.2`,`2.7.1`,
  `2.6.0`,`2.5.0`,`2.4.0`,`2.3.1`,`2.2.1`,`2.1.2`,`2.1.0`,`2.0.1`
- enable_roce: Default is `False`. This setting is only in effect for multi-node
  runs. It controls whether RDMA over Converged Ethernet (RoCE) is switched on
  or not.
- fast_moe: Default is `0`. Configures the amount of expert parallel sharding.
  number_gpus must be divisible by it
- fast_kernels: Default is `None`. Switches on fast kernels, the value is a list
  with strings of boolean values for
  `[fast_loss, fast_rms_layernorm, fast_rope_embeddings]`
- r: Default is `4`. The LORA rank
- lora_alpha: Default is `16`. Scales the learning weights.
- optim: Default is `adamw_torch`. The optimizer to use. Available options are
  `adamw_hf`, `adamw_torch`, `adamw_torch_fused`, `adamw_torch_xla`,
  `adamw_torch_npu_fused`, `adamw_apex_fused`, `adafactor`,
  `adamw_anyprecision`, `adamw_torch_4bit`, `ademamix`, `sgd`, `adagrad`,
  `adamw_bnb_8bit`, `adamw_8bit`, `ademamix_8bit`, `lion_8bit`, `lion_32bit`,
  `paged_adamw_32bit`, `paged_adamw_8bit`, `paged_ademamix_32bit`,
  `paged_ademamix_8bit`, `paged_lion_32bit`, `paged_lion_8bit`, `rmsprop`,
  `rmsprop_bnb`, `rmsprop_bnb_8bit`, `rmsprop_bnb_32bit`, `galore_adamw`,
  `galore_adamw_8bit`, `galore_adafactor`, `galore_adamw_layerwise`,
  `galore_adamw_8bit_layerwise`, `galore_adafactor_layerwise`, `lomo`,
  `adalomo`, `grokadamw`, `schedule_free_adamw`, `schedule_free_sgd`
- bf16: Default is `False`. Whether to use bf16 (mixed) precision instead of
  32-bit. Requires Ampere or higher NVIDIA add bf16 mixed precision support for
  NPU architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental
  API and it may change. Can be `True`, `False`.
- gradient_checkpointing_use_reentrant: Default is `False` Specify whether to
  use the activation checkpoint variant that requires reentrant autograd. This
  parameter should be passed explicitly. Torch version 2.5 will raise an
  exception if use_reentrant is not passed. If use_reentrant=False, checkpoint
  will use an implementation that does not require reentrant autograd. This
  allows checkpoint to support additional functionality, such as working as
  expected with torch.autograd.grad and support for keyword arguments input into
  the checkpointed function. Can be `True`, `False`.
- fsdp_sharding_strategy: Default is `FULL_SHARD`. [1] FULL_SHARD (shards
  optimizer states, gradients and parameters), " [2] SHARD_GRAD_OP (shards
  optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD
  (shards optimizer states, gradients and parameters within each node while
  each node has full copy - equivalent to FULL_SHARD for single-node runs),
  [5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients
  within each node while each node has full copy). For more information, please
  refer the official PyTorch docs.
- fsdp_state_dict_type: Default is `FULL_STATE_DICT`. [1] FULL_STATE_DICT, [2]
  LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT
- fsdp_use_orig_params: Default is `True`. If True, allows non-uniform
  `requires_grad` during init, which means support for interspersed frozen and
  trainable parameters. (useful only when `use_fsdp` flag is passed).
- accelerate_config_mixed_precision: Default is `no`. Whether to use mixed
  precision training or not. Choose from `no`,`fp16`,`bf16` or `fp8`. `fp8`
  requires the installation of transformers-engine.
- accelerate_config_fsdp_transformer_layer_cls_to_wrap: Default is None. List of
  transformer layer class names (case-sensitive) to wrap, e.g,
  `GraniteDecoderLayer`, `LlamaDecoderLayer`, `MistralDecoderLayer`,
  `BertLayer`, `GPTJBlock`, `T5Block` ... (useful only when using FSDP)
- dataset_text_field: Default is None. Training dataset text field containing
  single sequence. Either the dataset_text_field or data_formatter_template need
  to be supplied. For running vision language model tuning pass the column name
  for text data.
- dataset_image_field: Default is None. For running vision language model tuning
  pass the column name of the image data in the dataset.
- remove_unused_columns: Default is True. Remove columns not required by the
  model when using an nlp.Dataset.
- dataset_kwargs_skip_prepare_dataset: Default is False. When True, configures
  trl to skip preparing the dataset

Hardcoded:

Sets the `--target_modules` layer names based on the `model_name`:

- `granite-8b-code-instruct`: `["q_proj", "v_proj"]`
- `granite-7b-base`: `["q_proj", "v_proj"]`
- `granite-20b-v2`: `["c_attn", "c_proj"]`
- `granite-34b-code-base`: `["c_attn", "c_proj"]`
- `llama-7b`: `["q_proj", "k_proj"]`
- `llama3-70b`: `["q_proj", "v_proj"]`
- `mistral-7b-v0.1`: `["q_proj", "v_proj"]`
- `mixtral-8x7b-instruct-v0.1`: `["q_proj", "v_proj"]`
- `llama3.1-405b`: `["q_proj", "v_proj"]`
- `allam-1-13b`: `["q_proj", "v_proj"]`

> **NOTE**: Because running `accelerate` with a single gpu is unsupported, when
> setting `number_gpus` to 1 this experiment actually runs the
> `tuning.sft_trainer` script directly (i.e. a DataParallel (DP) run).

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Measured properties

We use AIM to collect profiling metadata. Then we convert the timeseries that
AIM collects into the metrics below.

- gpu_compute_utilization_min (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_compute_utilization_avg (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_compute_utilization_max (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_memory_utilization_min (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_memory_utilization_avg (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_memory_utilization_max (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_power_percent_min (0.0 if not using any GPUs): Measured in Percentages (0
  to 100) (see note 1)
- gpu_power_percent_avg (0.0 if not using any GPUs): Measured in Percentages (0
  to 100) (see note 1)
- gpu_power_percent_max (0.0 if not using any GPUs): Measured in Percentages (0
  to 100) (see note 1)
- gpu_power_watts_min (0.0 if not using any GPUs): Measured in Watts (see
  Note 1)
- gpu_power_watts_avg (0.0 if not using any GPUs): Measured in Watts (see
  Note 1)
- gpu_power_watts_max (0.0 if not using any GPUs): Measured in Watts (see
  Note 1)
- gpu_memory_utilization_peak (0.0 if not using any GPUs): peak GPU memory
  utilization percentage across all devices
- cpu_compute_utilization: Measured in Percentages (0 to 100 where 100 means 1
  full core) (see note 2)
- cpu_memory_utilization: Measured in Percentages (0 to 100) taken from AIM (see
  note 2)
- train_runtime: Measured in seconds
- train_samples_per_second
- train_steps_per_second
- train_tokens_per_second: How many tokens (including padding tokens) the run
  processed every second (for FSDP this is estimated from num_gpus \*
  rank_0_train_tokens_per_second). Omitted when stop_after_seconds is greater
  than 0
- train_tokens_per_gpu_per_second (will be equal to train_tokens_per_second when
  number_gpus <= 1, for FSDP this is reported using just rank 0). Omitted when
  stop_after_seconds is greater than 0
- dataset_tokens_per_second: How many tokens from the dataset the run processed
  every second (see note 3)
- dataset_tokens_per_second_per_gpu: How many tokens from the dataset the run
  processed every second per GPU (see note 3)
- is_valid (see `is_valid logic`)

Notes:

- (1) They are reported as the min/max/avg of the average in the timeseries
  metrics that AIM collects for the in-use GPUs of a run.
- (2) CPU compute and memory utilization are percentages. They are reported as
  the min/max/avg of the metrics that AIM collects for the sft_trainer.py
  process. The total memory capacity of the nodes varies from 100 GB to 400 GB.
  Currently, we do not store this information in our database.
- (3) `dataset_tokens_per_second` and `dataset_tokens_per_second_per_gpu` take
  into account the `tokenizer.model_max_length` and `max_seq_length` (i.e. for
  each entry, we report
  `min(len(tokens(entry["output"])), tokenizer.model_max_length, max_seq_length))`.

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### is_valid logic

A run for an entity is invalid if:

1. `batch_size` cannot be evenly divided by `number_gpus` (i.e.
   `batch_size % number_gpus != 0`)
2. `number_gpus` cannot be evenly divided by `number_nodes` (i.e.
   `number_gpus % number_nodes != 0`)
3. `number_nodes` must be greater than 0
4. `batch_size` must be greater than 0
5. if `number_gpus` is greater than 0 then `gpu_model` must be a non-empty
   string
6. if `fast_moe` is set and `number_gpus` is not divisible by it
7. if `fast_moe` is set and the `num_local_experts` of the Mixture of Experts
   (MoE) model is not divisible by `fast_moe` (which is interpreted as
   `ep_degrees` by fms-hf-tuning)

Runs raising the following errors are considered invalid due to running out of
GPU memory:

- `torch.cuda.OutOfMemoryError`
- `RuntimeError: CUDA error: an illegal memory access was encountered`

Measurements raising any other exception (including for example `RuntimeError`
containing the string `NCCL Error`) are considered to have `Failed`. They will
not contain the `is_valid` measured property, or any other property for that
matter. `Failed` measurements do not record any properties and can be repeated.

## PT Fine-Tuning Experiments

### finetune_pt_benchmark-v1.0.0

An experiment instance:

- performs prompt-tuning fine tuning
- the training data is artificial
- `use_flash_attn` is set to True
- `packing` is set to False
- `torch_dtype` is set to `bfloat16` by default, can also be float16
- uses the `FSDP` distributed backend for multi-gpu runs by default, can also be
  `DDP`
- multi-gpu runs with FSDP and DDP backends use 1 process per GPU (via
  `accelerate`)
- runs 1 epoch by default, can also run a custom number of steps
- does not save checkpoint
- loads weights from a PVC
- request 2 CPU cores per GPU device (with a minimum of 2 cores)

For FSDP runs we use the following `accelerate_config.yml` YAML file:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: "no"
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: ${fsdp_sharding_strategy}
  fsdp_state_dict_type: ${fsdp_state_dict_type}
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: ${accelerate_config_fsdp_transformer_layer_cls_to_wrap}
machine_rank: { $THE MACHINE RANK - always 0 for single-node runs }
main_training_function: main
mixed_precision: ${accelerate_config_mixed_precision}
num_machines: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: { $SOME_PORT }
num_processes: { $NUM_GPUS }
```

For DDP runs we use this instead:

```yaml
compute_environment: LOCAL_MACHINE
debug: False
downcast_bf16: no
distributed_type: MULTI_GPU
machine_rank: { $THE MACHINE RANK - always 0 for single-node runs }
main_training_function: main
mixed_precision: ${accelerate_config_mixed_precision}
num_machines: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
main_process_port: { $SOME_PORT }
num_processes: { $NUM_GPUS }
```

Commandline:

<!-- markdownlint-disable line-length -->
```commandline
accelerate launch --config_file ${PATH_ACCELERATE_CONFIG} --num_processes ${NUMBER_GPUS} \
  ${PATH_TO_OUR_WRAPPER_OF_FMS_HF_TUNING_SFT_TRAINER} --model_name_or_path ${MODEL} \
  --torch_dtype bfloat16 --use_flash_attn True --training_data_path ${DATASET_PATH} \
  --response_template "\n### Response:" --dataset_text_field output --log_level debug \
  --num_train_epochs 1 --per_device_train_batch_size ${BATCH_SIZE/NUM_GPUS} \
  --max_seq_length ${MODEL_MAX_LENGTH} --eval_strategy no --output_dir ${RANDOM_DIR} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --save_strategy no \
  --learning_rate 1e-05 --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --logging_steps 1 --include_tokens_per_second True --gradient_checkpointing True \
  --packing False --peft_method none \
  --fast_moe ${FAST_MOE}
```
<!-- markdownlint-enable line-length -->

**Note**: `--fast_moe` is only supported for fms-hf-tuning v2.4.0+

We use a thin wrapper of `sft_trainer.py` which injects a custom Callback that
exports the metrics collected by AIM. You can repeat our experiments by just
pointing the above command-line to `sft_trainer.py` from the `fms-hf-tuning`
package.

Versioning:

- Actuator version: `2.1.0`
- fms-hf-tuning versions:
  - 3.1.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.1.0_9aca2139f4244f500cf2f5b1a0fe2ef3f8251a82.txt](packages/fms-hf-tuning_v3.1.0_9aca2139f4244f500cf2f5b1a0fe2ef3f8251a82.txt)
  - 3.0.0.1 (this is a phony release)
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.0.0.1_51875160343064a1056e0105b7971ed8d9f26854.txt](packages/fms-hf-tuning_v3.0.0.1_51875160343064a1056e0105b7971ed8d9f26854.txt)
  - 3.0.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v3.0.0_d8cb1cbacfbab7ed23e91151f59516766ab339e2.txt](packages/fms-hf-tuning_v3.0.0_d8cb1cbacfbab7ed23e91151f59516766ab339e2.txt)
  - 2.8.2
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.8.2_ad594c7270e934679d48286aa87c5ade7bfc54e2.txt](packages/fms-hf-tuning_v2.8.2_ad594c7270e934679d48286aa87c5ade7bfc54e2.txt)
  - 2.7.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.7.1_456fe2a840e7f0b2d8d48c04cefe24faddbf261a.txt](packages/fms-hf-tuning_v2.7.1_456fe2a840e7f0b2d8d48c04cefe24faddbf261a.txt)
  - 2.6.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.6.0_53f2babaddf07c85f5274167af9aaa947f19faf3.txt](packages/fms-hf-tuning_v2.6.0_53f2babaddf07c85f5274167af9aaa947f19faf3.txt)
  - 2.5.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.5.0_6f9bab223987732826f625fc7a522a78b58697fb.txt](packages/fms-hf-tuning_v2.5.0_6f9bab223987732826f625fc7a522a78b58697fb.txt)
  - 2.4.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.4.0_76bd76d0cfef0852e8490c344b791a35a1080ead.txt](packages/fms-hf-tuning_v2.4.0_76bd76d0cfef0852e8490c344b791a35a1080ead.txt)
  - 2.3.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.3.1_3ec30a0f9c47b0b6b9f43ce9200ab4ff24ed01e8.txt](packages/fms-hf-tuning_v2.3.1_3ec30a0f9c47b0b6b9f43ce9200ab4ff24ed01e8.txt)
  - 2.2.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.2.1_e6f7a2205c06e703c6b22bdcc5e1f248823c2a2e.txt](packages/fms-hf-tuning_v2.2.1_e6f7a2205c06e703c6b22bdcc5e1f248823c2a2e.txt)
  - 2.1.2 (default)
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.2_1e82e020f64d5a53acf98eecccb33c3597881b5e.txt](packages/fms-hf-tuning_v2.1.2_1e82e020f64d5a53acf98eecccb33c3597881b5e.txt)
  - 2.1.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.1_e2ac09183d8ba29084e110fd16b6b6c872e4a267.txt](packages/fms-hf-tuning_v2.1.1_e2ac09183d8ba29084e110fd16b6b6c872e4a267.txt)
  - 2.1.0
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.1.0_8f168183a70b41cb66902f438ecba7734144138c.txt](packages/fms-hf-tuning_v2.1.0_8f168183a70b41cb66902f438ecba7734144138c.txt)
  - 2.0.1
    - The full list of packages is at
      [packages/fms-hf-tuning_v2.0.1_9b8245e74144f7ee73b7241a1687b6c77f0eb2e4.txt](packages/fms-hf-tuning_v2.0.1_9b8245e74144f7ee73b7241a1687b6c77f0eb2e4.txt)

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Requirements

- The S3 bucket `watson.runtime.wisdom.model.us-south` mounted under
  `/ibm-research-models`
  ([instructions](../../../../examples/fms-hf-tuning/README.md)).
- The PVC `hf-models-pvc` mounted under `/hf-models-pvc` - should contain the
  models:
  - LLaMa/models/hf/13B/
  - LLaMa/models/hf/7B/
  - LLaMa/models/hf/llama2-70b/
  - LLaMa/models/hf/llama3-70b/
  - LLaMa/models/hf/llama3-8b/
  - LLaMa/models/hf/llama3.1-405b/
  - LLaMa/models/hf/llama3.1-70b/
  - LLaMa/models/hf/llama3.1-8b/
  - Mixtral-8x7B-Instruct-v0.1/
  - allam-1-13b-instruct-20240607/
  - granite-13b-base-v2/step_300000_ckpt/
  - granite-20b-code-base-v2/step_280000_ckpt/
  - granite-34b-code-base/
  - granite-8b-code-base/
  - granite-8b-japanese-base-v1-llama/
  - mistralai-mistral-7b-v0.1/
  - mistral-large/fp16_240620
- The PVC `ray-disorch-storage` mounted under `/data` with the preprocessed
  `artificial-dataset` files
  (<https://github.ibm.com/ai-foundation/watson-fm-stack-tracker/issues/550>)
  under `/data/fms-hf-tuning/artificial-dataset`

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Entity space

Required:

- model_name: Supported models:
  <!-- markdownlint-disable-next-line line-length -->
  `["granite-3b-1.5", "hf-tiny-model-private/tiny-random-BloomForCausalLM", "llama-7b", "granite-13b-v2", "llama-13b", "granite-20b-v2", "granite-7b-base", "granite-8b-japanese", "granite-8b-code-base", "granite-34b-code-base", "mistral-7b-v0.1", "llama3-8b", "llama3-70b", "mixtral-8x7b-instruct-v0.1", "llama2-70b", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "granite-3b-code-base-128k", "granite-8b-code-base-128k", "allam-1-13b", "granite-3-8b", "granite-3.1-2b", "granite-3.1-8b-instruct", "mistral-123b-v2", "granite-3.1-3b-a800m-instruct", "granite-vision-3.2-2b", "smollm2-135m", "llava-v1.6-mistral-7b", "granite-4.0-micro", "granite-4.0-h-1b", "granite-4.0-350m", "granite-4.0-h-small", "granite-4.0-h-micro", "granite-4.0-h-tiny", "granite-3.3-8b"]`
- model_max_length: Maximum sequence length. Sequences will be right padded (and
  possibly truncated)
- number_gpus: The effective number of GPUs (to be evenly distributed to
  `number_nodes` machines)
- batch_size: the effective batch_size (will be evenly distributed to max(1,
  number_gpus) devices)
- gpu_model: The value of the kubernetes node label `nvidia.com/gpu.prod` for
  example
  - `NVIDIA-A100-80GB-PCIe`
  - `NVIDIA-A100-SXM4-80GB`
  - `NVIDIA-H100-PCIe`

Optional:

- dataset_id: Default is `news-tokens-16384plus-entries-4096`. Available options
  are:
  - `news-chars-512-entries-4096`: 4096 entries with samples of 512 + 127
    (prompt) + 512 characters
  - `news-chars-1024-entries-4096`: 4096 entries with samples of 1024 + 127
    (prompt) + 1024 characters
  - `news-chars-2048-entries-4096`: 4096 entries with samples of 2048 + 127
    (prompt) + 2048 characters
  - `news-tokens-16384plus-entries-4096`: 4096 entries, each entry has least
    16384 tokens when tokenized with any of the granite-13b-v2, llama-13b-v2,
    llama-7b, or granite-20b-v2 tokenizers
  - `vision-384x384-16384plus-entries-4096`: A vision dataset containing 4096
    entries. Each entry includes at least 16384 tokens when tokenized with
    `granite-vision-3.2-2b`, and consists of repeated copies of a single image
    with dimensions 384×384.
  - `vision-384x768-16384plus-entries-4096`: Similar to the above, this dataset
    also contains 4096 entries with a minimum of 16384 tokens per entry
    (tokenized using `granite-vision-3.2-2b`). Each entry uses repeated copies
    of a single image sized 384×768.
- gradient_checkpointing: Default is `True`. If `True`, use gradient
  checkpointing to save memory (i.e. higher batchsizes) at the expense of slower
  backward pass
- gradient_accumulation_steps: Default is 4. Number of update steps to
  accumulate before performing a backward/update pass. Only takes effect when
  gradient_checkpointing is True
- torch_dtype: Default is `bfloat16`. One of `bfloat16`, `float32`, `float16`
- max_steps: Default is `-1`. The number of optimization steps to perform. Set
  to -1 to respect num_train_epochs instead.
- num_train_epochs: Default is `1.0`. How many epochs to run. Ignored if
  max_steps is greater than 0.
- stop_after_seconds: Default is `-1.0`. If set, the optimizer will be asked to
  stop after the specified time elapses. The check is performed after the end of
  each training step.
- auto_stop_method: The default value is `None`. This parameter defines the
  method used to automatically stop the fine-tuning job. Supported values are
  `WARMUP_60S_STABLE_120S_OR_10_STEPS` and `None`. If set to
  `WARMUP_60S_STABLE_120S_OR_10_STEPS`, the job stops after spending at least 60
  seconds in the warmup phase plus the longer of 120 seconds or the duration of
  10 optimization steps. This method excludes the first 60 seconds of training
  when calculating throughput and system metrics.
- distributed_backend: Default is `FSDP` for multi-gpu measurements, `None`
  (i.e. Data Parallel (DP)) for single-gpu measurements. Which pytorch backend
  to use when training with multiple GPU devices.
- number_nodes: Default is `1`. If set, actuator distributes tasks on multiple
  nodes. Each Node will use number_gpus/number_nodes GPUs. Each Node will use 1
  process for each GPU it uses
- fms_hf_tuning_version: Default is `2.1.2`. Which version of fms-hf-tuning to
  use. Available options are: `3.1.0`, `3.0.0.1`, `3.0.0`, 2.8.2`,`2.7.1`,
  `2.6.0`,`2.5.0`,`2.4.0`,`2.3.1`,`2.2.1`,`2.1.2`,`2.1.0`,`2.0.1`
- enable_roce: Default is `False`. This setting is only in effect for multi-node
  runs. It controls whether RDMA over Converged Ethernet (RoCE) is switched on
  or not.
- fast_moe: Default is `0`. Configures the amount of expert parallel sharding.
  number_gpus must be divisible by it
- fast_kernels: Default is `None`. Switches on fast kernels, the value is a list
  with strings of boolean values for
  `[fast_loss, fast_rms_layernorm, fast_rope_embeddings]`
- optim: Default is `adamw_torch`. The optimizer to use. Available options are
  `adamw_hf`, `adamw_torch`, `adamw_torch_fused`, `adamw_torch_xla`,
  `adamw_torch_npu_fused`, `adamw_apex_fused`, `adafactor`,
  `adamw_anyprecision`, `adamw_torch_4bit`, `ademamix`, `sgd`, `adagrad`,
  `adamw_bnb_8bit`, `adamw_8bit`, `ademamix_8bit`, `lion_8bit`, `lion_32bit`,
  `paged_adamw_32bit`, `paged_adamw_8bit`, `paged_ademamix_32bit`,
  `paged_ademamix_8bit`, `paged_lion_32bit`, `paged_lion_8bit`, `rmsprop`,
  `rmsprop_bnb`, `rmsprop_bnb_8bit`, `rmsprop_bnb_32bit`, `galore_adamw`,
  `galore_adamw_8bit`, `galore_adafactor`, `galore_adamw_layerwise`,
  `galore_adamw_8bit_layerwise`, `galore_adafactor_layerwise`, `lomo`,
  `adalomo`, `grokadamw`, `schedule_free_adamw`, `schedule_free_sgd`
- bf16: Default is `False`. Whether to use bf16 (mixed) precision instead of
  32-bit. Requires Ampere or higher NVIDIA add bf16 mixed precision support for
  NPU architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental
  API and it may change. Can be `True`, `False`.
- gradient_checkpointing_use_reentrant: Default is `False` Specify whether to
  use the activation checkpoint variant that requires reentrant autograd. This
  parameter should be passed explicitly. Torch version 2.5 will raise an
  exception if use_reentrant is not passed. If use_reentrant=False, checkpoint
  will use an implementation that does not require reentrant autograd. This
  allows checkpoint to support additional functionality, such as working as
  expected with torch.autograd.grad and support for keyword arguments input into
  the checkpointed function. Can be `True`, `False`.
- fsdp_sharding_strategy: Default is `FULL_SHARD`. [1] FULL_SHARD (shards
  optimizer states, gradients and parameters), " [2] SHARD_GRAD_OP (shards
  optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD
  (shards optimizer states, gradients and parameters within each node while
  each node has full copy - equivalent to FULL_SHARD for single-node runs),
  [5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients
  within each node while each node has full copy). For more information, please
  refer the official PyTorch docs.
- fsdp_state_dict_type: Default is `FULL_STATE_DICT`. [1] FULL_STATE_DICT, [2]
  LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT
- fsdp_use_orig_params: Default is `True`. If True, allows non-uniform
  `requires_grad` during init, which means support for interspersed frozen and
  trainable parameters. (useful only when `use_fsdp` flag is passed).
- accelerate_config_mixed_precision: Default is `no`. Whether to use mixed
  precision training or not. Choose from `no`,`fp16`,`bf16` or `fp8`. `fp8`
  requires the installation of transformers-engine.
- accelerate_config_fsdp_transformer_layer_cls_to_wrap: Default is None. List of
  transformer layer class names (case-sensitive) to wrap, e.g,
  `GraniteDecoderLayer`, `LlamaDecoderLayer`, `MistralDecoderLayer`,
  `BertLayer`, `GPTJBlock`, `T5Block` ... (useful only when using FSDP)
- dataset_text_field: Default is None. Training dataset text field containing
  single sequence. Either the dataset_text_field or data_formatter_template need
  to be supplied. For running vision language model tuning pass the column name
  for text data.
- dataset_image_field: Default is None. For running vision language model tuning
  pass the column name of the image data in the dataset.
- remove_unused_columns: Default is True. Remove columns not required by the
  model when using an nlp.Dataset.
- dataset_kwargs_skip_prepare_dataset: Default is False. When True, configures
  trl to skip preparing the dataset

> **NOTE**: Because running `accelerate` with a single gpu is unsupported, when
> setting `number_gpus` to 1 this experiment actually runs the
> `tuning.sft_trainer` script directly (i.e. a DataParallel (DP) run).

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### Measured properties

We use AIM to collect profiling metadata. Then we convert the timeseries that
AIM collects into the metrics below.

- gpu_compute_utilization_min (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_compute_utilization_avg (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_compute_utilization_max (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_memory_utilization_min (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_memory_utilization_avg (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_memory_utilization_max (0.0 if not using any GPUs): Measured in
  Percentages (0 to 100) (see note 1)
- gpu_power_percent_min (0.0 if not using any GPUs): Measured in Percentages (0
  to 100) (see note 1)
- gpu_power_percent_avg (0.0 if not using any GPUs): Measured in Percentages (0
  to 100) (see note 1)
- gpu_power_percent_max (0.0 if not using any GPUs): Measured in Percentages (0
  to 100) (see note 1)
- gpu_power_watts_min (0.0 if not using any GPUs): Measured in Watts (see
  Note 1)
- gpu_power_watts_avg (0.0 if not using any GPUs): Measured in Watts (see
  Note 1)
- gpu_power_watts_max (0.0 if not using any GPUs): Measured in Watts (see
  Note 1)
- gpu_memory_utilization_peak (0.0 if not using any GPUs): peak GPU memory
  utilization percentage across all devices
- cpu_compute_utilization: Measured in Percentages (0 to 100 where 100 means 1
  full core) (see note 2)
- cpu_memory_utilization: Measured in Percentages (0 to 100) taken from AIM (see
  note 2)
- train_runtime: Measured in seconds
- train_samples_per_second
- train_steps_per_second
- train_tokens_per_second: How many tokens (including padding tokens) the run
  processed every second (for FSDP this is estimated from num_gpus \*
  rank_0_train_tokens_per_second). Omitted when stop_after_seconds is greater
  than 0
- train_tokens_per_gpu_per_second (will be equal to train_tokens_per_second when
  number_gpus <= 1, for FSDP this is reported using just rank 0). Omitted when
  stop_after_seconds is greater than 0
- dataset_tokens_per_second: How many tokens from the dataset the run processed
  every second (see note 3)
- dataset_tokens_per_second_per_gpu: How many tokens from the dataset the run
  processed every second per GPU (see note 3)
- is_valid (see `is_valid logic`)

Notes:

- (1) They are reported as the min/max/avg of the average in the timeseries
  metrics that AIM collects for the in-use GPUs of a run.
- (2) CPU compute and memory utilization are percentages. They are reported as
  the min/max/avg of the metrics that AIM collects for the sft_trainer.py
  process. The total memory capacity of the nodes varies from 100 GB to 400 GB.
  Currently, we do not store this information in our database.
- (3) `dataset_tokens_per_second` and `dataset_tokens_per_second_per_gpu` take
  into account the `tokenizer.model_max_length` and `max_seq_length` (i.e. for
  each entry, we report
  `min(len(tokens(entry["output"])), tokenizer.model_max_length, max_seq_length))`.

<!-- markdownlint-disable-next-line no-duplicate-heading -->
#### is_valid logic

A run for an entity is invalid if:

1. `batch_size` cannot be evenly divided by `number_gpus` (i.e.
   `batch_size % number_gpus != 0`)
2. `number_gpus` cannot be evenly divided by `number_nodes` (i.e.
   `number_gpus % number_nodes != 0`)
3. `number_nodes` must be greater than 0
4. `batch_size` must be greater than 0
5. if `number_gpus` is greater than 0 then `gpu_model` must be a non-empty
   string
6. if `fast_moe` is set and `number_gpus` is not divisible by it
7. if `fast_moe` is set and the `num_local_experts` of the Mixture of Experts
   (MoE) model is not divisible by `fast_moe` (which is interpreted as
   `ep_degrees` by fms-hf-tuning)

Runs raising the following errors are considered invalid due to running out of
GPU memory:

- `torch.cuda.OutOfMemoryError`
- `RuntimeError: CUDA error: an illegal memory access was encountered`

Measurements raising any other exception (including for example `RuntimeError`
containing the string `NCCL Error`) are considered to have `Failed`. They will
not contain the `is_valid` measured property, or any other property for that
matter. `Failed` measurements do not record any properties and can be repeated.
