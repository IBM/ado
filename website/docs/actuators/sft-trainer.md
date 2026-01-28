<!-- markdownlint-disable-next-line first-line-h1 -->
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

## Requirements

[fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning) imports
packages like `flash-attn` and `mamba-ssm`, which import `torch` during their  
build phase. This means that the base virtual environment of your Ray workers
must already include the appropriate version of `torch`:

<!-- markdownlint-disable line-length -->

- **`fms-hf-tuning <= 2.8.2`**

  - Install `torch==2.4.1`
  - For RayClusters on Kubernetes, use: `quay.io/ado/ado:1.0.1-py310-cu121-ofed2410v1140`

- **`fms-hf-tuning > 2.8.2`**
  - Install `torch==2.6.0`
    - Requires Python 3.11
  - For RayClusters on Kubernetes, use: `quay.io/ado/ado:c6ba952ad79a2d86d1174fd9aaebddd8953c78cf-py311-cu121-ofed2410v1140`
<!-- markdownlint-enable line-length -->

## Available experiments

The `SFTTrainer` actuator includes a set of experiments that evaluate different
fine-tuning strategies under controlled conditions. These experiments use
artificial datasets to ensure reproducibility and comparability across runs. A
full list of available experiments and their configurations is available in the
[README.MD file](https://github.com/ibm/ado/tree/main/plugins/actuators/sfttrainer/ado_actuators/sfttrainer)
of the Actuator.

The most frequently used experiments are:

### finetune_full_benchmark-v1.0.0

Performs full fine-tuning of all model parameters. This experiment is ideal for
evaluating end-to-end training performance and resource utilization on large
models.

??? note "Experiment documentation"

    An experiment instance:

    - performs full fine tuning
      - You may notice that even large-memory GPUs like the 80GB variant of the
        NVIDIA A100 chip need at least 2 GPUs to train models as big as 13B
        parameters.
    - the training data is artificial
    - `use_flash_attn` is set to True
    - `packing` is set to False
    - `torch_dtype` is set to `bfloat16` by default, can also be float16
    - uses the `FSDP` distributed backend for multi-gpu runs by default,
      can also be `DDP`
    - multi-gpu runs with FSDP and DDP backends use 1 process per GPU (via
      `accelerate`)
    - runs 1 epoch by default, can also run a custom number of steps
    - does not save checkpoint
    - loads weights from a PVC
    - request 2 CPU cores per GPU device (with a minimum of 2 cores)

    For FSDP runs we use the following `accelerate_config.yml` YAML file:

    <!-- markdownlint-disable line-length -->
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
    <!-- markdownlint-enable line-length -->

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
      - 3.0.0.1
      - 3.0.0
      - 2.8.2
      - 2.7.1
      - 2.6.0
      - 2.5.0
      - 2.4.0
      - 2.3.1
      - 2.2.1
      - 2.1.2 (default)
      - 2.1.1
      - 2.1.0
      - 2.0.1

    #### Full Finetuning Requirements

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
    - The PVC `ray-disorch-storage` mounted under `/data` with the synthetic
      datasets of the SFTTrainer actuator

    #### Full Finetuning Entity space

    Required:

    - model_name: Supported models:
      <!-- markdownlint-disable-next-line line-length -->
      `["granite-3b-1.5", "hf-tiny-model-private/tiny-random-BloomForCausalLM", "llama-7b", "granite-13b-v2", "llama-13b", "granite-20b-v2", "granite-7b-base", "granite-8b-japanese", "granite-8b-code-base", "granite-34b-code-base", "mistral-7b-v0.1", "llama3-8b", "llama3-70b", "mixtral-8x7b-instruct-v0.1", "llama2-70b", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "granite-3b-code-base-128k", "granite-8b-code-base-128k", "allam-1-13b", "granite-3-8b", "granite-3.1-2b", "granite-3.1-8b-instruct", "mistral-123b-v2", "granite-3.1-3b-a800m-instruct", "granite-vision-3.2-2b", "smollm2-135m", "llava-v1.6-mistral-7b", "granite-4.0-micro", "granite-4.0-h-1b", "granite-4.0-350m", "granite-4.0-h-small", "granite-4.0-h-micro", "granite-4.0-h-tiny", "granite-3.3-8b"]`
    - model_max_length: Maximum sequence length. Sequences will be right padded
      (and possibly truncated)
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
    - max_steps: Default is `-1`. The number of optimization steps to perform.
      Set to -1 to respect num_train_epochs instead.
    - num_train_epochs: Default is `1.0`. How many epochs to run. Ignored if
      max_steps is greater than 0.
    - stop_after_seconds: Default is `-1.0`. If set, the optimizer will be asked
      to stop after the specified time elapses. The check is performed after
      the end of each training step.
    - auto_stop_method: The default value is `None`. This parameter defines the
      method used to automatically stop the fine-tuning job. Supported values are
      `WARMUP_60S_STABLE_120S_OR_10_STEPS` and `None`. If set to
      `WARMUP_60S_STABLE_120S_OR_10_STEPS`, the job stops after spending at least
      60 seconds in the warmup phase plus the longer of 120 seconds or the
      duration of 10 optimization steps. This method excludes the first 60 seconds
      of training when calculating throughput and system metrics.
    - distributed_backend: Default is `FSDP` for multi-gpu measurements, `None`
      (i.e. Data Parallel (DP)) for single-gpu measurements. Which pytorch backend
      to use when training with multiple GPU devices.
    - number_nodes: Default is `1`. If set, actuator distributes tasks on multiple
      nodes. Each Node will use number_gpus/number_nodes GPUs. Each Node will use
      1 process for each GPU it uses
    - fms_hf_tuning_version: Default is `2.1.2`. Which version of fms-hf-tuning
      to use. Available options are: `2.8.2`, `2.7.1`, `2.6.0`, `2.5.0`, `2.4.0`,
      `2.3.1`, `2.2.1`, `2.1.2`, `2.1.0`, `2.0.1`
    - enable_roce: Default is `False`. This setting is only in effect for multi-node
      runs. It controls whether RDMA over Converged Ethernet (RoCE) is switched
      on or not.
    - fast_moe: Default is `0`. Configures the amount of expert parallel sharding.
      number_gpus must be divisible by it
    - fast_kernels: Default is `None`. Switches on fast kernels, the value is
      a list with strings of boolean values for
      `[fast_loss, fast_rms_layernorm, fast_rope_embeddings]`
    - optim: Default is `adamw_torch`. The optimizer to use.
      Available options are
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
      32-bit. Requires Ampere or higher NVIDIA add bf16 mixed precision support
      for NPU architecture or using CPU (use_cpu) or Ascend NPU.
      This is an experimental API and it may change. Can be `True`, `False`.
    - gradient_checkpointing_use_reentrant: Default is `False` Specify whether
      to use the activation checkpoint variant that requires reentrant autograd.
      This parameter should be passed explicitly. Torch version 2.5 will raise
      an exception if use_reentrant is not passed. If use_reentrant=False,
      checkpoint will use an implementation that does not require reentrant autograd.
      This allows checkpoint to support additional functionality, such as working
      as expected with torch.autograd.grad and support for keyword arguments input
      into the checkpointed function. Can be `True`, `False`.
    - fsdp_sharding_strategy: Default is `FULL_SHARD`. [1] FULL_SHARD (shards
      optimizer states, gradients and parameters), " [2] SHARD_GRAD_OP (shards
      optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD (shards
      optimizer states, gradients and parameters within each node while each node
      has full copy - equivalent to FULL_SHARD for single-node runs), 
      [5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients
      within each node while each node has full copy). For more information, please
      refer the official PyTorch docs.
    - fsdp_state_dict_type: Default is `FULL_STATE_DICT`.
      [1] FULL_STATE_DICT, [2] LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT
    - fsdp_use_orig_params: Default is `True`. If True, allows non-uniform
      `requires_grad` during init, which means support for interspersed frozen
      and trainable parameters. (useful only when `use_fsdp` flag is passed).
    - accelerate_config_mixed_precision: Default is `no`. Whether to use mixed
      precision training or not. Choose from `no`,`fp16`,`bf16` or `fp8`. `fp8`
      requires the installation of transformers-engine.
    - accelerate_config_fsdp_transformer_layer_cls_to_wrap: Default is None.
      List of transformer layer class names (case-sensitive) to wrap, e.g,
      `GraniteDecoderLayer`, `LlamaDecoderLayer`, `MistralDecoderLayer`,
      `BertLayer`, `GPTJBlock`, `T5Block` ... (useful only when using FSDP)
    - dataset_text_field: Default is None. Training dataset text field containing
      single sequence. Either the dataset_text_field or data_formatter_template
      need to be supplied.
      For running vision language model tuning pass the column name for text data.
    - dataset_image_field: Default is None. For running vision language model tuning
      pass the column name of the image data in the dataset.
    - remove_unused_columns: Default is True. Remove columns not required by the
      model when using an nlp.Dataset.
      - dataset_kwargs_skip_prepare_dataset: Default is False. When True, configures
        trl to skip preparing the dataset


    !!! info end

        Because running `accelerate` with a single gpu is unsupported, when setting
        `number_gpus` to 1 this experiment actually runs the `tuning.sft_trainer`
        script directly (i.e. a DataParallel (DP) run).

### finetune_full_stability-v1.0.0

Runs full fine-tuning five times and reports the proportion of tasks that fail
due to GPU memory limits, unknown errors, or complete successfully. This
experiment is useful for testing the model stability under different
configurations.

??? note "Experiment documentation"

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
      - 3.0.0.1
      - 3.0.0
      - 2.8.2
      - 2.7.1
      - 2.6.0
      - 2.5.0
      - 2.4.0
      - 2.3.1
      - 2.2.1
      - 2.1.2 (default)
      - 2.1.1
      - 2.1.0
      - 2.0.1

    #### Full Finetuning (Stability) Requirements

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
    - The PVC `ray-disorch-storage` mounted under `/data` with the synthetic
      datasets of the SFTTrainer actuator

    #### Full Finetuning (Stability) Entity space

          Required:

    - model_name: Supported models:
      <!-- markdownlint-disable-next-line line-length -->
      `["granite-3b-1.5", "hf-tiny-model-private/tiny-random-BloomForCausalLM", "llama-7b", "granite-13b-v2", "llama-13b", "granite-20b-v2", "granite-7b-base", "granite-8b-japanese", "granite-8b-code-base", "granite-34b-code-base", "mistral-7b-v0.1", "llama3-8b", "llama3-70b", "mixtral-8x7b-instruct-v0.1", "llama2-70b", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "granite-3b-code-base-128k", "granite-8b-code-base-128k", "allam-1-13b", "granite-3-8b", "granite-3.1-2b", "granite-3.1-8b-instruct", "mistral-123b-v2", "granite-3.1-3b-a800m-instruct", "granite-vision-3.2-2b", "smollm2-135m", "llava-v1.6-mistral-7b", "granite-4.0-micro", "granite-4.0-h-1b", "granite-4.0-350m", "granite-4.0-h-small", "granite-4.0-h-micro", "granite-4.0-h-tiny", "granite-3.3-8b"]`
    - model_max_length: Maximum sequence length. Sequences will be right padded
      (and possibly truncated)
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
    - torch_dtype: Default is `bfloat16`.
      One of `bfloat16`, `float32`, `float16`
    - stop_after_seconds: Default is `-1.0`. If set, the optimizer will be asked
      to stop after the specified time elapses. The check is performed after
      the end of each training step.
    - auto_stop_method: The default value is `None`. This parameter defines the
      method used to automatically stop the fine-tuning job. Supported values are
      `WARMUP_60S_STABLE_120S_OR_10_STEPS` and `None`. If set to
      `WARMUP_60S_STABLE_120S_OR_10_STEPS`, the job stops after spending at least
      60 seconds in the warmup phase plus the longer of 120 seconds or the
      duration of 10 optimization steps. This method excludes the first 60 seconds
      of training when calculating throughput and system metrics.
    - distributed_backend: Default is `FSDP` for multi-gpu measurements, `None`
      (i.e. Data Parallel (DP)) for single-gpu measurements. Which pytorch backend
      to use when training with multiple GPU devices.
    - number_nodes: Default is `1`. If set, actuator distributes tasks on multiple
      nodes. Each Node will use number_gpus/number_nodes GPUs.
      Each Node will use 1 process for each GPU it uses
    - fms_hf_tuning_version: Default is `2.1.2`. Which version of fms-hf-tuning
      to use. Available options are: `3.1.0`, `3.0.0.1`, `3.0.0`, 2.8.2`, 
      `2.7.1`, `2.6.0`, `2.5.0`, `2.4.0`, `2.3.1`, `2.2.1`, `2.1.2`, `2.1.0`,
      `2.0.1`
    - enable_roce: Default is `False`. This setting is only in effect for multi-node
      runs. It controls whether RDMA over Converged Ethernet (RoCE) is switched
      on or not.
    - fast_moe: Default is `0`. Configures the amount of expert parallel sharding.
      number_gpus must be divisible by it
    - fast_kernels: Default is `None`. Switches on fast kernels, the value is
      a list with strings of boolean values for
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
      32-bit. Requires Ampere or higher NVIDIA add bf16 mixed precision support 
      for NPU architecture or using CPU (use_cpu) or Ascend NPU.
      This is an experimental API and it may change. Can be `True`, `False`.
    - gradient_checkpointing_use_reentrant: Default is `False` Specify whether to
      use the activation checkpoint variant that requires reentrant autograd. This
      parameter should be passed explicitly. Torch version 2.5 will raise an
      exception if use_reentrant is not passed. If use_reentrant=False, checkpoint
      will use an implementation that does not require reentrant autograd. This
      allows checkpoint to support additional functionality, such as working as
      expected with torch.autograd.grad and support for keyword arguments input
      into the checkpointed function. Can be `True`, `False`.
    - fsdp_sharding_strategy: Default is `FULL_SHARD`. [1] FULL_SHARD (shards
      optimizer states, gradients and parameters), " [2] SHARD_GRAD_OP (shards
      optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD (shards
      optimizer states, gradients and parameters within each node while each node
      has full copy - equivalent to FULL_SHARD for single-node runs), 
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
    - accelerate_config_fsdp_transformer_layer_cls_to_wrap: Default is None.
      List of transformer layer class names (case-sensitive) to wrap, e.g,
      `GraniteDecoderLayer`, `LlamaDecoderLayer`, `MistralDecoderLayer`,
      `BertLayer`, `GPTJBlock`, `T5Block` ... (useful only when using FSDP)
    - dataset_text_field: Default is None. Training dataset text field containing
      single sequence. Either the dataset_text_field or data_formatter_template
      need to be supplied. For running vision language model tuning pass
      the column name for text data.
    - dataset_image_field: Default is None. For running vision language model tuning
      pass the column name of the image data in the dataset.
    - remove_unused_columns: Default is True. Remove columns not required by the
      model when using an nlp.Dataset.
    - dataset_kwargs_skip_prepare_dataset: Default is False. When True, configures
      trl to skip preparing the dataset

    #### Full Finetuning (Stability) Measured properties

    - f_gpu_oom: fraction of tasks that ran out of GPU memory
    - f_other_error: fraction of tasks that ran into an unknown error
    - f_no_error: fraction of tasks that completed successfully
    - is_valid: whether this collection of tasks is a valid point to investigate

    !!! info end

        Because running `accelerate` with a single gpu is unsupported, when setting
        `number_gpus` to 1 this experiment actually runs the `tuning.sft_trainer`
        script directly (i.e. a DataParallel (DP) run).

### finetune_lora_benchmark-v1.0.0

Executes LoRA-based fine-tuning, a parameter-efficient method that adapts only a
small subset of model weights. This benchmark is useful for scenarios where
compute or memory resources are limited, while still enabling meaningful
adaptation.

??? note "Experiment documentation"

    An experiment instance:

    - performs LORA fine tuning
    - the training data is artificial
    - `use_flash_attn` is set to True
    - `packing` is set to False
    - `torch_dtype` is set to `bfloat16` by default, can also be float16
    - uses the `FSDP` distributed backend for multi-gpu runs by default,
      can also be `DDP`
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
      - 3.0.0.1
      - 3.0.0
      - 2.8.2
      - 2.7.1
      - 2.6.0
      - 2.5.0
      - 2.4.0
      - 2.3.1
      - 2.2.1
      - 2.1.2 (default)
      - 2.1.1
      - 2.1.0
      - 2.0.1

    #### LoRA Requirements

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
    - The PVC `ray-disorch-storage` mounted under `/data` with the synthetic
      datasets of the SFTTrainer actuator

    #### LoRA Entity space

    Required:

    - model_name: Supported models:
      <!-- markdownlint-disable-next-line line-length -->
      `["granite-3b-1.5", "hf-tiny-model-private/tiny-random-BloomForCausalLM", "llama-7b", "granite-13b-v2", "llama-13b", "granite-20b-v2", "granite-7b-base", "granite-8b-japanese", "granite-8b-code-base", "granite-34b-code-base", "mistral-7b-v0.1", "llama3-8b", "llama3-70b", "mixtral-8x7b-instruct-v0.1", "llama2-70b", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "granite-3b-code-base-128k", "granite-8b-code-base-128k", "allam-1-13b", "granite-3-8b", "granite-3.1-2b", "granite-3.1-8b-instruct", "mistral-123b-v2", "granite-3.1-3b-a800m-instruct", "granite-vision-3.2-2b", "smollm2-135m", "llava-v1.6-mistral-7b", "granite-4.0-micro", "granite-4.0-h-1b", "granite-4.0-350m", "granite-4.0-h-small", "granite-4.0-h-micro", "granite-4.0-h-tiny", "granite-3.3-8b"]`
    - model_max_length: Maximum sequence length. Sequences will be right padded
      (and possibly truncated)
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
    - stop_after_seconds: Default is `-1.0`. If set, the optimizer will be asked
      to stop after the specified time elapses. The check is performed after
      the end of each training step.
    - auto_stop_method: The default value is `None`. This parameter defines the
      method used to automatically stop the fine-tuning job. Supported values are
      `WARMUP_60S_STABLE_120S_OR_10_STEPS` and `None`. If set to
      `WARMUP_60S_STABLE_120S_OR_10_STEPS`, the job stops after spending at least
      60 seconds in the warmup phase plus the longer of 120 seconds or the
      duration of 10 optimization steps. This method excludes the first 60 seconds
      of training when calculating throughput and system metrics.
    - distributed_backend: Default is `FSDP` for multi-gpu measurements, `None`
      (i.e. Data Parallel (DP)) for single-gpu measurements. Which pytorch backend
      to use when training with multiple GPU devices.
    - number_nodes: Default is `1`. If set, actuator distributes tasks on multiple
      nodes. Each Node will use number_gpus/number_nodes GPUs.
      Each Node will use 1 process for each GPU it uses
    - fms_hf_tuning_version: Default is `2.1.2`. Which version of fms-hf-tuning
      to use. Available options are: `3.1.0`, `3.0.0.1`, `3.0.0`, 2.8.2`, 
      `2.7.1`, `2.6.0`, `2.5.0`, `2.4.0`, `2.3.1`, `2.2.1`, `2.1.2`, `2.1.0`,
      `2.0.1`
    - enable_roce: Default is `False`. This setting is only in effect for multi-node
      runs. It controls whether RDMA over Converged Ethernet (RoCE) is switched
      on or not.
    - fast_moe: Default is `0`. Configures the amount of expert parallel sharding.
      number_gpus must be divisible by it
    - fast_kernels: Default is `None`. Switches on fast kernels, the value is
      a list with strings of boolean values for
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
      32-bit. Requires Ampere or higher NVIDIA add bf16 mixed precision support 
      for NPU architecture or using CPU (use_cpu) or Ascend NPU.
      This is an experimental API and it may change. Can be `True`, `False`.
    - gradient_checkpointing_use_reentrant: Default is `False` Specify whether to
      use the activation checkpoint variant that requires reentrant autograd. This
      parameter should be passed explicitly. Torch version 2.5 will raise an
      exception if use_reentrant is not passed. If use_reentrant=False, checkpoint
      will use an implementation that does not require reentrant autograd. This
      allows checkpoint to support additional functionality, such as working as
      expected with torch.autograd.grad and support for keyword arguments input
      into the checkpointed function. Can be `True`, `False`.
    - fsdp_sharding_strategy: Default is `FULL_SHARD`. [1] FULL_SHARD (shards
      optimizer states, gradients and parameters), " [2] SHARD_GRAD_OP (shards
      optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD (shards
      optimizer states, gradients and parameters within each node while each node
      has full copy - equivalent to FULL_SHARD for single-node runs), 
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
    - accelerate_config_fsdp_transformer_layer_cls_to_wrap: Default is None.
      List of transformer layer class names (case-sensitive) to wrap, e.g,
      `GraniteDecoderLayer`, `LlamaDecoderLayer`, `MistralDecoderLayer`,
      `BertLayer`, `GPTJBlock`, `T5Block` ... (useful only when using FSDP)
    - dataset_text_field: Default is None. Training dataset text field containing
      single sequence. Either the dataset_text_field or data_formatter_template
      need to be supplied. For running vision language model tuning pass
      the column name for text data.
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
    - `granite-4.0-micro`: `["q_proj", "v_proj"]`
    - `granite-4.0-h-1b`: `["q_proj", "v_proj"]`
    - `granite-4.0-350m`: `["q_proj", "v_proj"]`
    - `granite-4.0-h-small`: `["q_proj", "v_proj"]`
    - `granite-4.0-h-micro`: `["q_proj", "v_proj"]`
    - `granite-4.0-h-tiny`: `["q_proj", "v_proj"]`
    - `granite-3.0-1b-a400m-base`: `["q_proj", "v_proj"]`
    - `granite-3.1-3b-a800m-instruct`: `["q_proj", "v_proj"]`
    - `granite-vision-3.2-2b`: `["q_proj", "v_proj"]`
    - `llava-v1.6-mistral-7b`: `["q_proj", "v_proj"]`

    !!! info end

        Because running `accelerate` with a single gpu is unsupported, when setting
        `number_gpus` to 1 this experiment actually runs the `tuning.sft_trainer`
        script directly (i.e. a DataParallel (DP) run).

### finetune_pt_benchmark-v1.0.0

Runs prompt-tuning, a lightweight fine-tuning strategy that prepends trainable
prompts to the input. Similar to LoRA, this benchmark is useful for compute or
memory constrained environments.

??? note "Experiment documentation"

    An experiment instance:

    - performs prompt-tuning fine tuning
    - the training data is artificial
    - `use_flash_attn` is set to True
    - `packing` is set to False
    - `torch_dtype` is set to `bfloat16` by default, can also be float16
    - uses the `FSDP` distributed backend for multi-gpu runs by default,
      can also be `DDP`
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
      - 3.0.0.1
      - 3.0.0
      - 2.8.2
      - 2.7.1
      - 2.6.0
      - 2.5.0
      - 2.4.0
      - 2.3.1
      - 2.2.1
      - 2.1.2 (default)
      - 2.1.1
      - 2.1.0
      - 2.0.1

    #### Prompt Tuning Requirements

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
    - The PVC `ray-disorch-storage` mounted under `/data` with the synthetic
      datasets of the SFTTrainer actuator

    #### Prompt Tuning Entity space

    Required:

    - model_name: Supported models:
      <!-- markdownlint-disable-next-line line-length -->
      `["granite-3b-1.5", "hf-tiny-model-private/tiny-random-BloomForCausalLM", "llama-7b", "granite-13b-v2", "llama-13b", "granite-20b-v2", "granite-7b-base", "granite-8b-japanese", "granite-8b-code-base", "granite-34b-code-base", "mistral-7b-v0.1", "llama3-8b", "llama3-70b", "mixtral-8x7b-instruct-v0.1", "llama2-70b", "llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "granite-3b-code-base-128k", "granite-8b-code-base-128k", "allam-1-13b", "granite-3-8b", "granite-3.1-2b", "granite-3.1-8b-instruct", "mistral-123b-v2", "granite-3.1-3b-a800m-instruct", "granite-vision-3.2-2b", "smollm2-135m", "llava-v1.6-mistral-7b", "granite-4.0-micro", "granite-4.0-h-1b", "granite-4.0-350m", "granite-4.0-h-small", "granite-4.0-h-micro", "granite-4.0-h-tiny", "granite-3.3-8b"]`
    - model_max_length: Maximum sequence length. Sequences will be right padded
      (and possibly truncated)
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
    - stop_after_seconds: Default is `-1.0`. If set, the optimizer will be asked
      to stop after the specified time elapses. The check is performed after
      the end of each training step.
    - auto_stop_method: The default value is `None`. This parameter defines the
      method used to automatically stop the fine-tuning job. Supported values are
      `WARMUP_60S_STABLE_120S_OR_10_STEPS` and `None`. If set to
      `WARMUP_60S_STABLE_120S_OR_10_STEPS`, the job stops after spending at least
      60 seconds in the warmup phase plus the longer of 120 seconds or the
      duration of 10 optimization steps. This method excludes the first 60 seconds
      of training when calculating throughput and system metrics.
    - distributed_backend: Default is `FSDP` for multi-gpu measurements, `None`
      (i.e. Data Parallel (DP)) for single-gpu measurements. Which pytorch backend
      to use when training with multiple GPU devices.
    - number_nodes: Default is `1`. If set, actuator distributes tasks on multiple
      nodes. Each Node will use number_gpus/number_nodes GPUs.
      Each Node will use 1 process for each GPU it uses
    - fms_hf_tuning_version: Default is `2.1.2`. Which version of fms-hf-tuning
      to use. Available options are: `3.1.0`, `3.0.0.1`, `3.0.0`, 2.8.2`, 
      `2.7.1`, `2.6.0`, `2.5.0`, `2.4.0`, `2.3.1`, `2.2.1`, `2.1.2`, `2.1.0`,
      `2.0.1`
    - enable_roce: Default is `False`. This setting is only in effect for multi-node
      runs. It controls whether RDMA over Converged Ethernet (RoCE) is switched
      on or not.
    - fast_moe: Default is `0`. Configures the amount of expert parallel sharding.
      number_gpus must be divisible by it
    - fast_kernels: Default is `None`. Switches on fast kernels, the value is
      a list with strings of boolean values for
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
      32-bit. Requires Ampere or higher NVIDIA add bf16 mixed precision support 
      for NPU architecture or using CPU (use_cpu) or Ascend NPU.
      This is an experimental API and it may change. Can be `True`, `False`.
    - gradient_checkpointing_use_reentrant: Default is `False` Specify whether to
      use the activation checkpoint variant that requires reentrant autograd. This
      parameter should be passed explicitly. Torch version 2.5 will raise an
      exception if use_reentrant is not passed. If use_reentrant=False, checkpoint
      will use an implementation that does not require reentrant autograd. This
      allows checkpoint to support additional functionality, such as working as
      expected with torch.autograd.grad and support for keyword arguments input
      into the checkpointed function. Can be `True`, `False`.
    - fsdp_sharding_strategy: Default is `FULL_SHARD`. [1] FULL_SHARD (shards
      optimizer states, gradients and parameters), " [2] SHARD_GRAD_OP (shards
      optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD (shards
      optimizer states, gradients and parameters within each node while each node
      has full copy - equivalent to FULL_SHARD for single-node runs), 
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
    - accelerate_config_fsdp_transformer_layer_cls_to_wrap: Default is None.
      List of transformer layer class names (case-sensitive) to wrap, e.g,
      `GraniteDecoderLayer`, `LlamaDecoderLayer`, `MistralDecoderLayer`,
      `BertLayer`, `GPTJBlock`, `T5Block` ... (useful only when using FSDP)
    - dataset_text_field: Default is None. Training dataset text field containing
      single sequence. Either the dataset_text_field or data_formatter_template
      need to be supplied. For running vision language model tuning pass
      the column name for text data.
    - dataset_image_field: Default is None. For running vision language model tuning
      pass the column name of the image data in the dataset.
    - remove_unused_columns: Default is True. Remove columns not required by the
      model when using an nlp.Dataset.
    - dataset_kwargs_skip_prepare_dataset: Default is False. When True, configures
      trl to skip preparing the dataset

    !!! info end

        Because running `accelerate` with a single gpu is unsupported, when setting
        `number_gpus` to 1 this experiment actually runs the `tuning.sft_trainer`
        script directly (i.e. a DataParallel (DP) run).

### finetune_gtpq-lora_benchmark-v1.0.0

Combines LoRA with GPTQ quantization to enable fine-tuning on quantized models.
This benchmark is tailored for scenarios where model size and inference
efficiency are critical, and it leverages fused kernels and quantized weights
for performance.

??? note "Experiment documentation"

    An experiment instance:

    - performs LORA fine tuning
    - the training data is artificial
    - `use_flash_attn` is set to True
    - `packing` is set to False
    - `torch_dtype` is set to `float16`, cannot be a different value
    - uses the `FSDP` distributed backend for multi-gpu runs by default,
      can also be `DDP`
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
      - 3.0.0.1
      - 3.0.0
      - 2.8.2
      - 2.7.1
      - 2.6.0
      - 2.5.0
      - 2.4.0
      - 2.3.1
      - 2.2.1
      - 2.1.2 (default)
      - 2.1.1
      - 2.1.0
      - 2.0.1

    #### GPTQ LoRA Requirements

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
    - The PVC `ray-disorch-storage` mounted under `/data` with the synthetic
      datasets of the SFTTrainer actuator

    #### GPTQ LoRA Entity space

    Required:

    - model_name: Supported models:
      <!-- markdownlint-disable-next-line line-length -->
      `["llama-7b", "granite-20b-v2", "granite-7b-base", "granite-8b-code-instruct", "granite-34b-code-base", "mistral-7b-v0.1", "llama3-70b", "mixtral-8x7b-instruct-v0.1", "llama3.1-405b"]`
    - model_max_length: Maximum sequence length. Sequences will be right padded
      (and possibly truncated)
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
    - stop_after_seconds: Default is `-1.0`. If set, the optimizer will be asked
      to stop after the specified time elapses. The check is performed after
      the end of each training step.
    - auto_stop_method: The default value is `None`. This parameter defines the
      method used to automatically stop the fine-tuning job. Supported values are
      `WARMUP_60S_STABLE_120S_OR_10_STEPS` and `None`. If set to
      `WARMUP_60S_STABLE_120S_OR_10_STEPS`, the job stops after spending at least
      60 seconds in the warmup phase plus the longer of 120 seconds or the
      duration of 10 optimization steps. This method excludes the first 60 seconds
      of training when calculating throughput and system metrics.
    - distributed_backend: Default is `FSDP` for multi-gpu measurements, `None`
      (i.e. Data Parallel (DP)) for single-gpu measurements. Which pytorch backend
      to use when training with multiple GPU devices.
    - number_nodes: Default is `1`. If set, actuator distributes tasks on multiple
      nodes. Each Node will use number_gpus/number_nodes GPUs.
      Each Node will use 1 process for each GPU it uses
    - fms_hf_tuning_version: Default is `2.1.2`. Which version of fms-hf-tuning
      to use. Available options are: `3.1.0`, `3.0.0.1`, `3.0.0`, 2.8.2`, 
      `2.7.1`, `2.6.0`, `2.5.0`, `2.4.0`, `2.3.1`, `2.2.1`, `2.1.2`, `2.1.0`,
      `2.0.1`
    - enable_roce: Default is `False`. This setting is only in effect for multi-node
      runs. It controls whether RDMA over Converged Ethernet (RoCE) is switched
      on or not.
    - fast_moe: Default is `0`. Configures the amount of expert parallel sharding.
      number_gpus must be divisible by it
    - fast_kernels: Default is `None`. Switches on fast kernels, the value is
      a list with strings of boolean values for
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
      32-bit. Requires Ampere or higher NVIDIA add bf16 mixed precision support 
      for NPU architecture or using CPU (use_cpu) or Ascend NPU.
      This is an experimental API and it may change. Can be `True`, `False`.
    - gradient_checkpointing_use_reentrant: Default is `False` Specify whether to
      use the activation checkpoint variant that requires reentrant autograd. This
      parameter should be passed explicitly. Torch version 2.5 will raise an
      exception if use_reentrant is not passed. If use_reentrant=False, checkpoint
      will use an implementation that does not require reentrant autograd. This
      allows checkpoint to support additional functionality, such as working as
      expected with torch.autograd.grad and support for keyword arguments input
      into the checkpointed function. Can be `True`, `False`.
    - fsdp_sharding_strategy: Default is `FULL_SHARD`. [1] FULL_SHARD (shards
      optimizer states, gradients and parameters), " [2] SHARD_GRAD_OP (shards
      optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD (shards
      optimizer states, gradients and parameters within each node while each node
      has full copy - equivalent to FULL_SHARD for single-node runs), 
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
    - accelerate_config_fsdp_transformer_layer_cls_to_wrap: Default is None.
      List of transformer layer class names (case-sensitive) to wrap, e.g,
      `GraniteDecoderLayer`, `LlamaDecoderLayer`, `MistralDecoderLayer`,
      `BertLayer`, `GPTJBlock`, `T5Block` ... (useful only when using FSDP)
    - dataset_text_field: Default is None. Training dataset text field containing
      single sequence. Either the dataset_text_field or data_formatter_template
      need to be supplied. For running vision language model tuning pass
      the column name for text data.
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


    !!! info end

        Because running `accelerate` with a single gpu is unsupported, when setting
        `number_gpus` to 1 this experiment actually runs the `tuning.sft_trainer`
        script directly (i.e. a DataParallel (DP) run).

### Actuator Parameters

This section describes the fields you may optionally configure in your
`actuatorconfiguration` resource for the `SFTTrainer` actuator.

### Example Actuator Configuration YAML

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```yaml
actuatorIdentifier: SFTTrainer
parameters:
  match_exact_dependencies: true
  output_dir: "output"
  data_directory: "/data/fms-hf-tuning/artificial-dataset/"
  hf_home: "/hf-models-pvc/huggingface_home"
  model_map:
    granite-3.1-2b:
      Vanilla: "ibm-granite/granite-3.1-2b-base"
  num_tokens_cache_directory: "cache"
```

<!-- markdownlint-enable line-length -->

### Configuration Fields

#### `match_exact_dependencies` (bool, default: `true`)

- **Description**: If `true`, the measurement runs in a virtual environment that
  exactly matches the Python packages of the selected `fms-hf-tuning` version.
  This enables all optional features like `fast_kernels`, `fast_moe`, and
  `flash_attn`.
- **Set to `false`** if running on devices with limited support (e.g., MacBooks
  or ARM CPUs), to avoid incompatible packages and features that depend on using
  NVIDIA GPUs.

#### `output_dir` (str, default: `"output"`)

- **Description**: Directory prefix where the fine-tuned model weights will be
  saved.

#### `data_directory` (str, default: `"/data/fms-hf-tuning/artificial-dataset/"`)

- **Description**: Path to the directory containing the dataset files used for
  fine-tuning.

#### `aim_db` (str, default: None)

- **Description**: Endpoint of the AIM server used to log training metrics. When
  set to None the measurement will use a temporary AIM repository that will be
  garbage collected after the termination of the measurement.

#### `aim_dashboard_url` (str or null, optional)

- **Description**: URL of the AIM dashboard. If set, this will be included in
  the metadata of the measurement results.
- **Example**: `"http://aim-dashboard.example.com"`

#### `hf_home` (str, default: `"/hf-models-pvc/huggingface_home"`)

- **Description**: Directory where Hugging Face stores authentication tokens and
  model cache.

#### `model_map` (dict, optional)

- **Description**: Maps model identifiers to their corresponding Hugging Face
  model ids and absolute paths. The contents of this dictionary will override
  the defaults that ship with the Actuator.
- **Example**:

  <!-- markdownlint-disable-next-line code-block-style -->
  ```yaml
  model_map:
    granite-3.1-2b:
      Vanilla: "ibm-granite/granite-3.1-2b-base"
  ```

#### `num_tokens_cache_directory` (str or null, default: `"cache"`)

- **Description**: Directory used to cache token counts for datasets. This
  avoids recomputing token counts, which can be time-consuming. Relative paths
  are resolved under `@data_directory`.
- **Set to `null`** to disable caching.

### Measured Properties

Each experiment collects detailed runtime and system-level metrics using
[AIM](https://github.com/aimhubio/aim). The AIM metrics are aggregated into the
following before being stored in ado's database:

#### **GPU Metrics**

- **Compute Utilization**: `min`, `avg`, `max` (%)
- **Memory Utilization**: `min`, `avg`, `max`, `peak` (%)
- **Power Usage**: `min`, `avg`, `max` (Watts and %)

#### **CPU Metrics**

- **Compute Utilization**: Average CPU usage per core (%)
- **Memory Utilization**: Average memory usage of the training process (%)

#### **Training Performance**

- **`train_runtime`**: Duration in seconds from the start of the first training
  step to the end of the last training step.
- **`train_samples_per_second`**: May be inaccurate, as HuggingFace uses a
  heuristic to estimate this.
- **`train_steps_per_second`**: May be inaccurate due to HuggingFace's
  heuristic-based measurement.
- **`train_tokens_per_second`**: May be inaccurate, as it relies on
  HuggingFace's heuristic.
- **`train_tokens_per_gpu_per_second`**: May be inaccurate for the same reason,
  HuggingFace uses a heuristic.
- **`dataset_tokens_per_second`**: The actuator computes this in an accurate
  way.
- **`dataset_tokens_per_second_per_gpu`**: The actuator computes this in an
  accurate way.

!!! info end

    We report all **system metrics** as min/avg/max over the duration of the run.
    GPU metrics are collected per device; CPU metrics are collected for the training
    process. Token throughput accounts for padding and sequence truncation.

#### **Validation**

Each experiment includes a computed `is_valid` flag that indicates whether the
run was structurally and functionally valid. A run is marked **invalid** if any
of the following conditions are met:

##### **Configuration Errors**

- `batch_size` is not evenly divisible by `number_gpus`
- `number_gpus` is not evenly divisible by `number_nodes`
- `number_nodes` is less than 1
- `batch_size` is less than 1
- `gpu_model` is missing or empty when `number_gpus > 0`

##### **Incompatible Mixture of Experts (MoE) Settings**

- `fast_moe` is set but `number_gpus` is not divisible by it
- `fast_moe` is set but the model’s `num_local_experts` is not divisible by
  `fast_moe`

##### **Runtime Failures**

- The run raises a `torch.cuda.OutOfMemoryError` (considered invalid due to GPU
  memory exhaustion)
- The run raises a
  `RuntimeError: CUDA error: an illegal memory access was encountered` exception
  (considered invalid due to GPU memory exhaustion)
- The run raises other exceptions (e.g., `RuntimeError` with `NCCL Error`) -
  these are marked as **failed** and do not record any metrics

  > **Note**: Failed runs are not persisted into ado's database. Restarting an
  > operation will cause ado to retry them.

This validation logic ensures that only meaningful and resource-compatible runs
are included in the information we store in ado's database.

## Configure your RayCluster

Running SFTTrainer experiments requires:

- [GPU workers with custom resources indicating the available GPU devices](#annotating-gpu-workers-with-custom-resources)
- [A dataset](#creating-the-datasets)
- [The model weights](#model-weights)

Use the information below to
[deploy your RayCluster](../getting-started/installing-backend-services.md#deploying-kuberay-and-creating-a-raycluster).

### Annotating GPU workers with custom resources

The `SFTTrainer` actuator leverages Ray's custom resource scheduling to
efficiently allocate GPU-powered tasks to workers equipped with the appropriate
hardware. It uses the following custom resources:

#### Custom Resource Types

- **`full-worker`**  
  Some Ray tasks require exclusive access to an entire node. These tasks request
  the `full-worker` resource. GPU workers that occupy a full node should have
  exactly one `full-worker` custom resource.

- **`${GPU_MODEL}`**  
  This custom resource key corresponds to the specific GPU model available on
  the node, with the value indicating the number of devices. Supported GPU
  models include: - `NVIDIA-A100-SXM4-80GB` - `NVIDIA-A100-80GB-PCIe` -
  `NVIDIA-H100-80GB-HBM3` - `NVIDIA-H100-PCIe` - `Tesla-V100-PCIE-16GB` -
  `Tesla-T4` - `L40S`

- **`RoCE`**  
  Tasks that utilize RDMA over Converged Ethernet (RoCE) request the `RoCE`
  resource. For guidance on configuring RoCE in your RayCluster, refer to the
  instructions linked at the bottom of this page.

## Creating the datasets

The **SFTTrainer** actuator supports both **text-to-text** and **image-to-text**
tuning experiments. Installing the actuator provides access to 2 command-line
utilities for generating synthetic datasets.

By default, the actuator expects the dataset files under
`/data/fms-hf-tuning/artificial-dataset/`

You can override this path by setting the `data_directory` parameter via an
**ActuatorConfiguration** resource and referencing it in the **Operations** you
create. We include a link to the relevant documentation at the bottom of this
page.

### Dataset for text-to-text tasks

For text-to-text tasks, create a dataset file with the name
`news-tokens-16384plus-entries-4096.jsonl`.

Use the following command:

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
sfttrainer_generate_dataset_text -o /data/fms-hf-tuning/artificial-dataset/news-tokens-16384plus-entries-4096.jsonl
```

If you are working with a remote RayCluster, run this as a **remote Ray job**
using a Ray runtime environment that contains the python package for the
SFTTrainer actuator. At the bottom of this page you will find a link to our
documentation on submitting remote Ray jobs that use the code of Actuators.

!!! note

    If your RayCluster Worker nodes already have the SFTTrainer wheel installed,
    you can skip building the wheel and using a ray runtime environment.
    Go directly to the `ray job submit` step.
    Just change the commandline so that it does not use
    the `ray_runtime.yaml` file.

For example, build the wheel file for SFTTrainer and create the following
`ray_runtime_env.yaml`:

<!-- markdownlint-disable-next-line code-block-style -->
```yaml
pip:
  - ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/ado_sfttrainer-1.0.2.dev84+g1ab8f43d-py3-none-any.whl
env_vars:
  PYTHONUNBUFFERED: "x"
```

!!! note

    Your wheel file will have a different name so update the `ray_runtime_env.yaml`
    file accordingly. Make sure you keep the
    `${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/` prefix.

Then start a Ray job that executes `sfttrainer_generate_dataset_text` and
pointing it to your remote RayCluster and references your `ray_runtime_env.yaml`
file. For example, if your RayCluster is listening on `http://localhost:8265`
run the following command in the same directory as your `ray_runtime_env.yaml`
file:

!!! info end

    If you are using a remote RayCluster on Kubernetes remember to
    [start a port-forward to the RayCluster head node](../../getting-started/remote_run/#specifying-the-remote-ray-cluster-to-submit-to-address).

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ray job submit --address http://localhost:8265 --runtime-env ray_runtime_env.yaml --working-dir $PWD -v -- \
  sfttrainer_generate_dataset_text \
  -o /data/fms-hf-tuning/artificial-dataset/news-tokens-16384plus-entries-4096.jsonl
```

<!-- markdownlint-enable line-length -->

### Dataset for image-to-text tasks

SFTTrainer supports 2 datasets for text-to-image tasks:

- `vision-384x384-16384plus-entries-4096.parquet`
- `vision-384x768-16384plus-entries-4096.parquet`

To create the dataset files use the same `ray_runtime_env.yaml` file as above
but this time start 2 Ray Jobs:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ray job submit --address http://localhost:8265 --runtime-env ray_runtime_env.yaml --working-dir $PWD -v -- \
  sfttrainer_generate_dataset_vision --image-width 384  --image-height 384 \
  -o /data/fms-hf-tuning/artificial-dataset/vision-384x384-16384plus-entries-4096.parquet


ray job submit --address http://localhost:8265 --runtime-env ray_runtime_env.yaml --working-dir $PWD -v -- \
  sfttrainer_generate_dataset_vision --image-width 384  --image-height 768 \
  -o /data/fms-hf-tuning/artificial-dataset/vision-384x768-16384plus-entries-4096.parquet
```

<!-- markdownlint-enable line-length -->

## Model Weights

The actuator supports model weights from both the **HuggingFace repository** and
**local directories**. You can find the full list of supported models in the
[models.yaml](https://github.com/ibm/ado/blob/main/plugins/actuators/sfttrainer/ado_actuators/sfttrainer/config/models.yaml)
file.

!!! note

    The actuator attempts to cache Hugging Face model weights the first time
    it runs an operation that references them. To avoid race conditions when
    running multiple experiments with the same weights, we recommend
    **pre-fetching** the weights in advance.

Identify the models you want to cache and then create a `models.yaml` file
structured as a double-nested dictionary.

- The **outer dictionary** keys are the names of the models.
- Each **inner dictionary** maps model weight types to their corresponding
  Hugging Face identifiers.

Supported model weight types include:

- `Vanilla`
- `QPTQ-Quantized`

Here’s a simple example that caches the `HuggingFaceTB/SmolLM2-135M` model
weights from HuggingFace:

<!-- markdownlint-disable-next-line code-block-style -->
```yaml
smollm2-135m:
  Vanilla: HuggingFaceTB/SmolLM2-135M
```

Next, choose a directory to use as your HuggingFace home. By default, SFTTrainer
uses `/hf-models-pvc/huggingface_home`. To override this, set the `hf_home`
parameter in your **ActuatorConfiguration** resource just like you did for
overriding the location of dataset files.

For example, to cache the model weights under `/my/hf_home/` use the following
command:

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
sfttrainer_download_hf_weights -i models.yaml -o /my/hf_home
```

If you are working with a remote RayCluster then submit a Ray job similar to the
above section for generating datasets:

!!! info end

    If you are using a remote RayCluster on Kubernetes remember to
    [start a port-forward to the RayCluster head node](../../getting-started/remote_run/#specifying-the-remote-ray-cluster-to-submit-to-address).

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ray job submit --address http://localhost:8265 --runtime-env ray_runtime_env.yaml --working-dir $PWD -v -- \
  sfttrainer_download_hf_weights -i models.yaml -o /my/hf_home
```

<!-- markdownlint-enable line-length -->

## Configure your RayCluster for RDMA over Converged Ethernet (RoCE)

[**RoCE**](https://www.roceinitiative.org/roce-introduction/) enables
high-throughput, low-latency communication between GPU nodes distributed on
multiple nodes by bypassing the kernel and reducing CPU overhead. This is
especially beneficial for multi-node AI workloads that rely on fast inter-GPU
communication, such as distributed training with
[NVIDIA NCCL](https://developer.nvidia.com/nccl).

To enable RoCE in a RayCluster on Kubernetes, you need to:

1. **Build a GPU worker image** with the necessary OFED and NCCL libraries.
2. **Configure the RayCluster custom resource** to:
   - Set environment variables for NCCL that switch on the RoCE feature.
   - Mount the NCCL topology file to ensure optimal GPU-to-GPU communication
     paths are used during collective operations.
3. **Ensure the Kubernetes nodes and network** are RoCE-capable and properly
   configured.

### Prerequisites

- The **system administrator** has configured the GPU nodes and network
  infrastructure to support RoCE, including BIOS, firmware, switch settings, and
  lossless Ethernet features.
- The **NCCL topology file** is provided by the system administrator to optimize
  GPU communication paths.
- The **Kubernetes administrator** has granted the RayCluster service account
  appropriate RBAC permissions and PodSecurity settings necessary for RoCE. In
  this example we will:
  - Run containers as `root`.
  - Use the `IPC_LOCK` capability to lock memory.
- The **device plugin** for RoCE-capable NICs (e.g., NVIDIA Network Operator or
  custom RDMA plugin) is installed and configured on the cluster.
- The **GPU worker** has the required drivers and libraries. In this example, we
  will deploy Ray on a Kubernetes cluster. Thus, our image will contain:
  - OFED modules
  - the NVIDIA and NCCL runtime binaries
- The **system administrator** has shared the number of GPUs and RoCE-capable
  NICs available per node to guide resource requests and topology mapping.
- The **Kubernetes administrator** has explained how to:
  - Request RoCE devices (e.g., `nvidia.com/roce_gdr: 2`)
  - Enable pods to access the RDMA-enabled network zones
  - Schedule GPU workers on the correct nodes e.g via labels, taints, affinity
    rules, etc

### Install the required libraries and drivers

This example walks you through deploying a RayCluster on Kubernetes, including
building a custom image for the GPU worker nodes. We’ll use the
`mirror.gcr.io/rayproject/ray:latest-py310-cu121` base image, which includes
both Ray and the necessary NVIDIA libraries.

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```docker
ARG base_image=mirror.gcr.io/rayproject/ray:latest-py310-cu121
FROM $base_image

USER 0

ENV MOFED_VER=24.10-1.1.4.0
ENV OS_VER=ubuntu22.04
ENV PLATFORM=x86_64

# See the Requirements section at the top of this document
RUN pip install --system torch==$VERSION_OF_TORCH_THAT_FMS_HF_TUNING_NEEDS

RUN mkdir app && \
    cd app && \
    wget -q http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    tar -xvzf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    cd MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM} && \
    ./mlnxofedinstall --user-space-only --without-fw-update --without-ucx-cuda --all --force --distro $OS_VER && \
    cd .. && \
    rm -rf MLNX* && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*
```

<!-- markdownlint-enable line-length -->

!!! note

    Ensure you install the appropriate version of
    [torch for the fms-hf-tuning versions that you are planning to use](#requirements).

<!-- markdownlint-enable line-length -->

!!! note

    Mellanox OFED is now in long-term support and will reach end-of-life in Q4 2027.
    NVIDIA has replaced it with DOCA-OFED, which will receive all future updates
    and features. This example currently uses MLNX_OFED, but we'll update it with
    DOCA-OFED installation steps in a future revision.

### Collect all necessary information

#### Identify RoCE-Capable Network Devices

To determine which network interfaces support RoCE v2, run the `show_gids`
command on a GPU node. Look for entries where the **`VER`** column is `v2`,
which indicates RoCE v2 support.

For example, given the following output:

<!-- markdownlint-disable-next-line code-block-style -->
```terminaloutput
DEV    PORT INDEX GID                                   IPv4         VER DEV
---    ---- ----- ---                                   ------------ --- ---
mlx5_0 1       0 fe80:0000:0000:0000:0000:60ff:fe68:d096              v1 net1-0
mlx5_0 1       1 fe80:0000:0000:0000:0000:60ff:fe68:d096              v2 net1-0
...
mlx5_3 1       1 fe80:0000:0000:0000:0000:5fff:fe68:d09a              v2 net1-1
```

You should select the devices with `v2` under the `VER` column. In this case,
the RoCE-capable devices are:

- `mlx5_0_1`
- `mlx5_3_1`

You will use these device names to set the `NCCL_IB_HCA` environment variable in
your Ray GPU worker pods. For the above example you will set
`NCCL_IB_HCA="=mlx5_0,mlx5_3"`

You also need to configure `NCCL_IB_GID_INDEX`. Select the GID index such that
it maps to a v2 entry across all nodes to ensure consistent behavior. For the
above example you will set `NCCL_IB_GID_INDEX=1`

### Putting it all together

In this section we will use the information we gathered above to define a Ray
GPU worker with support for RoCE.

#### Summary of Steps

1. **Enable memory locking in containers**
   - Request the `IPC_LOCK` capability in your container’s security context.
   - Use a `ServiceAccount` (e.g. `gdr`) that grants permission to request
     `IPC_LOCK`.
   - To allow unlimited memory locking:
     - **Option A:** Run the container as root (in the example we assume that
       the `roce` service account has adequate RBAC to request this).
     - **Option B:** Configure the node with `ulimit -l unlimited` (not
       available on Vela).
2. **Attach and request RoCE-capable NICs**
   - On our cluster:
     - We add the annotation: `k8s.v1.cni.cncf.io/networks: multi-nic-network`
     - Request RoCE devices: `nvidia.com/roce_gdr: 2`
3. **Set NCCL environment variables**
   - Configure variables like `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, and others to
     enable RoCE and optimize performance.
4. **Mount the NCCL topology file**
   - Mount the `topology-roce` ConfigMap at `/var/run/nvidia-topologyd`.

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```yaml
# ... trimmed ...
workerGroupSpecs:
  - rayStartParams:
    block: "true"
    num-gpus: "8"
    # VV: We'll use the RoCE custom resource to ensure the jobs land on a properly configured node for RoCE
    # we support running up to 1 RoCE measurement. Similarly, we have a custom resource called
    # "full-worker" for reserving the entire GPU worker if necessary.
    resources:
      "\"{\\\"NVIDIA-A100-SXM4-80GB\\\": 8, \\\"full-worker\\\": 1,
      \\\"RoCE\\\": 1}\""
# Here, we configure an eightou GPU worker with A100 that can have up to 4 replicas
replicas: 4
minReplicas: 4
maxReplicas: 4
numOfHosts: 1
groupName: eight-A100-80G-gpu-WG
template:
  metadata:
    annotations:
      # We use this annotation on our cluster to get access to the appropriate network zone
      k8s.v1.cni.cncf.io/networks: multi-nic-network
    labels:
      helm.sh/chart: ray-cluster-1.1.0
      app.kubernetes.io/instance: ray-disorch
  # ServiceAccount gives your pod adequate RBAC to request the IPC_LOCK capability and run as root
  serviceAccountName: roce
  # RoCE requires root privileges.
  # An alternative to using a root account is to request the capability CAP_SYS_RESOURCE and
  # run `ulimit -l unlimited` before starting up the Ray worker
  securityContext:
    fsGroup: 0
    runAsGroup: 0
    runAsNonRoot: false
    runAsUser: 0
  volumes:
    volumes:
      - name: topology-volume
        configMap:
          name: topology-roce
      - name: dshm
        emptyDir:
          medium: Memory
    # Add your remaining PVCs here e.g. a GPFS volume for storing the
    # HF_HOME path that you will use with "accelerate launch" etc
  containers:
    - name: pytorch
      image: $YOUR_ROCE_ENABLED_IMAGE_HERE
      imagePullPolicy: Always
      securityContext:
        allowPrivilegeEscalation: false
        capabilities:
          add:
            - IPC_LOCK # for RoCE to work
      env:
        # To enable RoCE
        - name: NCCL_IB_HCA
          value: =mlx5_0,mlx5_3
        - name: NCCL_IB_GID_INDEX
          value: "1"
        - name: NCCL_IB_DISABLE
          value: "0" # Set this to "1" to disable RoCE
        # To visually verify that RoCE is On based on the logs that NCCL prints
        - name: NCCL_DEBUG
          value: INFO
        - name: NCCL_DEBUG_SUBSYS
          value: "INIT,BOOTSTRAP,ENV"
        # Remaining NCCL environment variables we use on our cluster
        - name: NCCL_IB_QPS_PER_CONNECTION
          value: "8"
        - name: NCCL_IB_SPLIT_DATA_ON_QPS
          value: "0"
        - name: NCCL_IB_PCI_RELAXED_ORDERING
          value: "1"
        - name: NCCL_ALGO
          value: Ring
        - name: NCCL_IGNORE_CPU_AFFINITY
          value: "1"
        - name: NCCL_SOCKET_NTHREADS
          value: "2"
        - name: NCCL_CROSS_NIC
          value: "0"
        - name: OMP_NUM_THREADS
          value: "16"
  volumeMounts:
    - name: topology-volume
      mountPath: /var/run/nvidia-topologyd
    # Your other volumemounts here
    - name: dshm
      mountPath: "/dev/shm"
  resources:
    # Here we are requesting an entire node
    requests:
      cpu: 60
      nvidia.com/gpu: 8
      memory: 720Gi
      nvidia.com/roce_gdr: 2
    limits:
      cpu: 60
      nvidia.com/gpu: 8
      memory: 720Gi
      nvidia.com/roce_gdr: 2
```

<!-- markdownlint-enable line-length -->

!!! note

    We recommend enabling RoCE only for GPU workers that occupy an entire Kubernetes
    node. This ensures that multi-node jobs are using separate Kubernetes nodes,
    allowing RoCE to be effectively utilized.

#### Verify you're using RoCE

Remember, RoCE only applies to multi-node jobs. To verify that it’s working, run
a multi-node NCCL job and inspect the logs. If your GPU workers are properly
configured for RoCE, you should see output similar to the snippet below. We’ve
annotated the important lines with `<--` and added comments to highlight what to
look for.

    <!-- markdownlint-disable line-length -->
    [0] NCCL INFO NCCL_IB_DISABLE set by environment to 0. <-- double check that this is set to 0
    [4] NCCL INFO NCCL_IB_HCA set to mlx5_0,mlx5_3 <-- This does not confirm that you are using RoCE
    [3] NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [1]mlx5_3:1/RoCE [RO]; OOB net1-0:1.2.3.21<0> <-- Name of the NICs
                                                                                                    and /RoCE
    [3] NCCL INFO Using non-device net plugin version 0
    [3] NCCL INFO Using network IB <-- Uses the IB network
    <!-- markdownlint-enable line-length -->

NCCL falls back to "Socket" network when RoCE is unavailable. In this scenario
your log output will be similar to the snippet below.

    <!-- markdownlint-disable line-length -->
    [1] NCCL INFO NCCL_IB_DISABLE set by environment to 0. <-- if this is set to 1, you will NOT use RoCE even
                                                               if it is properly configured
    [1] NCCL INFO NCCL_SOCKET_IFNAME set by environment to net1-0,net1-1
    [1] NCCL INFO NCCL_IB_HCA set to mlx5_0,mlx5_3
    [1] NCCL INFO NET/IB : No device found. # <-- No network Infiniband network found
    [1] NCCL INFO NCCL_SOCKET_IFNAME set by environment to net1-0,net1-1
    [1] NCCL INFO NET/Socket : Using [0]net1-0:1.2.3.30<0> [1]net1-1:1.2.4.30<0> # <-- No mention of /RoCE
                                                                                       or the NICs
    [1] NCCL INFO Using non-device net plugin version 0
    [2] NCCL INFO Using network Socket # <-- Switches to TCP
    <!-- markdownlint-enable line-length -->

!!! note

    You might see warnings indicating that NCCL failed to load certain .so files.
    These messages are harmless and unrelated to RoCE configuration. You can ignore
    them safely.

## Next steps

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- ⚙️ **Customize Actuators using ActuatorConfiguration resources**

    ---

    Learn how to use **ActuatorConfiguration** resources to customize the SFTTrainer **Operations**

    [ActuatorConfiguration documentation](../resources/actuatorconfig.md)

- 🖥️ **Ready to try it out?**

    ---

    The SFTTrainer actuator can run experiments locally as well. Just follow the example below to get started:

    :link: [Run a local fine-tuning experiment](../examples/finetune-locally.md)

- :octicons-rocket-24:{ .lg .middle } **Take it to the next level**

    ---

    Do you have a RayCluster with GPUs in it?

    :link: [Run a fine-tuning experiment on a remote RayCluster](../examples/finetune-remotely.md)

</div>
<!-- markdownlint-enable line-length -->
