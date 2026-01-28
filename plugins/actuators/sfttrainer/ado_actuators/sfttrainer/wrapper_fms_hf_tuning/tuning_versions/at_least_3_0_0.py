# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import os
import typing

import tuning.config.configs
import tuning.sft_trainer

if typing.TYPE_CHECKING:
    import transformers.trainer_callback
    from tuning.sft_trainer import SFTTrainer


def parse_arguments_and_execute_wrapper(
    callbacks: list["transformers.trainer_callback.TrainerCallback"] | None,
    job_config: dict[str, typing.Any],
) -> tuple["SFTTrainer", dict]:

    parser = tuning.sft_trainer.get_parser()

    (
        model_args,
        data_args,
        training_args,
        trainer_controller_args,
        tune_config,
        quantized_lora_config,
        fusedops_kernels_config,
        attention_and_distributed_packing_config,
        fast_moe_config,
        tracker_configs,
        _exp_metadata,
    ) = tuning.sft_trainer.parse_arguments(parser, job_config)

    if not os.path.isdir(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    training_args = typing.cast(
        "tuning.config.configs.TrainingArguments", training_args
    )
    training_args.trackers = []

    metadata = {}

    return tuning.sft_trainer.train(
        model_args=model_args,
        data_args=data_args,
        train_args=training_args,
        peft_config=tune_config,
        trainer_controller_args=trainer_controller_args,
        tracker_configs=tracker_configs,
        additional_callbacks=callbacks,
        exp_metadata=metadata,
        quantized_lora_config=quantized_lora_config,
        fusedops_kernels_config=fusedops_kernels_config,
        attention_and_distributed_packing_config=attention_and_distributed_packing_config,
        fast_moe_config=fast_moe_config,
    )
