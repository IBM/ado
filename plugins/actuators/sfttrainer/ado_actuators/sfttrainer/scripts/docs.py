# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json

import ado_actuators.sfttrainer.experiments.common as common
import ado_actuators.sfttrainer.experiments.lora as lora

models_vanilla = []
models_gptq = []

weights_vanilla = []
weights_gptq = []

for name, collection in common.ModelMap.items():
    if common.WeightsFormat.Vanilla in collection:
        models_vanilla.append(name)
        weights = collection[common.WeightsFormat.Vanilla]

        if weights.startswith("/hf-models-pvc/"):
            weights_vanilla.append(weights[15:])

    if common.WeightsFormat.GPTQQuantized in collection:
        models_gptq.append(name)

        weights = collection[common.WeightsFormat.GPTQQuantized]

        if weights.startswith("/hf-models-pvc/"):
            weights_gptq.append(weights[15:])

print(f"- model_name: Supported models: `{json.dumps(models_vanilla)}`")
print(f"- model_name Supported models for GPTQ Lora: `{json.dumps(models_gptq)}`")


print("----------------")
print(
    "- The PVC `hf-models-pvc` mounted under `/hf-models-pvc` - should contain the models:"
)
for x in sorted(weights_vanilla):
    if not x.endswith("/"):
        x += "/"
    print("  -", x)

print(
    "- The PVC `hf-models-pvc` mounted under `/hf-models-pvc` - should contain the models:"
)
for x in sorted(weights_gptq):
    if not x.endswith("/"):
        x += "/"
    print("  -", x)


print("-------LORA---------")
models_lora = [k for k in models_vanilla if k in lora.default_target_modules]
print(f"- model_name: Supported models for LORA: `{json.dumps(models_lora)}`")
print(
    "  - Specifically for `target_modules` the value `default` maps to the following `target_modules` "
    "based on the `model_name`:"
)

for name, layers in lora.default_target_modules.items():
    print("    -", f"`{name}`: `{json.dumps(layers)}`")


print("-------fms-hf-tuning versions--------")

print(f"- Actuator version: `{common.ACTUATOR_VERSION}`")
print("- fms-hf-tuning versions:")
# - fms-hf-tuning versions:
#   - 2.1.0 (default)
#     - The full list of packages is at [packages/fms-hf-tuning_v2.1.0_8f168183a70b41cb66902f438ecba7734144138c.txt](packages/fms-hf-tuning_v2.1.0_8f168183a70b41cb66902f438ecba7734144138c.txt)
#   - 2.0.0
#     - The full list of packages is at [packages/fms-hf-tuning_v2.0.1_9b8245e74144f7ee73b7241a1687b6c77f0eb2e4.txt](packages/fms-hf-tuning_v2.0.1_9b8245e74144f7ee73b7241a1687b6c77f0eb2e4.txt)
for idx, (version, commit) in enumerate(reversed(common.FMS_HF_TUNING_COMMIT.items())):
    if idx == 0:
        print("  -", version, "(default)")
    else:
        print("  -", version)
    path = f"packages/fms-hf-tuning_v{version}_{commit}.txt"
    print(f"    - The full list of packages is at [{path}]({path})")
