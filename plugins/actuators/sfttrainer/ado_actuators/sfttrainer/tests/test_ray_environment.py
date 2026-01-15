# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from ado_actuators.sfttrainer.ray_env.utils import (
    apply_exclude_package_rules,
    packages_requiring_nvidia_development_binaries,
)


def test_exclude_packages() -> None:
    packages = [
        "yarl==1.20.0",
        "bitsandbytes==0.43.3",
        "causal-conv1d==1.5.0.post8",
        "fms-hf-tuning @ file:///wheel.whl",
        "nvidia-cublas-cu12==12.1.3.1",
        "flash_attn==2.7.4.post1",
    ]
    exclude_packages = [
        *packages_requiring_nvidia_development_binaries(),
        "bitsandbytes",
        "fms-hf-tuning",
    ]
    filtered, dropped = apply_exclude_package_rules(exclude_packages, packages)

    assert filtered == ["yarl==1.20.0"]
    assert dropped == [
        "bitsandbytes==0.43.3",
        "causal-conv1d==1.5.0.post8",
        "fms-hf-tuning @ file:///wheel.whl",
        "nvidia-cublas-cu12==12.1.3.1",
        "flash_attn==2.7.4.post1",
    ]
