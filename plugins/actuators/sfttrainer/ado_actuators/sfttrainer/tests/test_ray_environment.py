# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib

from ado_actuators.sfttrainer.ray_env.utils import (
    apply_exclude_package_rules,
    get_pinned_packages,
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


def test_get_pinned_packages_extra_packages(tmp_path: pathlib.Path) -> None:
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("torch==2.5.0\ntransformers==4.46.0\n", encoding="utf-8")

    packages = get_pinned_packages(
        str(req_file),
        extra_packages=["extra-pkg-1==0.1.0", "extra-pkg-2"],
    )

    assert packages == [
        "torch==2.5.0",
        "transformers==4.46.0",
        "extra-pkg-1==0.1.0",
        "extra-pkg-2",
    ]


def test_get_pinned_packages_extra_packages_replace_existing(
    tmp_path: pathlib.Path,
) -> None:
    req_file = tmp_path / "requirements.txt"
    req_file.write_text(
        "psutil==6.0.0\npsutil-extras==1.0\nnumpy>=1.0\npandas >= 2.0\n"
        "scipy!=1.0\nyarl\ntorch==2.5.0\nnvidia_ml_py==12.0\nPSUTIL==5.0\n",
        encoding="utf-8",
    )

    packages = get_pinned_packages(
        str(req_file),
        extra_packages=[
            "psutil==7.2.2",
            "numpy",
            "pandas<3",
            "scipy>1",
            "yarl",
            "nvidia-ml-py==13.0",
        ],
    )

    assert packages == [
        "psutil-extras==1.0",
        "torch==2.5.0",
        "psutil==7.2.2",
        "numpy",
        "pandas<3",
        "scipy>1",
        "yarl",
        "nvidia-ml-py==13.0",
    ]
