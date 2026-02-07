# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

# pyright: reportCallIssue=false

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from autoconf.utils.pydantic_models import JobConfig
from autoconf.utils.recommender import (
    get_model_prediction_and_metadata,
    recommend_min_gpu,
)

# Example configurations

valid_config_dict = {
    "model_name": "granite-3.2-8b-instruct",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 8192.0,
    "batch_size": 16.0,
    "number_gpus": 2,
    "model_version": "2.0.0",
}

invalid_config_dict = {
    "model_name": "llama-13b",
    "method": "full",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 4096.0,
    "batch_size": 128.0,
    "number_gpus": 8,
    "model_version": "2.0.0",
}

# Convert to JobConfig instances

valid_job_config = JobConfig(**valid_config_dict)
invalid_job_config = JobConfig(**invalid_config_dict)

# Mock return values

mock_prediction_valid = [1]
mock_prediction_invalid = [0]
mock_rbc_valid = (1, [])
mock_rbc_invalid = (0, ["Invalid configuration"])


@pytest.fixture
def mock_predictor_valid() -> MagicMock:
    mock = MagicMock()

    def mock_predict(df: pd.DataFrame) -> pd.Series:
        # Simulate prediction logic based on gpus_per_worker
        print(df)
        val = 1 if int(df["number_gpus"].values[0]) == 2 else 0
        return pd.Series([val])  # Mimics .values[0] behavior

    mock.predict.side_effect = mock_predict
    return mock


@pytest.fixture
def mock_predictor_invalid() -> MagicMock:
    mock = MagicMock()

    def mock_predict(df: pd.DataFrame) -> pd.Series:
        val = -1
        return pd.Series([val])

    mock.predict.side_effect = mock_predict
    return mock


@patch(
    "autoconf.utils.rule_based_classifier.is_row_valid",
    return_value=mock_rbc_valid,
)
def test_get_model_prediction_and_metadata_valid(
    mock_is_row_valid: MagicMock, mock_predictor_valid: MagicMock
) -> None:
    print(valid_job_config)
    pred, metadata = get_model_prediction_and_metadata(
        valid_job_config, predictor=mock_predictor_valid
    )
    assert pred == 1
    assert isinstance(metadata, dict)
    assert metadata["Rule-Based Classifier error"] == ""


@patch(
    "autoconf.utils.rule_based_classifier.is_row_valid",
    return_value=mock_rbc_invalid,
)
def test_get_model_prediction_and_metadata_invalid(
    mock_is_row_valid: MagicMock, mock_predictor_valid: MagicMock
) -> None:
    pred, metadata = get_model_prediction_and_metadata(
        invalid_job_config, predictor=mock_predictor_valid
    )
    assert pred is None or pred == 0
    assert isinstance(metadata, dict)


@patch(
    "autoconf.utils.rule_based_classifier.is_row_valid",
    return_value=mock_rbc_valid,
)
def test_recommend_min_gpu_valid(
    mock_is_row_valid: MagicMock, mock_predictor_valid: MagicMock
) -> None:
    min_gpu, metadata = recommend_min_gpu(
        valid_job_config, valid_n_gpu_list=[1, 2, 4], predictor=mock_predictor_valid
    )
    assert min_gpu == 2
    assert isinstance(metadata, dict)


@patch(
    "autoconf.utils.rule_based_classifier.is_row_valid",
    return_value=mock_rbc_invalid,
)
def test_recommend_min_gpu_invalid(
    mock_is_row_valid: MagicMock, mock_predictor_invalid: MagicMock
) -> None:
    min_gpu, metadata = recommend_min_gpu(
        invalid_job_config, valid_n_gpu_list=[1, 2, 4], predictor=mock_predictor_invalid
    )
    assert min_gpu == -1
    assert isinstance(metadata, dict)


# Unit tests for min_gpu_recommender function (lines 161-187)


@pytest.fixture
def mock_load_model() -> MagicMock:
    """Mock the load_model function to return a mock predictor."""
    mock = MagicMock()
    mock_predictor = MagicMock()

    def mock_predict(df: pd.DataFrame) -> pd.Series:
        # Simple prediction: valid if number_gpus <= 4
        val = 1 if int(df["number_gpus"].values[0]) <= 4 else 0
        return pd.Series([val])

    mock_predictor.predict.side_effect = mock_predict
    mock.return_value = mock_predictor
    return mock


def test_min_gpu_recommender_valid_model_name_mapping_granite_3_1_2b() -> None:
    """Test that invalid granite 3.1 2B model names are mapped to valid ones."""
    from autoconf.min_gpu_recommender import min_gpu_recommender
    from autoconf.utils.config_mapper import mapped_models

    # Access the original unwrapped function
    original_func = getattr(min_gpu_recommender, "_original_func", min_gpu_recommender)

    # Test various granite 3.1 2B variants that should be mapped
    invalid_names = [
        "granite-2b",
        "granite-2b-base",
        "granite-2b-instruct",
        "granite-3.2-2b",
    ]

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = pd.Series([1])
        mock_load.return_value = mock_predictor

        with patch(
            "autoconf.utils.rule_based_classifier.is_row_valid", return_value=(1, [])
        ):
            for invalid_name in invalid_names:
                result = original_func(  # type: ignore[call-arg]
                    model_name=invalid_name,
                    method="lora",
                    gpu_model="NVIDIA-A100-80GB-PCIe",
                    tokens_per_sample=2048,
                    batch_size=8,
                    model_version="3.1.0",
                    gpus_per_worker=8,
                    max_gpus=8,
                )

                # Should succeed because name gets mapped
                assert (
                    result["can_recommend"] is True
                ), f"Failed for model: {invalid_name}"

                # Verify the predictor was called with mapped name
                call_args = mock_predictor.predict.call_args[0][0]
                assert (
                    call_args["model_name"].values[0] == mapped_models["GRANITE_3_1_2B"]
                )


def test_min_gpu_recommender_valid_model_name_mapping_granite_3_1_8b() -> None:
    """Test that invalid granite 3.1 8B model names are mapped to valid ones."""
    from autoconf.min_gpu_recommender import min_gpu_recommender
    from autoconf.utils.config_mapper import mapped_models

    original_func = getattr(min_gpu_recommender, "_original_func", min_gpu_recommender)

    invalid_names = [
        "granite-8b",
        "granite-8b-instruct",
        "granite-3.2-8b-instruct",
    ]

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = pd.Series([1])
        mock_load.return_value = mock_predictor

        with patch(
            "autoconf.utils.rule_based_classifier.is_row_valid", return_value=(1, [])
        ):
            for invalid_name in invalid_names:
                result = original_func(  # type: ignore[call-arg]
                    model_name=invalid_name,
                    method="lora",
                    gpu_model="NVIDIA-A100-80GB-PCIe",
                    tokens_per_sample=2048,
                    batch_size=8,
                    model_version="3.1.0",
                )

                assert (
                    result["can_recommend"] is True
                ), f"Failed for model: {invalid_name}"
                call_args = mock_predictor.predict.call_args[0][0]
                assert (
                    call_args["model_name"].values[0] == mapped_models["GRANITE_3_1_8B"]
                )


def test_min_gpu_recommender_valid_model_name_mapping_llama_3_1_8b() -> None:
    """Test that invalid llama 3.1 8B model names are mapped to valid ones."""
    from autoconf.min_gpu_recommender import min_gpu_recommender
    from autoconf.utils.config_mapper import mapped_models

    original_func = getattr(min_gpu_recommender, "_original_func", min_gpu_recommender)

    invalid_names = [
        "llama-3.1-8b-instruct",
        "llama-3.1-8b-base",
    ]

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = pd.Series([1])
        mock_load.return_value = mock_predictor

        with patch(
            "autoconf.utils.rule_based_classifier.is_row_valid", return_value=(1, [])
        ):
            for invalid_name in invalid_names:
                result = original_func(  # type: ignore[call-arg]
                    model_name=invalid_name,
                    method="lora",
                    gpu_model="NVIDIA-A100-80GB-PCIe",
                    tokens_per_sample=2048,
                    batch_size=8,
                    model_version="3.1.0",
                )

                assert (
                    result["can_recommend"] is True
                ), f"Failed for model: {invalid_name}"
                call_args = mock_predictor.predict.call_args[0][0]
                assert (
                    call_args["model_name"].values[0] == mapped_models["LLAMA_3_1_8B"]
                )


def test_min_gpu_recommender_valid_model_name_mapping_granite_4_variants() -> None:
    """Test that invalid granite 4.0 model names are mapped to valid ones."""
    from autoconf.min_gpu_recommender import min_gpu_recommender
    from autoconf.utils.config_mapper import mapped_models

    original_func = getattr(min_gpu_recommender, "_original_func", min_gpu_recommender)

    test_cases = [
        ("granite-4.0-small", "GRANITE_4_SMALL"),
        ("granite-4.0-tiny", "GRANITE_4_TINY"),
        ("granite-4.0-micro", "GRANITE_4_MICRO"),
    ]

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = pd.Series([1])
        mock_load.return_value = mock_predictor

        with patch(
            "autoconf.utils.rule_based_classifier.is_row_valid", return_value=(1, [])
        ):
            for invalid_name, expected_key in test_cases:
                result = original_func(  # type: ignore[call-arg]
                    model_name=invalid_name,
                    method="lora",
                    gpu_model="NVIDIA-A100-80GB-PCIe",
                    tokens_per_sample=2048,
                    batch_size=8,
                    model_version="3.1.0",
                )

                assert (
                    result["can_recommend"] is True
                ), f"Failed for model: {invalid_name}"
                call_args = mock_predictor.predict.call_args[0][0]
                assert call_args["model_name"].values[0] == mapped_models[expected_key]


def test_min_gpu_recommender_unmapped_model_name_unchanged() -> None:
    """Test that model names without mapping patterns remain unchanged."""
    from autoconf.min_gpu_recommender import min_gpu_recommender

    original_func = getattr(min_gpu_recommender, "_original_func", min_gpu_recommender)

    unmapped_names = [
        "custom-model-123",
        "gpt-4",
        "unknown-model",
    ]

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = pd.Series([1])
        mock_load.return_value = mock_predictor

        with patch(
            "autoconf.utils.rule_based_classifier.is_row_valid", return_value=(1, [])
        ):
            for unmapped_name in unmapped_names:
                result = original_func(  # type: ignore[call-arg]
                    model_name=unmapped_name,
                    method="lora",
                    gpu_model="NVIDIA-A100-80GB-PCIe",
                    tokens_per_sample=2048,
                    batch_size=8,
                    model_version="3.1.0",
                )

                # Should still work, name just not mapped
                assert (
                    result["can_recommend"] is True
                ), f"Failed for model: {unmapped_name}"
                call_args = mock_predictor.predict.call_args[0][0]
                # Name should remain unchanged
                assert call_args["model_name"].values[0] == unmapped_name


def test_min_gpu_recommender_successful_recommendation() -> None:
    """Test successful GPU recommendation with valid inputs."""
    from autoconf.min_gpu_recommender import min_gpu_recommender

    original_func = getattr(min_gpu_recommender, "_original_func", min_gpu_recommender)

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        mock_predictor = MagicMock()

        def mock_predict(df: pd.DataFrame) -> pd.Series:
            # Return valid for 2 GPUs
            val = 1 if int(df["number_gpus"].values[0]) == 2 else 0
            return pd.Series([val])

        mock_predictor.predict.side_effect = mock_predict
        mock_load.return_value = mock_predictor

        with patch(
            "autoconf.utils.rule_based_classifier.is_row_valid", return_value=(1, [])
        ):
            result = original_func(  # type: ignore[call-arg]
                model_name="granite-3.1-8b-instruct",
                method="lora",
                gpu_model="NVIDIA-A100-80GB-PCIe",
                tokens_per_sample=2048,
                batch_size=8,
                model_version="3.1.0",
                gpus_per_worker=8,
                max_gpus=8,
            )

            assert result["can_recommend"] is True
            assert "gpus" in result
            assert "workers" in result
            assert result["gpus"] > 0
            assert result["workers"] > 0


def test_min_gpu_recommender_no_recommendation_error() -> None:
    """Test handling when no valid GPU configuration is found."""
    from autoconf.min_gpu_recommender import min_gpu_recommender

    original_func = getattr(min_gpu_recommender, "_original_func", min_gpu_recommender)

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        mock_predictor = MagicMock()
        # Always return invalid prediction
        mock_predictor.predict.return_value = pd.Series([0])
        mock_load.return_value = mock_predictor

        with patch(
            "autoconf.utils.rule_based_classifier.is_row_valid", return_value=(1, [])
        ):
            result = original_func(  # type: ignore[call-arg]
                model_name="granite-3.1-8b-instruct",
                method="full",
                gpu_model="NVIDIA-A100-80GB-PCIe",
                tokens_per_sample=8192,
                batch_size=128,
                model_version="3.1.0",
                gpus_per_worker=8,
                max_gpus=8,
            )

            assert result["can_recommend"] is False
            assert "gpus" not in result
            assert "workers" not in result


def test_min_gpu_recommender_validation_error() -> None:
    """Test handling of validation errors (e.g., invalid parameter values)."""
    from autoconf.min_gpu_recommender import min_gpu_recommender

    original_func = getattr(min_gpu_recommender, "_original_func", min_gpu_recommender)

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        mock_predictor = MagicMock()
        mock_load.return_value = mock_predictor

        # Test with invalid tokens_per_sample (negative value)
        result = original_func(  # type: ignore[call-arg]
            model_name="granite-3.1-8b-instruct",
            method="lora",
            gpu_model="NVIDIA-A100-80GB-PCIe",
            tokens_per_sample=-100,  # Invalid: negative
            batch_size=8,
            model_version="3.1.0",
        )

        assert result["can_recommend"] is False


def test_min_gpu_recommender_default_parameters() -> None:
    """Test that default parameters are used correctly."""
    from autoconf.min_gpu_recommender import min_gpu_recommender

    original_func = getattr(min_gpu_recommender, "_original_func", min_gpu_recommender)

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        mock_predictor = MagicMock()

        def mock_predict(df: pd.DataFrame) -> pd.Series:
            return pd.Series([1])

        mock_predictor.predict.side_effect = mock_predict
        mock_load.return_value = mock_predictor

        with patch(
            "autoconf.utils.rule_based_classifier.is_row_valid", return_value=(1, [])
        ):
            # Call without optional parameters
            result = original_func(  # type: ignore[call-arg]
                model_name="granite-3.1-8b-instruct",
                method="lora",
                gpu_model="NVIDIA-A100-80GB-PCIe",
                tokens_per_sample=2048,
                batch_size=8,
                model_version="3.1.0",
                # gpus_per_worker defaults to 8
                # max_gpus defaults to 8
            )

            assert result["can_recommend"] is True


def test_min_gpu_recommender_model_version_validation() -> None:
    """Test that different model versions are handled correctly."""
    from autoconf.min_gpu_recommender import min_gpu_recommender

    original_func = getattr(min_gpu_recommender, "_original_func", min_gpu_recommender)

    valid_versions = ["3.0.0", "3.1.0"]

    for version in valid_versions:
        with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
            mock_predictor = MagicMock()
            mock_predictor.predict.return_value = pd.Series([1])
            mock_load.return_value = mock_predictor

            with patch(
                "autoconf.utils.rule_based_classifier.is_row_valid",
                return_value=(1, []),
            ):
                result = original_func(  # type: ignore[call-arg]
                    model_name="granite-3.1-8b-instruct",
                    method="lora",
                    gpu_model="NVIDIA-A100-80GB-PCIe",
                    tokens_per_sample=2048,
                    batch_size=8,
                    model_version=version,
                )

                assert result["can_recommend"] is True
                mock_load.assert_called_once_with(model_version=version)


def test_min_gpu_recommender_gpu_worker_calculation() -> None:
    """Test GPU and worker calculation logic."""
    from autoconf.min_gpu_recommender import min_gpu_recommender

    original_func = getattr(min_gpu_recommender, "_original_func", min_gpu_recommender)

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        mock_predictor = MagicMock()

        def mock_predict(df: pd.DataFrame) -> pd.Series:
            # Return valid for 4 GPUs
            val = 1 if int(df["number_gpus"].values[0]) == 4 else 0
            return pd.Series([val])

        mock_predictor.predict.side_effect = mock_predict
        mock_load.return_value = mock_predictor

        with patch(
            "autoconf.utils.rule_based_classifier.is_row_valid", return_value=(1, [])
        ):
            result = original_func(  # type: ignore[call-arg]
                model_name="granite-3.1-8b-instruct",
                method="lora",
                gpu_model="NVIDIA-A100-80GB-PCIe",
                tokens_per_sample=2048,
                batch_size=8,
                model_version="3.1.0",
                gpus_per_worker=8,
                max_gpus=8,
            )

            assert result["can_recommend"] is True
            # With min_gpus=4 and gpus_per_worker=8:
            # workers = ceil(4/8) = 1
            # gpus = ceil(4/1) = 4
            assert result["workers"] == 1
            assert result["gpus"] == 4


# Unit tests for avoid_oom_recommender function (lines 279-328)


def test_avoid_oom_recommender_predictor_is_none() -> None:
    """Test that avoid_oom_recommender handles when predictor is None."""
    from autoconf.min_gpu_recommender import avoid_oom_recommender

    original_func = getattr(
        avoid_oom_recommender, "_original_func", avoid_oom_recommender
    )

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        # Mock load_model to return None
        mock_load.return_value = None

        result = original_func(  # type: ignore[call-arg]
            model_name="granite-3.1-8b-instruct",
            method="lora",
            gpu_model="NVIDIA-A100-80GB-PCIe",
            tokens_per_sample=2048,
            per_device_train_batch_size=8,
            gpus_per_worker=8,
            max_gpus=64,
        )

        # Should return can_recommend=False when predictor is None
        assert result["can_recommend"] is False
        assert result["gpus"] == -1
        assert result["workers"] == -1


def test_avoid_oom_recommender_no_valid_gpu_config() -> None:
    """Test that avoid_oom_recommender sets can_recommend to False when recommend_min_gpu returns no valid configuration."""
    from autoconf.min_gpu_recommender import avoid_oom_recommender

    original_func = getattr(
        avoid_oom_recommender, "_original_func", avoid_oom_recommender
    )

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        mock_predictor = MagicMock()
        mock_load.return_value = mock_predictor

        with patch("autoconf.min_gpu_recommender.recommend_min_gpu") as mock_recommend:
            # Mock recommend_min_gpu to always return -1 (no valid configuration)
            mock_recommend.return_value = (-1, {"error": "No valid configuration"})

            result = original_func(  # type: ignore[call-arg]
                model_name="granite-3.1-8b-instruct",
                method="full",
                gpu_model="NVIDIA-A100-80GB-PCIe",
                tokens_per_sample=8192,
                per_device_train_batch_size=128,
                gpus_per_worker=8,
                max_gpus=64,
            )

            # Verify that can_recommend is False
            assert result["can_recommend"] is False
            assert result["gpus"] == -1
            assert result["workers"] == -1


def test_avoid_oom_recommender_successful_recommendation_4_gpus() -> None:
    """Test that avoid_oom_recommender returns correct values when recommend_min_gpu returns 4 GPUs."""
    from autoconf.min_gpu_recommender import avoid_oom_recommender

    original_func = getattr(
        avoid_oom_recommender, "_original_func", avoid_oom_recommender
    )

    with patch("autoconf.min_gpu_recommender.load_model") as mock_load:
        mock_predictor = MagicMock()
        mock_load.return_value = mock_predictor

        with patch("autoconf.min_gpu_recommender.recommend_min_gpu") as mock_recommend:
            # Mock recommend_min_gpu to return 4 GPUs for r_num_gpu=4
            def side_effect_recommend(
                job_config: JobConfig,
                predictor: MagicMock,
                valid_n_gpu_list: list[int],
            ) -> tuple[int, dict[str, bool | str]]:
                # Return 4 if the valid list contains 4
                if 4 in valid_n_gpu_list:
                    return (4, {"success": True})
                return (-1, {"error": "No valid configuration"})

            mock_recommend.side_effect = side_effect_recommend

            result = original_func(  # type: ignore[call-arg]
                model_name="granite-3.1-8b-instruct",
                method="lora",
                gpu_model="NVIDIA-A100-80GB-PCIe",
                tokens_per_sample=2048,
                per_device_train_batch_size=8,
                gpus_per_worker=4,
                max_gpus=64,
            )

            # Verify that can_recommend is True
            assert result["can_recommend"] is True
            # With min_gpus=4 and gpus_per_worker=4:
            # workers = ceil(4/4) = 1
            # gpus = ceil(4/1) = 4
            assert result["gpus"] == 4
            assert result["workers"] == 1
