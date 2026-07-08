# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for execute_bfcl_benchmark command construction."""

from unittest.mock import MagicMock, patch

from ado_actuators.vllm_performance.vllm_performance_test.benchmark_models import (
    BenchmarkParameters,
)
from ado_actuators.vllm_performance.vllm_performance_test.execute_benchmark import (
    BFCL_DATASET_PATH,
    execute_bfcl_benchmark,
)


class TestExecuteBfclBenchmarkCommand:
    """Test that execute_bfcl_benchmark constructs the vllm bench serve command correctly."""

    def _captured_command(
        self, bfcl_categories: str = "simple,live_simple,multiple"
    ) -> list[str]:
        """Run execute_bfcl_benchmark with subprocess mocked; return the captured command."""
        mock_result = MagicMock()
        mock_result.duration = 10.0
        mock_result.completed = 100
        mock_result.total_input_tokens = 5000.0
        mock_result.total_output_tokens = 2000.0
        mock_result.request_throughput = 10.0
        mock_result.output_throughput = 200.0
        mock_result.total_token_throughput = 700.0
        for metric in [
            "mean_ttft_ms",
            "median_ttft_ms",
            "std_ttft_ms",
            "p25_ttft_ms",
            "p50_ttft_ms",
            "p75_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "std_tpot_ms",
            "p25_tpot_ms",
            "p50_tpot_ms",
            "p75_tpot_ms",
            "p99_tpot_ms",
            "mean_itl_ms",
            "median_itl_ms",
            "std_itl_ms",
            "p25_itl_ms",
            "p50_itl_ms",
            "p75_itl_ms",
            "p99_itl_ms",
            "mean_e2el_ms",
            "median_e2el_ms",
            "std_e2el_ms",
            "p25_e2el_ms",
            "p50_e2el_ms",
            "p75_e2el_ms",
            "p99_e2el_ms",
        ]:
            setattr(mock_result, metric, 0.0)

        captured: list[list[str]] = []

        def fake_check_call(cmd: list[str], **kwargs: object) -> None:  # noqa: ANN401
            captured.append(list(cmd))

        with (
            patch("subprocess.check_call", side_effect=fake_check_call),
            patch(
                "ado_actuators.vllm_performance.vllm_performance_test.execute_benchmark.get_results",
                return_value=mock_result,
            ),
        ):
            execute_bfcl_benchmark(
                base_url="http://localhost:8000",
                model="Qwen/Qwen3-30B-A3B-FP8",
                bfcl_categories=bfcl_categories,
                num_prompts=200,
            )

        assert len(captured) == 1, "subprocess.check_call should have been called once"
        return captured[0]

    def test_backend_is_openai_chat(self) -> None:
        """--backend openai-chat must be in the command when using BFCL dataset."""
        cmd = self._captured_command()
        assert "--backend" in cmd
        idx = cmd.index("--backend")
        assert cmd[idx + 1] == "openai-chat", (
            f"Expected 'openai-chat', got '{cmd[idx + 1]}'"
        )

    def test_dataset_name_is_hf(self) -> None:
        """--dataset-name hf must be in the command when using BFCL dataset."""
        cmd = self._captured_command()
        assert "--dataset-name" in cmd
        idx = cmd.index("--dataset-name")
        assert cmd[idx + 1] == "hf", f"Expected 'hf', got '{cmd[idx + 1]}'"

    def test_dataset_path_is_bfcl(self) -> None:
        """--dataset-path gorilla-llm/Berkeley-Function-Calling-Leaderboard must be present."""
        cmd = self._captured_command()
        assert "--dataset-path" in cmd
        idx = cmd.index("--dataset-path")
        assert cmd[idx + 1] == BFCL_DATASET_PATH, (
            f"Expected '{BFCL_DATASET_PATH}', got '{cmd[idx + 1]}'"
        )

    def test_endpoint_is_v1_chat_completions(self) -> None:
        """--endpoint /v1/chat/completions must be in the command."""
        cmd = self._captured_command()
        assert "--endpoint" in cmd
        idx = cmd.index("--endpoint")
        assert cmd[idx + 1] == "/v1/chat/completions", (
            f"Expected '/v1/chat/completions', got '{cmd[idx + 1]}'"
        )

    def test_bfcl_categories_single(self) -> None:
        """--bfcl-categories <value> is passed correctly for a single category."""
        cmd = self._captured_command(bfcl_categories="simple")
        assert "--bfcl-categories" in cmd
        idx = cmd.index("--bfcl-categories")
        assert cmd[idx + 1] == "simple", f"Expected 'simple', got '{cmd[idx + 1]}'"

    def test_bfcl_categories_multiple(self) -> None:
        """--bfcl-categories <value> is passed correctly for multiple categories."""
        cmd = self._captured_command(bfcl_categories="simple,live_simple,multiple")
        assert "--bfcl-categories" in cmd
        idx = cmd.index("--bfcl-categories")
        assert cmd[idx + 1] == "simple,live_simple,multiple", (
            f"Expected 'simple,live_simple,multiple', got '{cmd[idx + 1]}'"
        )

    def test_num_prompts_passed(self) -> None:
        """--num-prompts is passed to the command."""
        cmd = self._captured_command()
        assert "--num-prompts" in cmd
        idx = cmd.index("--num-prompts")
        assert cmd[idx + 1] == "200", f"Expected '200', got '{cmd[idx + 1]}'"


class TestBenchmarkParametersBfclCategories:
    """Test the bfcl_categories field on BenchmarkParameters."""

    def test_default_is_none(self) -> None:
        """bfcl_categories defaults to None."""
        params = BenchmarkParameters(model="some-model")
        assert params.bfcl_categories is None

    def test_set_bfcl_categories(self) -> None:
        """bfcl_categories can be set to a string."""
        params = BenchmarkParameters(
            model="some-model", bfcl_categories="simple,multiple"
        )
        assert params.bfcl_categories == "simple,multiple"

    def test_lifecycle_create_dump_restore(self) -> None:
        """BenchmarkParameters with bfcl_categories survives a create → dump → restore round-trip."""
        original = BenchmarkParameters(
            model="Qwen/Qwen3-30B-A3B-FP8",
            dataset="bfcl",
            bfcl_categories="simple,live_simple,multiple",
            num_prompts=200,
        )
        dumped = original.model_dump()
        restored = BenchmarkParameters.model_validate(dumped)
        assert restored.bfcl_categories == original.bfcl_categories
        assert restored.dataset == "bfcl"
        assert restored.num_prompts == 200
