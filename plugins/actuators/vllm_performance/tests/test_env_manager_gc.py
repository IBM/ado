# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for EnvironmentManager garbage-collection of stale free environments."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from ado_actuators.vllm_performance.env_manager import (
    Environment,
    EnvironmentManager,
    EnvironmentState,
)

# @ray.remote wraps the class in an ActorClass. The original plain Python class
# is accessible via __ray_actor_class__ and is what we instantiate in tests.
_EnvironmentManagerClass = EnvironmentManager.__ray_actor_class__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DUMMY_CONFIGURATION = '{"model": "test-model", "image": "vllm:latest"}'
DUMMY_MODEL = "test-model"


def _make_manager(
    ttl: int = 300,
    gc_force_delete: bool = False,
    gc_force_delete_threshold: int = 3,
) -> _EnvironmentManagerClass:
    """Construct an EnvironmentManager (plain class) with K8s calls mocked out."""
    created_tasks: list = []

    def _capture_task(coro: object) -> None:
        # Close the coroutine immediately to suppress "never awaited" warnings.
        if hasattr(coro, "close"):
            coro.close()  # type: ignore[union-attr]

    with (
        patch("ado_actuators.vllm_performance.env_manager.ComponentsManager") as MockCM,
        patch("asyncio.get_event_loop") as mock_loop,
    ):
        MockCM.return_value = MagicMock()
        mock_event_loop = MagicMock()
        mock_event_loop.create_task.side_effect = _capture_task
        mock_loop.return_value = mock_event_loop
        manager = _EnvironmentManagerClass(
            namespace="test-ns",
            max_concurrent=3,
            free_environment_ttl=ttl,
            gc_force_delete=gc_force_delete,
            gc_force_delete_threshold=gc_force_delete_threshold,
        )
    _ = created_tasks  # unused but kept for clarity
    # Replace ComponentsManager with a fresh MagicMock after construction so
    # assertions on K8s calls are clean.
    manager.manager = MagicMock()
    return manager


def _free_env(manager: EnvironmentManager, age_seconds: float) -> Environment:
    """Add a pre-aged free environment to the manager's free list."""
    import time

    env = Environment(
        model=DUMMY_MODEL,
        configuration=DUMMY_CONFIGURATION,
        state=EnvironmentState.READY,
    )
    env.freed_at = time.monotonic() - age_seconds
    manager.free_environments.append(env)
    return env


# ---------------------------------------------------------------------------
# Environment model tests
# ---------------------------------------------------------------------------


class TestEnvironmentModel:
    def test_freed_at_is_none_by_default(self) -> None:
        """A freshly constructed Environment has freed_at=None."""
        env = Environment(model=DUMMY_MODEL, configuration=DUMMY_CONFIGURATION)
        assert env.freed_at is None

    def test_freed_at_can_be_set(self) -> None:
        """freed_at accepts a float timestamp."""
        env = Environment(model=DUMMY_MODEL, configuration=DUMMY_CONFIGURATION)
        env.freed_at = 12345.6
        assert env.freed_at == pytest.approx(12345.6)


# ---------------------------------------------------------------------------
# done_using tests
# ---------------------------------------------------------------------------


class TestDoneUsingGC:
    def test_ttl_zero_deletes_immediately(self) -> None:
        """TTL=0: done_using deletes K8s resources immediately, nothing added to free list,
        and env is tracked in deleting_environments for GC monitoring."""
        manager = _make_manager(ttl=0)
        env = Environment(
            model=DUMMY_MODEL,
            configuration=DUMMY_CONFIGURATION,
            state=EnvironmentState.READY,
        )
        manager.in_use_environments[env.k8s_name] = env

        manager.done_using(identifier=env.k8s_name)

        manager.manager.delete_service.assert_called_once()
        manager.manager.delete_deployment.assert_called_once()
        assert len(manager.free_environments) == 0
        assert env in manager.deleting_environments

    def test_ttl_positive_adds_to_free_list_with_timestamp(self) -> None:
        """TTL>0: done_using appends env to free list and sets freed_at."""
        import time

        manager = _make_manager(ttl=300)
        env = Environment(
            model=DUMMY_MODEL,
            configuration=DUMMY_CONFIGURATION,
            state=EnvironmentState.READY,
        )
        manager.in_use_environments[env.k8s_name] = env

        before = time.monotonic()
        manager.done_using(identifier=env.k8s_name)
        after = time.monotonic()

        assert len(manager.free_environments) == 1
        freed = manager.free_environments[0]
        assert freed.freed_at is not None
        assert before <= freed.freed_at <= after
        # K8s resources must NOT be deleted yet
        manager.manager.delete_service.assert_not_called()
        manager.manager.delete_deployment.assert_not_called()

    def test_return_to_pool_false_skips_free_list(self) -> None:
        """return_to_pool=False: env is removed without entering the free pool."""
        manager = _make_manager(ttl=300)
        env = Environment(
            model=DUMMY_MODEL,
            configuration=DUMMY_CONFIGURATION,
            state=EnvironmentState.READY,
        )
        manager.in_use_environments[env.k8s_name] = env

        manager.done_using(identifier=env.k8s_name, return_to_pool=False)

        assert len(manager.free_environments) == 0
        manager.manager.delete_service.assert_not_called()


# ---------------------------------------------------------------------------
# cleanup_failed_deployment tests
# ---------------------------------------------------------------------------


class TestCleanupFailedDeployment:
    def test_tracks_env_in_deleting_environments(self) -> None:
        """cleanup_failed_deployment adds the env to deleting_environments for GC monitoring."""
        manager = _make_manager(ttl=300)
        env = Environment(
            model=DUMMY_MODEL,
            configuration=DUMMY_CONFIGURATION,
            state=EnvironmentState.READY,
        )
        manager.in_use_environments[env.k8s_name] = env

        manager.cleanup_failed_deployment(identifier=env.k8s_name)

        manager.manager.delete_service.assert_called_once()
        manager.manager.delete_deployment.assert_called_once()
        assert env in manager.deleting_environments
        assert env.k8s_name not in manager.in_use_environments
        assert env not in manager.free_environments


# ---------------------------------------------------------------------------
# _gc_free_environments tests
# ---------------------------------------------------------------------------


class TestGcFreeEnvironments:
    def test_stale_environment_is_removed(self) -> None:
        """An environment older than TTL is deleted and removed from the free list."""
        manager = _make_manager(ttl=60)
        _free_env(manager, age_seconds=120)  # 120s idle > 60s TTL → stale

        manager._gc_free_environments()

        assert len(manager.free_environments) == 0
        manager.manager.delete_service.assert_called_once()
        manager.manager.delete_deployment.assert_called_once()

    def test_fresh_environment_is_kept(self) -> None:
        """An environment younger than TTL is NOT deleted."""
        manager = _make_manager(ttl=300)
        _free_env(manager, age_seconds=10)  # 10s idle < 300s TTL → fresh

        manager._gc_free_environments()

        assert len(manager.free_environments) == 1
        manager.manager.delete_service.assert_not_called()

    def test_only_stale_removed_when_mixed(self) -> None:
        """Only stale environments are removed; fresh ones are kept."""
        manager = _make_manager(ttl=60)
        stale = _free_env(manager, age_seconds=120)
        fresh = _free_env(manager, age_seconds=10)

        manager._gc_free_environments()

        assert len(manager.free_environments) == 1
        remaining = manager.free_environments[0]
        assert remaining is fresh
        assert stale not in manager.free_environments

    def test_k8s_error_keeps_in_free_list(self) -> None:
        """A K8s ApiException during GC keeps the entry in the free list for retry."""
        from kubernetes.client import ApiException

        manager = _make_manager(ttl=60)
        _free_env(manager, age_seconds=120)
        manager.manager.delete_service.side_effect = ApiException(status=500)

        manager._gc_free_environments()

        # K8s failure: entry is kept so the next GC cycle can retry deletion
        assert len(manager.free_environments) == 1

    def test_noop_when_free_list_empty(self) -> None:
        """_gc_free_environments does nothing on an empty free list."""
        manager = _make_manager(ttl=60)
        manager._gc_free_environments()
        manager.manager.delete_service.assert_not_called()

    def test_stale_env_moves_to_deleting_list(self) -> None:
        """A successfully deleted stale env is added to deleting_environments."""
        manager = _make_manager(ttl=60)
        env = _free_env(manager, age_seconds=120)

        manager._gc_free_environments()

        assert env in manager.deleting_environments

    def test_k8s_error_does_not_add_to_deleting_list(self) -> None:
        """A failed delete does not add the env to deleting_environments."""
        from kubernetes.client import ApiException

        manager = _make_manager(ttl=60)
        _free_env(manager, age_seconds=120)
        manager.manager.delete_service.side_effect = ApiException(status=500)

        manager._gc_free_environments()

        assert len(manager.deleting_environments) == 0


# ---------------------------------------------------------------------------
# _gc_confirm_deletions tests
# ---------------------------------------------------------------------------


class TestGcConfirmDeletions:
    def _add_deleting_env(self, manager: _EnvironmentManagerClass) -> Environment:
        """Add an env directly to the deleting list."""
        env = Environment(model=DUMMY_MODEL, configuration=DUMMY_CONFIGURATION)
        manager.deleting_environments.append(env)
        return env

    def test_confirmed_gone_is_removed(self) -> None:
        """An env no longer present in K8s is removed from deleting_environments."""
        manager = _make_manager(ttl=60)
        self._add_deleting_env(manager)
        manager.manager.check_deployment_exist.return_value = False

        manager._gc_confirm_deletions()

        assert len(manager.deleting_environments) == 0

    def test_still_present_increments_counter(self) -> None:
        """Each GC cycle where the env is still present increments delete_attempts."""
        manager = _make_manager(ttl=60)
        env = self._add_deleting_env(manager)
        manager.manager.check_deployment_exist.return_value = True

        manager._gc_confirm_deletions()

        assert env.delete_attempts == 1
        assert env in manager.deleting_environments

    def test_mixed_confirmed_and_pending(self) -> None:
        """Only confirmed-gone envs are removed; still-present ones are kept."""
        manager = _make_manager(ttl=60)
        gone = self._add_deleting_env(manager)
        still_there = self._add_deleting_env(manager)

        manager.manager.check_deployment_exist.side_effect = [
            False,  # gone
            True,  # still_there
        ]

        manager._gc_confirm_deletions()

        assert gone not in manager.deleting_environments
        assert still_there in manager.deleting_environments

    def test_noop_when_deleting_list_empty(self) -> None:
        """_gc_confirm_deletions is a no-op when deleting_environments is empty."""
        manager = _make_manager(ttl=60)
        manager._gc_confirm_deletions()
        manager.manager.check_deployment_exist.assert_not_called()

    def test_no_force_delete_env_kept_in_watch(self) -> None:
        """With gc_force_delete=False the env stays in the watch list indefinitely."""
        manager = _make_manager(ttl=60, gc_force_delete=False)
        env = self._add_deleting_env(manager)
        manager.manager.check_deployment_exist.return_value = True

        # Run several cycles past gc_force_delete_threshold — env must stay watched.
        for _ in range(5):
            manager._gc_confirm_deletions()

        manager.manager.force_delete_environment.assert_not_called()
        assert env in manager.deleting_environments

    def test_force_delete_at_threshold_when_enabled(self) -> None:
        """At gc_force_delete_threshold cycles with gc_force_delete=True, force-delete is issued and env is dropped."""
        threshold = 3
        manager = _make_manager(
            ttl=60, gc_force_delete=True, gc_force_delete_threshold=threshold
        )
        env = self._add_deleting_env(manager)
        env.delete_attempts = threshold - 1  # one more check will hit threshold
        manager.manager.check_deployment_exist.return_value = True

        manager._gc_confirm_deletions()

        assert env.delete_attempts == threshold
        manager.manager.force_delete_environment.assert_called_once_with(
            k8s_name=env.k8s_name
        )
        assert env not in manager.deleting_environments

    def test_force_delete_api_error_env_retained(self) -> None:
        """An ApiException during force-delete is logged; env is retained in watch list."""
        from kubernetes.client import ApiException

        threshold = 3
        manager = _make_manager(
            ttl=60, gc_force_delete=True, gc_force_delete_threshold=threshold
        )
        env = self._add_deleting_env(manager)
        env.delete_attempts = threshold - 1
        manager.manager.check_deployment_exist.return_value = True
        manager.manager.force_delete_environment.side_effect = ApiException(status=500)

        manager._gc_confirm_deletions()

        assert env in manager.deleting_environments


# ---------------------------------------------------------------------------
# _gc_loop tests
# ---------------------------------------------------------------------------


class TestGcLoop:
    @pytest.mark.asyncio
    async def test_loop_stops_when_stop_flag_set(self) -> None:
        """_gc_loop exits after _stop_gc is set to True."""
        manager = _make_manager(ttl=1)
        # Set stop flag before starting so the loop exits after the first sleep.
        manager._stop_gc = True
        # Should complete without hanging (poll_interval = min(1, 60) = 1s).
        await asyncio.wait_for(manager._gc_loop(), timeout=3.0)
