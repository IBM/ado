# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import contextlib
import logging
import os
import threading
import typing

import ray._private.runtime_env.packaging
from ray._private.runtime_env import virtualenv_utils
from ray._private.runtime_env.pip import PipPlugin
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.validation import parse_and_validate_pip
from ray.runtime_env.runtime_env import RuntimeEnv

if typing.TYPE_CHECKING:
    from ray._private.runtime_env.context import RuntimeEnvContext

default_logger = logging.getLogger(__name__)

# VV: The function virtualenv_utils.create_or_get_virtualenv function raises an
# exception if the virtual environment already exists. We're just patching it
# so that ordered_pip.phases[1+] can reuse the venv of ordered_pip.phases[0]
original_create_or_get_virtualenv = virtualenv_utils.create_or_get_virtualenv


async def create_or_get_virtualenv(path: str, cwd: str, logger: logging.Logger) -> None:
    virtualenv_path = os.path.join(path, "virtualenv")
    if not os.path.exists(virtualenv_path):
        await original_create_or_get_virtualenv(path=path, cwd=cwd, logger=logger)


_monkey_patch_lock = threading.RLock()


@contextlib.contextmanager
def patch_create_or_get_virtualenv(
    phase_index: int,
) -> typing.Generator[None, None, None]:
    with _monkey_patch_lock:
        if phase_index > 0:
            virtualenv_utils.create_or_get_virtualenv = create_or_get_virtualenv
        try:
            yield
        finally:
            virtualenv_utils.create_or_get_virtualenv = (
                original_create_or_get_virtualenv
            )


class OrderedPipPlugin(RuntimeEnvPlugin):
    """A RuntimeEnvPlugin that enables you to guide the build order of packages.
    This is useful for installing packages that required other packages to already be installed in the
    virtual environment.

    An example would be `mamba-ssm` which depends on `torch` during its build phase.

    You can use the `ordered_pip` RuntimeEnvPlugin like so:

    First,

    export RAY_RUNTIME_ENV_PLUGINS='[{"class":"orchestrator.utilities.ray_env.ordered_pip.OrderedPipPlugin"}]'

    This way Ray will dynamically load this plugin.

    Then, start your python script:

    ```
    @ray.remote(
        runtime_env={
            "ordered_pip": {
                "phases": [
                    # The rules for entries under `ordered_pip.phases` is that they must match the
                    # `pip` schema (e.g. a list of a packages, or a `pip` dictionary, etc)
                    ["torch==2.6.0"],
                    {
                        "packages": ["mamba-ssm==2.2.5"],

                        # --no-build-isolation is important here. This will instruct pip
                        # to build the wheel in the same virtual environment that it's installing it in.
                        # In this case, because we installed `torch==2.6.0` in the previous phase,
                        # the pip install for mamba-ssm will succeed
                        "pip_install_options": ["--no-build-isolation"],
                    }
                ]
            }
        }
    )
    def try_import_torch():
        import torch

        print(torch.__file__)
        assert torch.__version__ == "2.6.0"
        return True

    assert ray.get(try_import_torch.remote())
    ```
    """

    name = "ordered_pip"

    # VV: Configure Ray to use this RuntimeEnvPlugin last
    priority = 100
    ClassPath = "orchestrator.utilities.ray_env.ordered_pip.OrderedPipPlugin"

    def __init__(self, resources_dir: str | None = None) -> None:
        self._global_mtx = threading.RLock()
        self._create_env_mtx: dict[str, threading.RLock] = {}
        self._pip_resources_dir = resources_dir

        # VV: Maintains a cache of the environments that have been built thus far
        self._cache = {}

    def _try_switch_resources_dir_from_context(
        self,
        context: "RuntimeEnvContext",  # noqa: F821
        logger: logging.Logger | None = default_logger,
    ) -> None:
        # VV: When ray instantiates custom RuntimeEnvPlugins it does not provide a resources_dir path.
        # This method is a HACK that the resources_dir based on the RuntimeEnvContext which is known
        # at the time of CREATING a virtual environment i.e. **after** the RuntimeEnvPlugin is initialized.

        with self._global_mtx:
            # VV: Stick with whatever resources dir we've already picked
            if self._pip_resources_dir:
                return

            logger.info("Generating resources dir")
            unique = set()
            if "PYTHONPATH" in context.env_vars:
                # VV: This is a HACK to find the "runtime_resources" path inside the PYTHONPATH env-var
                # This is an env-var that the WorkingDirPlugin inserts.
                # I noticed that sometimes the PYTHONPATH contains multiple copies of the same PATH.
                # The PYTHONPATH looks like this:
                # /tmp/ray/session_$timestamp/runtime_resources/working_dir_files/_ray_pkg_$uid
                many = context.env_vars["PYTHONPATH"].split(os.pathsep)
                logger.info(f"Current PYTHONPATH {many}")
                runtime_resources_followup = f"{os.sep}working_dir_files{os.sep}"
                unique.update(
                    [
                        os.path.join(
                            x.split(runtime_resources_followup, 1)[0], "ordered_pip"
                        )
                        for x in many
                        if runtime_resources_followup in x
                    ]
                )

            logger.info(f"The candidate locations of runtime_resources: {list(unique)}")

            if len(unique) != 1:
                import tempfile

                unique.clear()
                unique.add(tempfile.mkdtemp(prefix="ordered_pip_", dir="/tmp/ray"))

            self._switch_resources_dir(unique.pop())

    def _switch_resources_dir(self, resources_dir: str) -> None:
        with self._global_mtx:
            from ray._common.utils import try_to_create_directory

            self._pip_resources_dir = resources_dir
            try_to_create_directory(self._pip_resources_dir)

    @property
    def _pip_plugin(self) -> PipPlugin:
        # The PipPlugin keeps an internal cache of virtual environments it has created but not yet deleted.
        # When .create() is called, it checks this cache for a venv matching the given URI.
        # If a match is found, it assumes the venv already exists and skips re-creation.
        # However, ordered_pip needs to reuse the same venv multiple times (once per "phase").
        # Thus, we create a new PipPlugin instance on demand for each phase of ordered_pip.
        # Also, we maintain our own record of venvs to decide whether to create a new "ordered_pip"
        # venv or reuse an existing one.
        return PipPlugin(self._pip_resources_dir)

    @staticmethod
    def validate(runtime_env_dict: dict[str, typing.Any]) -> RuntimeEnv:
        """Validate user entry for this plugin.

        The method is invoked upon installation of runtime env.

        Args:
            runtime_env_dict: The user-supplied runtime environment dict.

        Raises:
            ValueError: If the validation fails.
        """

        if not isinstance(runtime_env_dict, dict):
            raise ValueError("runtime_env must be a dictionary")

        if "ordered_pip" not in runtime_env_dict:
            return RuntimeEnv(**runtime_env_dict)

        if not isinstance(runtime_env_dict["ordered_pip"], dict):
            raise ValueError("runtime_env['ordered_pip'] must be a dictionary")

        if not isinstance(runtime_env_dict["ordered_pip"]["phases"], list):
            raise ValueError(
                "runtime_env['ordered_pip']['phases'] must be an array of pip entries"
            )

        phases = []

        for i, p in enumerate(runtime_env_dict["ordered_pip"]["phases"]):
            try:
                validated = parse_and_validate_pip(p)
                if len(validated["packages"]) == 0:
                    continue
                phases.append(validated)
            except ValueError as e:
                raise ValueError(
                    f"runtime_env['ordered_pip']['phases'][{i}] must be consistent with the pip validation rules but "
                    f"validation failed with error {e}"
                ) from e

        # VV: keep extra fields - these are not directly used by ordered_pip but
        # may help developers troubleshoot issues
        others = {k: v for k, v in runtime_env_dict.items() if k != "ordered_pip"}
        result = RuntimeEnv(ordered_pip={"phases": phases}, **others)
        logging.info(
            f"Rewrote runtime_env `ordered_pip` field from {runtime_env_dict} to {result}."
        )

        return result

    def get_uris(self, runtime_env: "RuntimeEnv") -> list[str]:
        if not self.is_ordered_pip_runtimeenv(runtime_env):
            return []

        # VV: We want the hash to be invariant to the order of package names within a phase,
        # and we also want the order of phases to be reflected in the hash.
        aggregate_packages = [
            # VV: Ensure that each entry is expanded to the full pip spec
            sorted(p.get("packages", []))
            for p in self.validate(runtime_env)["ordered_pip"]["phases"]
        ]

        import hashlib

        return [
            "pip://"
            + hashlib.sha1(
                str(aggregate_packages).encode("utf-8"), usedforsecurity=False
            ).hexdigest()
        ]

    def is_ordered_pip_runtimeenv(self, runtime_env: "RuntimeEnv") -> bool:
        return bool(self.validate(runtime_env).get("ordered_pip"))

    async def create(
        self,
        uri: str,
        runtime_env: "RuntimeEnv",  # noqa: F821
        context: "RuntimeEnvContext",  # noqa: F821
        logger: logging.Logger | None = default_logger,
    ) -> int:
        self._try_switch_resources_dir_from_context(context, logger)

        if not self.is_ordered_pip_runtimeenv(runtime_env):
            return 0

        uri = self.get_uris(runtime_env)[0]

        with self._global_mtx:
            if uri not in self._create_env_mtx:
                self._create_env_mtx[uri] = threading.RLock()

        with self._create_env_mtx[uri]:
            logger.info(f"Creating {uri} for {runtime_env}")
            try:
                if os.path.isdir(self.get_path_to_pip_venv(uri)):
                    logger.info(f"Virtual environment for {uri} already exists")
                    return self._cache[uri]
            except KeyError:
                pass

            self._cache[uri] = 0
            for idx, pip in enumerate(
                self.validate(runtime_env)["ordered_pip"]["phases"]
            ):
                with patch_create_or_get_virtualenv(idx):
                    logger.info(f"Creating {idx} for {uri}")

                    self._cache[uri] += await self._pip_plugin.create(
                        uri=uri,
                        runtime_env=RuntimeEnv(pip=pip),
                        context=context,
                        logger=logger,
                    )
                    logger.info(f"Done creating {idx} for {uri}")

        return self._cache[uri]

    def get_path_to_pip_venv(self, uri: str) -> str:
        _, env_hash = ray._private.runtime_env.packaging.parse_uri(uri)
        return os.path.join(self._pip_resources_dir, "pip", env_hash)

    def delete_uri(
        self, uri: str, logger: logging.Logger | None = default_logger
    ) -> int:
        logger.info(f"Cleaning up {uri}")
        del self._cache[uri]

        import shutil

        import ray._private.utils

        env_dir = self.get_path_to_pip_venv(uri)
        num_bytes = ray._private.utils.get_directory_size_bytes(env_dir)

        try:
            shutil.rmtree(env_dir)
        except Exception as e:
            logger.warning(f"Exception while cleaning up {env_dir} {e!s} - will ignore")

        return num_bytes

    def modify_context(
        self,
        uris: list[str],
        runtime_env: "RuntimeEnv",  # noqa: F821
        context: "RuntimeEnvContext",  # noqa: F821
        logger: logging.Logger = default_logger,
    ) -> None:
        self._try_switch_resources_dir_from_context(context)

        runtime_env = self.validate(runtime_env)
        if not runtime_env.get("ordered_pip"):
            return

        logger.info(f"Modifying the context for {uris} and {runtime_env}")
        phases = runtime_env["ordered_pip"]["phases"]

        if not len(phases):
            return

        env_dir = self.get_path_to_pip_venv(uris[0])

        if not os.path.isdir(env_dir):
            logger.warning(
                f"The pip environment at {env_dir} has been garbage collected - recreating it"
            )
            import asyncio

            asyncio.run(
                self.create(
                    uris[0], runtime_env=runtime_env, context=context, logger=logger
                )
            )

        self._pip_plugin.modify_context(
            uris=uris,
            runtime_env=RuntimeEnv(pip=phases[0]),
            context=context,
            logger=logger,
        )

        if "PYTHONPATH" in context.env_vars:
            # VV: Ensure unique paths in PYTHONPATH
            paths = context.env_vars["PYTHONPATH"].split(os.pathsep)

            unique = []
            for k in paths:
                if k not in unique:
                    unique.append(k)

            context.env_vars["PYTHONPATH"] = os.pathsep.join(unique)

        logger.info(
            f"Modified the context for {uris} and {runtime_env} with {context.py_executable} {context.env_vars}"
        )
