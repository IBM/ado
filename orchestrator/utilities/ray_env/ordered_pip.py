# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import contextlib
import logging
import os
import typing

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


async def create_or_get_virtualenv(path: str, cwd: str, logger: logging.Logger):
    virtualenv_path = os.path.join(path, "virtualenv")
    if not os.path.exists(virtualenv_path):
        await original_create_or_get_virtualenv(path=path, cwd=cwd, logger=logger)


@contextlib.contextmanager
def patch_create_or_get_virtualenv(phase_index: int):
    if phase_index > 0:
        setattr(virtualenv_utils, "create_or_get_virtualenv", create_or_get_virtualenv)
    try:
        yield
    finally:
        setattr(
            virtualenv_utils,
            "create_or_get_virtualenv",
            original_create_or_get_virtualenv,
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
    ClassPath = "orchestrator.utilities.ray_env.ordered_pip.OrderedPipPlugin"

    def __init__(self, resources_dir: str | None = None):
        if resources_dir is None:
            import ray._private.ray_constants as ray_constants

            resources_dir = os.environ.get(
                ray_constants.RAY_RUNTIME_ENV_CREATE_WORKING_DIR_ENV_VAR
            )

        if not resources_dir:
            import tempfile

            resources_dir = tempfile.mkdtemp(prefix="ordered_pip_", dir="/tmp/ray")

        self._pip_resources_dir = resources_dir

        from ray._common.utils import try_to_create_directory

        try_to_create_directory(self._pip_resources_dir)
        self._pip_plugin = PipPlugin(self._pip_resources_dir)

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
            raise ValueError("missing the 'ordered_pip' key", runtime_env_dict)

        if not isinstance(runtime_env_dict["ordered_pip"], dict):
            raise ValueError("runtime_env['ordered_pip'] must be a dictionary")

        if not isinstance(runtime_env_dict["ordered_pip"]["phases"], list):
            raise ValueError(
                "runtime_env['ordered_pip']['phases'] must be a dictionary consistent with pip"
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
                )

        result = RuntimeEnv(ordered_pip={"phases": phases})
        logging.debug(
            f"Rewrote runtime_env `ordered_pip` field from {runtime_env_dict} to {result}."
        )

        return result

    def get_uris(self, runtime_env: "RuntimeEnv") -> list[str]:
        # VV: We want the hash to be invariant to the order of package names within a phase,
        # and we also want the order of phases to be reflected in the hash.
        aggregate_packages = [
            # VV: Ensure that each entry is expanded to the full pip spec
            sorted(p.get("packages", []))
            for p in self.validate(runtime_env)["ordered_pip"]["phases"]
        ]

        import hashlib

        return [
            "pip://" + hashlib.sha1(str(aggregate_packages).encode("utf-8")).hexdigest()
        ]

    async def create(
        self,
        uri: str,
        runtime_env: "RuntimeEnv",  # noqa: F821
        context: "RuntimeEnvContext",  # noqa: F821
        logger: logging.Logger | None = default_logger,
    ) -> int:
        uri = self.get_uris(runtime_env)[0]
        total_bytes = 0

        for idx, pip in enumerate(self.validate(runtime_env)["ordered_pip"]["phases"]):
            with patch_create_or_get_virtualenv(idx):
                total_bytes += await self._pip_plugin.create(
                    uri=uri,
                    runtime_env=RuntimeEnv(pip=pip),
                    context=context,
                    logger=logger,
                )

        return total_bytes

    def delete_uri(
        self, uri: str, logger: logging.Logger | None = default_logger
    ) -> int:
        return self._pip_plugin.delete_uri(uri=uri, logger=logger)

    def modify_context(
        self,
        uris: list[str],
        runtime_env: "RuntimeEnv",  # noqa: F821
        context: "RuntimeEnvContext",  # noqa: F821
        logger: logging.Logger = default_logger,
    ):
        phases = self.validate(runtime_env)["ordered_pip"]["phases"]

        if not len(phases):
            return

        self._pip_plugin.modify_context(
            uris=uris,
            runtime_env=RuntimeEnv(pip=phases[0]),
            context=context,
            logger=logger,
        )
