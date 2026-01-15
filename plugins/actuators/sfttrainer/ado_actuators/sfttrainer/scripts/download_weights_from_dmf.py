# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import lakehouse_export.export_utils as export_utils
import ray


class Datalake(export_utils.Datalake):
    def model_download(
        self,
        model_name: str,
        revision: str,
        output: str,
        force_download: bool = False,
        table: str = "model_shared",
    ) -> None:

        from lakehouse.assets import Model

        self._init_lakehouse()

        model = Model(lh=self._lakehouse)

        # VV: DMF places the weights under ${output}/${model_name}/${revision}
        model.pull(
            model=model_name,
            table=table,
            namespace=self._namespace,
            revision=revision,
            model_dir=output,
            force_download=force_download,
        )


@ray.remote(
    runtime_env={
        "pip": [
            "${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/dmf-lib-1.9.2.zip",
        ],
    },
)
def download_from_dmf(
    token: str,
    model_name: str,
    revision: str,
    namespace: str,
    output: str,
) -> None:
    dl = Datalake(token=token, namespace=namespace)
    dl.model_download(model_name=model_name, revision=revision, output=output)


# VV: Your lakehouse token in here
with open("token.txt") as f:
    token = f.read().rstrip()

ray.get(
    download_from_dmf.remote(
        token=token,
        model_name="mistral-large",
        revision="fp16_240620",
        namespace="mistral",
        output="/hf-models-pvc/models/external/mistral/",
    )
)
