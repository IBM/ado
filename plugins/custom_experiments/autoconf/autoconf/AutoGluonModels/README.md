# autoconf Model Information

This repository contains the models developed for `autoconf`

The current model is trained on ~15500 rows of data stored in
<https://lake-llm.cash.sl.cloud9.ibm.com/superset/dashboard/136/> (accessed on
30th October), achieving approximately 90% accuracy. You can find the models in
[the directory with this README](./).

## How we trained the models

To build the models on an environment which is as clean as possible we run
the following commands from the root of this project.

```terminal
uv sync --no-default-groups --python 3.12
uv activate .venv     
uv pip install -e plugins/custom_experiments/autoconf/
```

Then, we run the script from the virtual environment, we
also inserted the path to the data. The dataset we used will be
available soon.

```terminal
cd plugins/custom_experiments/autoconf/autoconf/utils/autoconf_build
python ml_classifier.py --data-path [modify_here]
```

## Versioning Information

All model subfolders in
[`autoconf/AutoGluonModels/`](autoconf/AutoGluonModels/) must follow the naming
convention: `vX-Y-Z`, where `X`, `Y` and `Z` are integers.

- **Major version (X):** Incremented when changes are made to:
  - **Columns**: Feature set or target variables (e.g., adding/removing
    features, switching to multi-label prediction).
  - **Rows**: Dataset used for building and testing the model.

- **Minor version (Y):** Incremented when changes are made to:
  - **Algorithms**: Modifications that may affect dependencies.
  - **Inference speed**: Performance optimizations.
  - **Model performance**: Improvements in accuracy or other metrics.
  - **Model size**: Significant changes (≥10× difference).

> **Note:** Minor version (`Y`) is not incremented if the major version (`X`) is
> updated. Furthermore, if models differs by the training splits used they will
> different in the minor version.

- **Patch version (Z):** Incremented when changes are made to:
  - Parameters for model invocation (see below) are updated independent of model

## Available Versions

The current available model versions are:

<!-- markdownlint-disable line-length -->

| Version | Folder                    | Comments                                                         | Author  | Status |
| ------- | ------------------------- | ---------------------------------------------------------------- | ------- | ------ |
| v3.1.0  | AutoGluonModels/v3-1-0... | Trained on the full Lakehouse data (refit-medium quality preset) | Daniele | Active |
| v3.0.0  | AutoGluonModels/v3-0-0... | Trained on the full Lakehouse data in (medium quality preset)    | Daniele | Active |

<!-- markdownlint-enable line-length -->

For more details, read the [changelog](changelog.md)

## Deprecated Versions

The deprecated model versions have been trained with autogluon v1.4 and are:

<!-- markdownlint-disable line-length -->

| Version | Folder                                                    | Comments                                                               | Author  | Status     |
| ------- | --------------------------------------------------------- | ---------------------------------------------------------------------- | ------- | ---------- |
| v2.0.0  | AutoGluonModels/v2-0-0_ag-20251113_154241-refit-clone-opt | Trained on Lakehouse data in Nov 2025 (Including `granite-4.0` models) | Daniele | Deprecated |
| v1.1.0  | AutoGluonModels/v1-1-0_ag-20251112_155927-refit-clone-opt | Removes LightGBM, fixes issues with `libomp` on macOS                  | Daniele | Deprecated |
| v1.0.0  | AutoGluonModels/v1-0-0_ag-20251024_100825-refit-clone-opt | Trained on Lakehouse data in Oct 2025                                  | Daniele | Deprecated |
| v0.0.0  | AutoGluonModels/v0-0-0_20251024_100825-refit-clone-opt    | Trained on Lakehouse data in Sept 2024                                 | Daniele | Deprecated |

<!-- markdownlint-enable line-length -->
