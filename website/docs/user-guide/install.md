<!-- markdownlint-disable-next-line first-line-h1 -->
# Installing `ado`

**ado** can be installed in one of three ways:

1. From **PyPI**
2. From **GitHub**
3. By **cloning the GitHub repository** locally

???+ warning

    Before proceeding, ensure you are using a supported Python version. Run
    `python --version` in your terminal and check that you are on **Python**
    **3.10**, **3.11**, **3.12**, **3.13**, or **3.14**.

    It is also highly recommended to create a **virtual environment** for
    `ado`, to avoid dependency conflicts with other packages. You can do so
    with:

    ```shell
    python -m venv ado-venv
    ```

    And activate it with

    ```shell
    source ado-venv/bin/activate
    ```

=== "From PyPI"

    This method installs the [`ado-core`](https://pypi.org/project/ado-core/)
    package from PyPI.

    ```shell
    pip install ado-core
    ```

=== "From GitHub"

    This installs the current repository directly from GitHub.

    ```shell
    pip install git+https://github.com/IBM/ado.git
    ```

=== "Cloning the repo locally"

    Clone the repository locally, then install the top-level package:

    ```shell
    git clone https://github.com/IBM/ado.git
    cd ado
    pip install .
    ```

    If intend to develop ado, refer to our
    [development setup guidelines](./developing.md).

## Installing plugins

`ado` uses a plugin system to provide **additional actuators**,
**operators**, and **custom experiments**. We maintain a set of plugins
[in the ado main repo](https://github.com/IBM/ado/tree/main/plugins/).
Some plugins are also distributed separately on PyPI.
You can install these plugins as follows:

!!! info

    Some plugins may have dependencies that require credentials or additional
    system setup. Check the plugin's documentation if you encounter issues
    installing a specific plugin.

=== "From PyPI"

    The following plugin packages are available on PyPI:
    `ado-anomalous-series`, `ado-autoconf`, `ado-cplex-mip`,
    `ado-profile-space`, `ado-ray-tune`, `ado-sfttrainer`, `ado-trim`, and
    `ado-vllm-performance`

    ```shell
    pip install $PLUGIN_NAME
    ```

=== "From GitHub"

    For actuators:

    ```shell
    pip install "git+https://github.com/IBM/ado.git#subdirectory=plugins/actuators/$ACTUATOR_NAME"
    ```

    For operators:

    ```shell
    pip install "git+https://github.com/IBM/ado.git#subdirectory=plugins/operators/$OPERATOR_NAME"
    ```

    For custom experiments:

    ```shell
    pip install "git+https://github.com/IBM/ado.git#subdirectory=plugins/custom_experiments/$CUSTOM_EXPERIMENT_NAME"
    ```

=== "Cloning the repo"

    If you've cloned the `ado` repository locally in the previous step, run
    these commands from the top level of the cloned repository.

    For actuators:

    ```shell
    pip install plugins/actuators/$ACTUATOR_NAME
    ```

    For operators:

    ```shell
    pip install plugins/operators/$OPERATOR_NAME
    ```

    For custom experiments:

    ```shell
    pip install plugins/custom_experiments/$CUSTOM_EXPERIMENT_NAME
    ```

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Let's get started!**

    ---

    Learn what you can do with `ado`

    [Follow the guide :octicons-arrow-right-24:](ado.md)

- :octicons-database-24:{ .lg .middle } **Collaborate with others**

    ---

    Learn how to install the components that allow you to collaborate with others.

    [Installing the Backend Services :octicons-arrow-right-24:](installing-backend-services.md)

</div>
<!-- markdownlint-enable line-length -->
