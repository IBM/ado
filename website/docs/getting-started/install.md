<!-- markdownlint-disable-next-line first-line-h1 -->
# Installing `ado`

**ado** can be installed in one of three ways:

1. From **PyPi**
2. From **GitHub**
3. By **cloning the GitHub repository** locally

???+ warning

    Before proceeding ensure you are using a supported Python version: run
    `python --version` in your terminal and check that you are on either **Python**
    **3.10**, **3.11**, **3.12**, or **3.13**.

    It is also highly recommended to create a **virtual environment** for ado, to
    avoid dependency conflicts with other packages. You can do so with:

    ```shell
    python -m venv ado-venv
    ```

    And activate it with

    ```shell
    source ado-venv/bin/activate
    ```

=== "From PyPi"

    This method installs the `ado-core` package from PyPi

    ```shell
    pip install ado-core
    ```

=== "From GitHub"

    ```shell
    pip install git+https://github.com/IBM/ado.git
    ```

=== "Cloning the repo locally"

    ```shell
    git clone https://github.com/IBM/ado.git
    cd ado
    pip install .
    ```

## Installing plugins

ado uses a plugin system to provide **additional actuators** and **operators**.
We maintain a set of actuators and operators
[in the ado main repo](https://github.com/ibm/ado/tree/main/plugins/).
Some plugins may also be available on PyPi.
You can install these actuators as follows:

!!! info

    Some plugins may have dependencies that may require credentials to access.
    Check the plugin's documentation if you encounter issues installing 
    a specific actuator.

=== "From PyPi"

    The following plugin packages are available:
    `ado-sfttrainer`, `ado-vllm-performance`, and `ado-ray-tune`

    ```shell
    pip install $PLUGIN_NAME
    ```

=== "From GitHub"

    For actuators

    ```shell
    pip install "git+https://github.com/IBM/ado.git#subdirectory=plugins/actuators/$ACTUATOR_NAME"
    ```

    For operators

    ```shell
    pip install "git+https://github.com/IBM/ado.git#subdirectory=plugins/operators/$OPERATOR_NAME"
    ```

=== "Cloning the repo"

    If you've cloned the ado repository locally in the previous step,
    you can run **from the top-level of the cloned repository**

    ```shell
    pip install plugins/actuators/$ACTUATOR_NAME
    ```

    or

    ```shell
    pip install plugins/operators/$OPERATOR_NAME
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