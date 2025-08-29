---
status: published #Status can be draft, reviewed or published. 
---

## Installing `ado`

**ado** can be installed in one of three ways:

1. By **cloning the GitHub repository** locally 
2. Via **GitHub** .


???+ warning

    Before proceeding ensure you are using a supported Python version: run
    `python --version` in your terminal and check that you are on either **Python**
    **3.10**, **3.11**, or **3.12**.

    It is also highly recommended to create a **virtual environment** for ado, to
    avoid dependency conflicts with other packages. You can do so with:

    ```shell
    python -m venv ado-venv
    ```

    And activate it with

    ```shell
    source ado-venv/bin/activate
    ```

=== "Cloning the repo"

    This, method **requires having set up an SSH key** for the repository to be cloned. In case
    you haven't already, you can do so [here](https://github.ibm.com/settings/keys)
    by following GitHub's
    [documentation](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

    ```shell
    git clone https://github.com/IBM/ado.git
    cd ado
    pip install .
    ```

=== "Via GitHub"

    This, method **requires having set up an SSH key** for the repository to be cloned. In case
    you haven't already, you can do so [here](https://github.ibm.com/settings/keys)
    by following GitHub's
    [documentation](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

    ```shell
    pip install git+https://github.com/IBM/ado"
    ```

## Installing plugins

ado uses a plugin system to provide **additional actuators** and **operators**. 
We maintain a set of actuators and operators in the ado main repo which you can see [here](https://github.com/ibm/ado/tree/main/plugins/).
You can install these actuators as follows:

!!! info 

    Some plugins may have dependencies that may require credentials to access. 
    Check the plugins's docs if you encounter issues installing a specific actuator. 

=== "Cloning the repo"


    If you've cloned the ado repository locally in the previous step, you can run **from the top-level of the cloned repository**

    ```shell
    pip install plugins/actuators/$ACTUATOR_NAME
    ```

    or 
    
    ```shell
    pip install plugins/operators/$OPERATOR_NAME
    ```


=== "Via GitHub"

    We assume you have an SSH key configured on your GitHub account.
    If you don't, look at the previous section for additional pointers.

    ```shell
    pip install "git+https://github.com/IBM/ado.git#subdirectory=plugins/actuators/$ACTUATOR_NAME"
    ```


    

## What's next

<div class="grid cards" markdown>

-   :octicons-rocket-24:{ .lg .middle } __Let's get started!__

    ---

    Learn what you can do with `ado`

    [Follow the guide :octicons-arrow-right-24:](ado.md)

-   :octicons-database-24:{ .lg .middle } __Collaborate with others__

    ---

    Learn how to install the components that allow you to collaborate with others.

    [Installing the Backend Services :octicons-arrow-right-24:](installing-backend-services.md)

</div>
