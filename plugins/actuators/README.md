# Content

This directory contains a set of `ado` actuator plugins that provide
experiments.

Each actuator must be installed independently. However, all will be installed
into a namespace package called "ado_actuators".

This means to import a module "actuator.py" provided by an actuator "myactuator"
you would do

```python
import ado_actuators.myactuator.actuator
```

Namespace packages allow developers to independently create and distribute
Python modules that will be installed under a common package name.

## Installing Actuators

To install an actuator $NAME that is in this directory, first cd into this
directory, then:

```commandline
pip install ./$NAME
```

To uninstall

```commandline
pip uninstall $NAME
```

Note: Don't execute the uninstall command in the $NAME directory as `pip` will
see the local package not the installed package.

For example, for `sfttrainer`

```commandline
pip install ./sfttrainer
pip uninstall sfttrainer
```

## Installing an Actuator on a Remote Ray Cluster

The recommended way of installing an actuator in a remote Ray cluster is to
build a wheel. You would do this if you have local changes to one or more
plugins you want to use for a run OR the version of the orchestrator in the
remote ray image is behind the latest.

The simplest scenario is outlined below and it uses the actuator
`fms_hf_tuning` as a concrete example. It assumes _you are in the directory of
the actuator you want to install_ : in this case
`orchestrator_plugins/actuator/fms_hf_tuning`

1. `rm -rf dist/ build/ *.egg-info`

   This removes any previous build artifacts and wheels. This prevents issues
   with old files being included in the new wheel.

2. `python -m build -o dist`

   This creates a `dist/` directory with the wheel. It will have a name like
   `ado_sfttrainer-XXXX.whl`

3. `mv dist/*.whl` .

   Required if the plugins is in the ado repo as `dist/` is in `.gitignore` and
   will not be uploaded by `ray job submit`

4. Write a `ray_runtime_env.yaml` file that installs the wheel from your current
   dir using pip e.g.

   ```yaml
   pip:
     - ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/ado_sfttrainer-1.1.0.dev10+ge8ad7b8e.d20250212-py3-none-any.whl
   ```

5. Submit the ray job specifying the current directory as the working dir.

Here we just use the simple example of running `ado get actuators` to list the
installed plugins:

```commandline
ray job submit --working-dir $PWD \
    --runtime-env runtime_env.yaml \
    --address=http://localhost:8265 -- ado get actuators --details
```

### Things to Check

- Make sure plugin code changes are committed.
  - If they are not committed then the version of the built wheel will not
    change i.e. it will be same as for a wheel build before the changes
  - If a wheel with the version was already sent, ray will use the cached
    version, and not your new version
- Make sure new files you want to package with the wheel are committed
  - The setup.py for the plugins only adds committed non-python files

If you need to install multiple actuators you repeat steps 1-3 above for each
additional actuator, and then in step 4 add it to the list under the `pip` key.

### Details and Alternate Methods

**You can skip this section unless you want to know why the plugins are
installed as wheels**

Sending the namespace package directory and trying to install from it on the
remote cluster is problematic for the following reasons

1. You cannot use ray `py_modules` with the package directory as it doesn't
   actually install the package
   - It just adds the module dir to python path which does not work for
     namespace packages as they have no `__init__.py`.
2. For plugins stored in `ado` repo, you can't use ray `pip` field with the
   package directory as it requires sending `.git` dir of ado which exceeds ray
   transfer limit
   - This is because the namespace package `setup.py` uses `setuptools_scm` to
     dynamically compute version and identify non-python files.
   - NOTE: If you have a standalone plugin repository you may not have this
     issue.

If you want to use the `ado_actuators` directory and not a wheel you must

1. edit the `setup.py` to remove use of `setuptool_scm`
2. ensure the non python files required, like `actuator_definition.yaml`, are
   discovered by setup.py

NB: If you do step one, but not step two, you will break the installation as
required non-python files will not be in the wheel.
