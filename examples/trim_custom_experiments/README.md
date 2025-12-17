# TRIM EXAMPLE

Be sure you are in the branch `dl_operator_trim`, my `git log -1` is
commit:
a83053c6eebf9f4ae0aaebcf37d1afeb9a023770

Install Trim, from root
`uv pip install plugins/operators/trim`

Install the custom experiments, from root
`uv pip install -e examples/trim_custom_experiments`

Create the resources as follows.

cd in the config folder, i.e. from root
`cd examples/trim_custom_experiments/trim_custom_experiments/configs`

run `ado create space -f space_pressure.yaml --new-sample-store`

and finally run `ado create space -f space_pressure.yaml --new-sample-store`

Note:

The custom experiment `calculate_pressure_ideal_gas` is
defined following the guide at <https://ibm.github.io/ado/actuators/creating-custom-experiments/#decorating-your-custom-experiment-function>
