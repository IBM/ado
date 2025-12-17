# TRIM EXAMPLE — Quickstart

## Install (from repo root)

```bash
uv pip install plugins/operators/trim
uv pip install -e examples/trim_custom_experiments
```

## Create the space

```bash
cd examples/trim_custom_experiments/trim_custom_experiments/configs
ado create space -f space_pressure.yaml --new-sample-store
```

> **Note:** The custom experiment `calculate_pressure_ideal_gas`
> is defined following  
> <https://ibm.github.io/ado/actuators/creating-custom-experiments/#decorating-your-custom-experiment-function>

---

## Troubleshooting

- **Check branch/commit**

  ```bash
  git checkout dl_operator_trim
  git log -1
  # expect: a83053c6eebf9f4ae0aaebcf37d1afeb9a023770
  ```
