---
status: published #Status can be draft, reviewed or published. 
---

!!! info  end

    If you are not familiar with the concept of a Discovery Space check [here](../core-concepts/discovery-spaces.md)

## Creating a `discoveryspace` resource

### Pre-requisites

In order to create a `discoveryspace` you must provide an existing `samplestore` that the `discoveryspace` will use for storage.
See the [samplestores](sample-stores.md) documentation for more details.

### Discovery Space configuration YAML

!!! info  end
    You can execute
    ```commandline 
        ado template space --include-schema`
    ```
    to output example YAML and full schema information for `discoveryspace` 

An example `discoveryspace` is given below. Note, the values in this YAML are for illustrative purposes and 
need to be changed to define a valid space. 

```yaml
sampleStoreIdentifier: source_abc123 # The id of the sample store to use
entitySpace: #A list of constitutive properties
  - identifier: my_property1 # The id of the first dimension/constitutive property of the space
  - identifier: my_property2 
experiments: # A list of experiments. The measurementspace of this discovery space
  - acuatatorIdentifier: someactuator # The id of the actuator that contains the experiment
    experimentIdentifier: experiment_one # The id of the experiment to execute
metadata:
  description: "This is an example discovery space"
  name: exampleSpace
```

The [describing constitutive properties](#defining-the-domains-of-constitutive-properties-in-the-entityspace) has more information on the 
options available for defining constitutive properties. 

Once you have your `discoveryspace` YAML in a file called `FILE.yaml` create it with

```commandline
ado create space -f FILE.yaml
```

If there are errors or inconsistencies in the space definition the create command will output an error. 
A common reason for inconsistency is that the properties defined in the entity-space do not match the properties
required for the experiments in the measurement space. 
The next section show a way around this issue. 

### Generating an initial YAML from an Experiment list

Given a list of experiment ids, that you want to use for the `measurementspace`, you can create an initial compatible `discoveryspace`
which you can then edit. See [constitutive properties and domains](#defining-the-domains-of-constitutive-properties-in-the-entityspace) for more.

Assuming you are interested in the `finetune-gptq-lora-dp-r-4-a-16-tm-default-v1.1.0` experiment,
you can create your space template using:

```commandline
ado template space --from-experiment finetune-gptq-lora-dp-r-4-a-16-tm-default-v1.1.0
```

More in-depth documentation about this feature can be found in the section about [`ado template`](../getting-started/ado.md#ado-template)

### Differences between input configuration YAML and stored configuration YAML

After creating a `discoveryspace` if you `ado get` its YAML you will notice there is far more information output then you input.
This is because currently the list of experiment references set in the config is expanded into the full experiment definitions and stored
with the `discoveryspace`. 

!!! warning end

    This is done to avoid having a dependency on actuator availability to read the `discoveryspace` internally in `ado`. 
    It is likely this will change in future versions. 

## Discovery Spaces and shared Sample Stores

Multiple `discoveryspace` resources can use the same `samplestore` resource - giving them a *common context*.
In this case you can think of the `discoveryspace` as a "View" on the `samplestore` contents, filtering just the
`entities` that match its description. 

To be more rigorous, give a `discoveryspace` you can apply this filter in two ways:

1. Filter `entities` that were placed in the `samplestore` via an operation the `discoveryspace`
2. Filter `entities` in the `samplestore` that match the `discoveryspace` 

To understand the difference in these two methods imagine two overlapping `discoveryspaces`, A and B, that use the same `samplestore`.
If someone uses method one on `discoveryspace` A, they will only see the `entities` placed there by operations on `discoveryspace` A.
However if someone uses method two on `discoveryspace` A, they will see `entities` placed there via operations on both `discoveryspace` A and space B.

## Accessing Entities

A common task is to see a table of measured entities associated with a `discoveryspace`

### `ado` cli

The `show` command is used to show things related to a resource.
In this case we want to show the entities related to a `discoveryspace` so we use:

```commandline
ado show entities space
```
By default, this will output the entities as a table. 
There are various option flags that control this behaviour e.g. output to a CSV file. 

Following the [above section](#discovery-spaces-and-shared-sample-stores) there are two lists of entities this could show.
The command above will use filter (1) - `entities` that were placed in the `samplestore` via an operation the `discoveryspace`.

If you want to use filter (2) - `entities` in the `samplestore` that match the `discoveryspace` - use. 

```commandline
ado show entities space --include matching
```

!!! info  end

    Note: in both cases measurements on the entity will be filtered to be only those defined by the `measurementspace` of the `discoveryspace`

Two other options are 

* `--include unsampled` which lists `entities` defined by the `discoveryspace` but not yet sampled
by any operation (as long as the space is finite).
* `--include missing` which  lists `entities` defined by the `discoveryspace` but not in the `samplestore`

### Programmatically

Assuming you have your [context](metastore.md#contexts-and-projects) in a file "my_context.yaml"

```python
import yaml
from orchestrator.metastore.project import ProjectContext
from orchestrator.core.discoveryspace.space import DiscoverySpace

with open("my_context.yaml") as f:
    c = ProjectContext.model_validate(yaml.safe_load(f))

space = DiscoverySpace.from_stored_configuration(project_context=c, space_identifier='space_abc123')
# Get the sampled and measured entities. Returns a pandas DataFrame
table = space.measuredEntitiesTable()
# Get the matching. Returns a pandas DataFrame
table = space.matchingEntitiesTable()
```

## Target v observed property formats 

There are two formats the entities can be output controlled by the `--property-format` option to `show entities`

The observed format outputs one row per entity. 
The columns are constitutive property names and the observed property names i.e. they include both the experiment id and target property id.
This ensures that with one row per entity there are no clashing column names.

The target format outputs one row per entity+experiment combination: so if there are two experiments in the Measurement Space
then there will be two rows per entity. 
In this format the columns are constitutive property names and target property names. 

!!! info  end
    
    With `property-format=target` if there multiple experiments measure *different* target properties this will result in many empty fields
    in the table. This is because the column for a given target of one experiment will not have values in the rows corresponding to other experiments.

## Defining the domains of constitutive properties in the entityspace

The YAML for the constitutive properties in the `entityspace` has the following structure

```yaml
identifier: model_name # The name of the property
propertyDomain: # The domain describes the values the property can take
  variableType: # The type of the variable: CATEGORICAL_VARIABLE_TYPE, DISCRETE_VARIABLE_TYPE, CONTINUOUS_VARIABLE_TYPE or UNKNOWN_VARIABLE_TYPE
                # The type defines what values the next fields can take
  values:   # If the variable is CATEGORICAL_VARIABLE_TYPE this is a list of the categories
    -       # If the variable is DISCRETE_VARIABLE_TYPE this can be a list of discrete float or integer values it can take
  domainRange: # If the variables is DISCRETE_VARIABLE_TYPE or CONTINUOUS_VARIABLE_TYPE this is the min inclusive, max exclusive range it can take
               # If the variable is DISCRETE_VARIABLE_TYPE and values are given this must be compatible with the values
  interval: # If the variable is DISCRETE_VARIABLE_TYPE this is the interval between the values. 
            # If given domainRange is required and values cannot be given

```

As long as all constitutive properties are not "UNKNOWN_VARAIBLE_TYPE" there is sufficient information to sample new entities from the `entityspace` description. 

### Ensuring the `entityspace` and `measurementspace` are compatible

This section elaborates on [Generating an initial YAML from an Experiment list](#generating-an-initial-yaml-from-an-experiment-list).

Experiments take entities as inputs and those entities must have values for various properties in order for the experiments to be able to process them.
This means the domains of the properties in the `entityspace` must be compatible with the experiments - if not entities could be sampled that
experiments in the `measurementspace` cannot measure. 

Assuming that:
- The experiment you are interested in is `finetune-full-fsdp-v1.6.0`
- It's part of the actuator `SFTTrainer`

To see the input requirements of an experiment you can run

```shell
ado describe experiment finetune-full-fsdp-v1.6.0 --actuator-id SFTTrainer
```


you will get output like 
```commandline
Identifier: SFTTrainer.finetune-full-fsdp-v1.6.0
Measures the performance of full-fine tuning a model with FSDP+flash-attention for a given (GPU model, number GPUS, batch_size, model_max_length) combination.
Inputs:
  Constitutive Properties:
      dataset_id
      Domain:
        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['news-chars-1024-entries-1024', 'news-chars-1024-entries-256', 'news-chars-1024-entries-4096', 'news-chars-2048-entries-1024', 'news-chars-2048-entries-256', 'news-chars-2048-entries-4096', 'news-chars-512-entries-1024', 'news-chars-512-entries-256', 'news-chars-512-entries-4096', 'news-tokens-128kplus-entries-320', 'news-tokens-128kplus-entries-4096', 'news-tokens-16384plus-entries-4096']
        
      
      model_name
      Domain:
        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['granite-13b-v2', 'granite-20b-v2', 'granite-34b-code-base', 'granite-3b-1.5', 'granite-3b-code-base-128k', 'granite-7b-base', 'granite-8b-code-base', 'granite-8b-code-base-128k', 'granite-8b-japanese', 'hf-tiny-model-private/tiny-random-BloomForCausalLM', 'llama-13b', 'llama-7b', 'llama2-70b', 'llama3-70b', 'llama3-8b', 'llama3.1-405b', 'llama3.1-70b', 'llama3.1-8b', 'mistral-7b-v0.1', 'mixtral-8x7b-instruct-v0.1']
        
      
      model_max_length
      Domain:
        Type: DISCRETE_VARIABLE_TYPE Interval: 1.0 Range: [1, 131073] 
      
      torch_dtype
      Domain:
        Type: CATEGORICAL_VARIABLE_TYPE Values: ['bfloat16'] 
      
      number_gpus
      Domain:
        Type: DISCRETE_VARIABLE_TYPE Interval: 1.0 Range: [2, 9] 
      
      gpu_model
      Domain:
        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['NVIDIA-A100-SXM4-80GB', 'NVIDIA-A100-80GB-PCIe', 'Tesla-T4', 'L40S', 'Tesla-V100-PCIE-16GB']
        
      
      batch_size
      Domain:
        Type: DISCRETE_VARIABLE_TYPE Interval: 1.0 Range: [1, 129] 
      
      
Outputs:
  finetune-full-fsdp-v1.6.0-gpu_compute_utilization_min
  finetune-full-fsdp-v1.6.0-gpu_compute_utilization_avg
  finetune-full-fsdp-v1.6.0-gpu_compute_utilization_max
  finetune-full-fsdp-v1.6.0-gpu_memory_utilization_min
  finetune-full-fsdp-v1.6.0-gpu_memory_utilization_avg
  finetune-full-fsdp-v1.6.0-gpu_memory_utilization_max
  finetune-full-fsdp-v1.6.0-gpu_memory_utilization_peak
  finetune-full-fsdp-v1.6.0-gpu_power_watts_min
  finetune-full-fsdp-v1.6.0-gpu_power_watts_avg
  finetune-full-fsdp-v1.6.0-gpu_power_watts_max
  finetune-full-fsdp-v1.6.0-gpu_power_percent_min
  finetune-full-fsdp-v1.6.0-gpu_power_percent_avg
  finetune-full-fsdp-v1.6.0-gpu_power_percent_max
  finetune-full-fsdp-v1.6.0-cpu_compute_utilization
  finetune-full-fsdp-v1.6.0-cpu_memory_utilization
  finetune-full-fsdp-v1.6.0-train_runtime
  finetune-full-fsdp-v1.6.0-train_samples_per_second
  finetune-full-fsdp-v1.6.0-train_steps_per_second
  finetune-full-fsdp-v1.6.0-train_tokens_per_second
  finetune-full-fsdp-v1.6.0-train_tokens_per_gpu_per_second
  finetune-full-fsdp-v1.6.0-model_load_time
  finetune-full-fsdp-v1.6.0-dataset_tokens_per_second
  finetune-full-fsdp-v1.6.0-dataset_tokens_per_second_per_gpu
  finetune-full-fsdp-v1.6.0-is_valid
```

Note the experiment gives the full domains it supports for each constitutive property. 
However, when constructing the `entityspace` you usually only want to use a sub-domain.


## Parameterizing Experiments

If an experiment has [optional properties](../core-concepts/actuators.md#optional-properties) you can define equivalent
properties in the entity space. If you don't the default value for the property will be used.

In addition, you can define your own custom parameterization of the experiment. For example, take the following experiment:

```
Identifier: robotic_lab.peptide_mineralization

Measures adsorption of peptide lanthanide combinations

Required Inputs:
  Constitutive Properties:
      peptide_identifier
      Domain:
        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['test_peptide', 'test_peptide_new']
        
      
      peptide_concentration
      Domain:
        Type: DISCRETE_VARIABLE_TYPE
        Values: [0.1, 0.4, 0.6, 0.8]
        Range: [0.1, 1.8]
        
      
      lanthanide_concentration
      Domain:
        Type: DISCRETE_VARIABLE_TYPE
        Values: [0.1, 0.4, 0.6, 0.8]
        Range: [0.1, 1.8]
        
      
      
Optional Inputs and Default Values:
  temperature
  Domain:
    Type: CONTINUOUS_VARIABLE_TYPE Range: [0, 100] 
  
  Default value: 23.0
  
  replicas
  Domain:
    Type: DISCRETE_VARIABLE_TYPE Interval: 1.0 Range: [1, 4] 
  
  Default value: 1.0
  
  robot_identifier
  Domain:
    Type: CATEGORICAL_VARIABLE_TYPE Values: ['harry', 'hermione'] 
  
  Default value: hermione
  
  
Outputs:
  peptide_mineralization-adsorption_timeseries
  peptide_mineralization-adsorption_plateau_value
```

It has three optional properties: `temperature`, `robot_identifier` and `replicas`. 

### Example: Customizing an experiment

The default temperature is `23` degrees C, however imagine you want to run this experiment at `30` degrees C.
You can define a `discoveryspace` like:

```yaml
sampleStoreIdentifier: c04713 
entitySpace:
- identifier:  peptide_identifier
  propertyDomain:
    values: ['test_peptide']
- identifier: peptide_concentration
  propertyDomain:
    values: [0.1,0.4,0.6,0.8]
- identifier: lanthanide_concentration
  propertyDomain:
    values: [ 0.1,0.4,0.6,0.8 ]
experiments:
- actuatorIdentifier: robotic_lab
  experimentIdentifier: peptide_mineralization
  parameterization:
    - value: 30
      property:
        identifier: 'temperature'
metadata:
  description: Space for exploring the absorption properties of test_peptide
```

### Example: Multiple customizations of the same experiment


You can add the multiple custom parameterizations of the same experiment e.g. one experiment that runs at 30 degrees C and 
another at 25 degrees. 

```yaml
sampleStoreIdentifier: c04713 # PUT REAL ID HERE
entitySpace:
- identifier:  peptide_identifier
  propertyDomain:
    values: ['test_peptide']
- identifier: peptide_concentration
  propertyDomain:
    values: [0.1,0.4,0.6,0.8]
- identifier: lanthanide_concentration
  propertyDomain:
    values: [ 0.1,0.4,0.6,0.8 ]
experiments:
- actuatorIdentifier: robotic_lab
  experimentIdentifier: peptide_mineralization
  parameterization:
    - value: 30
      property:
        identifier: 'temperature'
- actuatorIdentifier: robotic_lab
  experimentIdentifier: peptide_mineralization
  parameterization:
    - value: 25
      property:
        identifier: 'temperature'
metadata:
  description: Space for exploring the absorption properties of test_peptide
```

### Example: Using an optional property in the `entityspace`


Finally, if you want to scan a range of temperatures in your discovery space, the best would be to move this parameter into
the `entityspace`:

```yaml
sampleStoreIdentifier: c04713 # PUT REAL ID HERE
entitySpace:
- identifier:  peptide_identifier
  propertyDomain:
    values: ['test_peptide']
- identifier: peptide_concentration
  propertyDomain:
    values: [0.1,0.4,0.6,0.8]
- identifier: lanthanide_concentration
  propertyDomain:
    values: [ 0.1,0.4,0.6,0.8 ]
- identifier: temperature
  propertyDomain:
    domainRange: [20,30]
    interval: 1
experiments:
- actuatorIdentifier: robotic_lab
  experimentIdentifier: peptide_mineralization
metadata:
  description: Space for exploring the absorption properties of test_peptide
```

Here entities will be generated with a temperatures property that ranges from 20 to 30 degrees. 
When the experiment is run on the entity it will retrieve the value of the temperature from it rather than the Experiment. 

Our toy [example actuator](https://github.ibm.com/Discovery-Orchestrator/example_actuator) contains the above examples.
You can use it to experiment and explore custom parameterization. 