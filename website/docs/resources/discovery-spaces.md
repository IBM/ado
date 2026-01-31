<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
!!! info end

    If you are not familiar with the concept of a Discovery Space check [here](../core-concepts/discovery-spaces.md)

## Creating a `discoveryspace` resource

### Pre-requisites

In order to create a `discoveryspace`, you must provide a `samplestore` that the
`discoveryspace` will use for storage. To see existing `samplestores` run:

```commandline
ado get samplestores
```

Alternatively, if there is no existing store, when creating the space you can
use the `--new-sample-store` flag. See the [samplestores](sample-stores.md)
documentation for more details.

### Discovery Space configuration YAML

!!! info end

    You can execute `ado template space --include-schema` to output
    example YAML and full schema information for `discoveryspace`

An example `discoveryspace` is given below. Note, the values in this YAML are
for illustrative purposes and need to be changed to define a valid space.

<!-- markdownlint-disable line-length -->

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

<!-- markdownlint-enable line-length -->

The
[describing constitutive properties](#defining-the-domains-of-constitutive-properties-in-the-entityspace)
page provides more information on the available options for defining
constitutive properties.

Once you have your `discoveryspace` YAML in a file called `FILE.yaml` create it
with

```commandline
ado create space -f FILE.yaml
```

If there are errors or inconsistencies in the space definition the create
command will output an error. A common reason for inconsistency is that the
properties defined in the entity space do not match the properties required for
the experiments in the measurement space. The next section shows a way to handle
this issue.

### Generating an initial YAML from an Experiment list

Given a list of experiment ids that you want to use for the `measurementspace`,
you can create an initial compatible `discoveryspace` which you can then edit.
See
[constitutive properties and domains](#defining-the-domains-of-constitutive-properties-in-the-entityspace)
for more.

Assuming you are interested in the
`finetune-gptq-lora-dp-r-4-a-16-tm-default-v1.1.0` experiment, you can create
your space template using:

```commandline
ado template space --from-experiment finetune-gptq-lora-dp-r-4-a-16-tm-default-v1.1.0
```

More in-depth documentation about this feature can be found in the section about
[`ado template`](../getting-started/ado.md#ado-template)

### Differences between input configuration YAML and stored configuration YAML

After creating a `discoveryspace`, if you `ado get` its YAML you will notice
that the information output is different from the one you provided in input.
This is because the list of experiment references set in the YAML is expanded
into the full experiment definitions and stored with the `discoveryspace`.

## `discoveryspaces` and shared `samplestores`

Multiple `discoveryspace` resources can use the same `samplestore` resource. In
this case you can think of the `discoveryspace` as a "view" on the `samplestore`
contents, filtering just the `entities` that match its description.

To be more rigorous, given a `discoveryspace` you can apply this filter in two
ways:

1. Filter `entities` that were placed in the `samplestore` via an operation on
   the `discoveryspace`
2. Filter `entities` in the `samplestore` that match the `discoveryspace`

To understand the difference in these two methods imagine two overlapping
`discoveryspaces`, A and B, that use the same `samplestore`. If someone uses
method one on `discoveryspace` A, they will only see the `entities` placed there
by operations on `discoveryspace` A. However, if someone uses method two on
`discoveryspace` A, they will see `entities` placed there via operations on both
`discoveryspace` A and space B.

Shared samples stores also allow data to be reused across `discoveryspaces`,
potentially accelerating exploration operations. See the
[shared sample store](../core-concepts/data-sharing.md) documentation for
further details.

## Accessing Entities

A common task is to see a table of measured entities associated with a
`discoveryspace`

### `ado` cli

The `show` command is used to show things related to a resource. In this case we
want to show the entities related to a `discoveryspace` so we use:

```commandline
ado show entities space
```

By default, this will output the entities as a table. There are various option
flags that control this behaviour e.g. output to a CSV file.

Following the [above section](#discoveryspaces-and-shared-samplestores) there
are two lists of entities this could show. The command above will use filter
(1) - `entities` that were placed in the `samplestore` via an operation the
`discoveryspace`.

If you want to use filter (2) - `entities` in the `samplestore` that match the
`discoveryspace` - use:

```commandline
ado show entities space --include matching
```

!!! info end

    Note: in both cases measurements on the entity will be filtered to be only those
    defined by the `measurementspace` of the `discoveryspace`

Two other options are

- `--include unsampled` which lists `entities` defined by the `discoveryspace`
  but not yet sampled by any operation (as long as the space is finite).
- `--include missing` which lists `entities` defined by the `discoveryspace` but
  not in the `samplestore`

### Programmatically

Assuming you have your [context](metastore.md#contexts-and-projects) in a file
"my_context.yaml"

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

## Target vs observed property formats

There are two formats the entities can be output controlled by the
`--property-format` option to `show entities`

The observed format outputs one row per entity. The columns are constitutive
property names and the observed property names i.e. they include both the
experiment id and target property id. This ensures that with one row per entity
there are no clashing column names.

The target format outputs one row per entity+experiment combination: so if there
are two experiments in the Measurement Space then there will be two rows per
entity. In this format the columns are constitutive property names and target
property names.

!!! info end

    With `property-format=target` if the measurement space contains multiple
    experiments measuring _different_ target properties, this will result in many
    empty fields in the table. This is because the column for a given target
    of one experiment will not have values in the rows corresponding
    to other experiments.

## Defining the domains of constitutive properties in the entityspace

The YAML for the constitutive properties in the `entityspace` has the following
structure

<!-- markdownlint-disable line-length -->

```yaml
identifier: model_name # The name of the property
propertyDomain: # The domain describes the values the property can take
  variableType:# The type of the variable: CATEGORICAL_VARIABLE_TYPE, DISCRETE_VARIABLE_TYPE, CONTINUOUS_VARIABLE_TYPE or UNKNOWN_VARIABLE_TYPE
    # The type defines what values the next fields can take
  values: # If the variable is CATEGORICAL_VARIABLE_TYPE this is a list of the categories
    -  # If the variable is DISCRETE_VARIABLE_TYPE this can be a list of discrete float or integer values it can take
  domainRange:# If the variables is DISCRETE_VARIABLE_TYPE or CONTINUOUS_VARIABLE_TYPE this is the min inclusive, max exclusive range it can take
    # If the variable is DISCRETE_VARIABLE_TYPE and values are given this must be compatible with the values
  interval:# If the variable is DISCRETE_VARIABLE_TYPE this is the interval between the values.
    # If given domainRange is required and values cannot be given
```

<!-- markdownlint-enable line-length -->

As long as all constitutive properties are not "UNKNOWN_VARIABLE_TYPE" there is
sufficient information to sample new entities from the `entityspace`
description.

### Ensuring the `entityspace` and `measurementspace` are compatible

This section elaborates on
[Generating an initial YAML from an Experiment list](#generating-an-initial-yaml-from-an-experiment-list).

Experiments take entities as inputs and those entities must have values for
various properties in order for the experiments to be able to process them. This
means the domains of the properties in the `entityspace` must be compatible with
the experiments - if not entities could be sampled that experiments in the
`measurementspace` cannot measure.

For example, to see the input requirements of the experiment
`finetune_full_benchmark-v1.0.0` you can run:

```shell
ado describe experiment finetune_full_benchmark-v1.0.0 --actuator-id SFTTrainer
```

you will get output like

<!-- markdownlint-disable line-length -->
```terminaloutput
Identifier: SFTTrainer.finetune_full_benchmark-v1.0.0
Description: Measures the performance of full-finetuning a model for a given (GPU model, number GPUS, batch_size, 
model_max_length, number nodes) combination.

Required Inputs:
                                                                                                           
   Constitutive Properties:                                                                                
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: model_name                                                                                
     Description: The huggingface name or path to the model                                                
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: [                                                                                          
            'allam-1-13b',                                                                                 
            'granite-13b-v2',                                                                              
            'granite-20b-v2',                                                                              
            'granite-3-8b',                                                                                
            'granite-3.0-1b-a400m-base',                                                                   
            'granite-3.1-2b',                                                                              
            'granite-3.1-3b-a800m-instruct',                                                               
            'granite-3.1-8b-instruct',                                                                     
            'granite-3.3-8b',                                                                              
            'granite-34b-code-base',                                                                       
            'granite-3b-1.5',                                                                              
            'granite-3b-code-base-128k',                                                                   
            'granite-4.0-1b',                                                                              
            'granite-4.0-350m',                                                                            
            'granite-4.0-h-1b',                                                                            
            'granite-4.0-h-micro',                                                                         
            'granite-4.0-h-small',                                                                         
            'granite-4.0-h-tiny',                                                                          
            'granite-4.0-micro',                                                                           
            'granite-7b-base',                                                                             
            'granite-8b-code-base',                                                                        
            'granite-8b-code-base-128k',                                                                   
            'granite-8b-code-instruct',                                                                    
            'granite-8b-japanese',                                                                         
            'granite-vision-3.2-2b',                                                                       
            'hf-tiny-model-private/tiny-random-BloomForCausalLM',                                          
            'llama-13b',                                                                                   
            'llama-7b',                                                                                    
            'llama2-70b',                                                                                  
            'llama3-70b',                                                                                  
            'llama3-8b',                                                                                   
            'llama3.1-405b',                                                                               
            'llama3.1-70b',                                                                                
            'llama3.1-8b',                                                                                 
            'llama3.2-1b',                                                                                 
            'llama3.2-3b',                                                                                 
            'llava-v1.6-mistral-7b',                                                                       
            'mistral-123b-v2',                                                                             
            'mistral-7b-v0.1',                                                                             
            'mixtral-8x7b-instruct-v0.1',                                                                  
            'smollm2-135m'                                                                                 
        ]                                                                                                  
                                                                                                           
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: model_max_length                                                                          
     Description: The maximum context size. Dataset entries with more tokens they are truncated. Entri     
     are padded                                                                                            
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Interval: 1                                                                                        
        Range: [1, 131073]                                                                                 
                                                                                                           
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: batch_size                                                                                
     Description: The total batch size to use                                                              
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Interval: 1                                                                                        
        Range: [1, 4097]                                                                                   
                                                                                                           
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: number_gpus                                                                               
     Description: The total number of GPUs to use                                                          
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Interval: 1                                                                                        
        Range: [0, 33]                                                                                     
                                                                                                           
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
                                                                                                           
Optional Inputs and Default Values:
                                                                                                           
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: max_steps                                                                                 
     Description: The number of optimization steps to perform. Set to -1 to respect num_train_epochs i     
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Interval: 1                                                                                        
        Range: [-1, 10001]                                                                                 
                                                                                                           
     Default value: -1                                                                                     
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: num_train_epochs                                                                          
     Description: How many epochs to run. Ignored if max_steps is greater than 0                           
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Interval: 1                                                                                        
        Range: [1.0, 10001.0]                                                                              
                                                                                                           
     Default value: 1.0                                                                                    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: stop_after_seconds                                                                        
     Description: If set, the optimizer will be asked to stop after the specified time elapses. The ch     
     performed after the end of each training step.                                                        
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Interval: 1                                                                                        
        Range: [-1.0, 1000001.0]                                                                           
                                                                                                           
     Default value: -1.0                                                                                   
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: dataset_id                                                                                
     Description: The identifier of the dataset to use for training                                        
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: [                                                                                          
            'news-chars-1024-entries-1024',                                                                
            'news-chars-1024-entries-256',                                                                 
            'news-chars-1024-entries-4096',                                                                
            'news-chars-2048-entries-1024',                                                                
            'news-chars-2048-entries-256',                                                                 
            'news-chars-2048-entries-4096',                                                                
            'news-chars-512-entries-1024',                                                                 
            'news-chars-512-entries-256',                                                                  
            'news-chars-512-entries-4096',                                                                 
            'news-tokens-128kplus-entries-320',                                                            
            'news-tokens-128kplus-entries-4096',                                                           
            'news-tokens-16384plus-entries-4096',                                                          
            'vision-384x384-16384plus-entries-4096',                                                       
            'vision-384x768-16384plus-entries-4096'                                                        
        ]                                                                                                  
                                                                                                           
     Default value: 'news-tokens-16384plus-entries-4096'                                                   
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: gradient_checkpointing                                                                    
     Description: If True, use gradient checkpointing to save memory (i.e. higher batchsizes) at the e     
     slower backward pass                                                                                  
     Domain:                                                                                               
                                                                                                           
        Type: BINARY_VARIABLE_TYPE                                                                         
        Values: [True, False]                                                                              
                                                                                                           
     Default value: 1                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: torch_dtype                                                                               
     Description: The torch datatype to use                                                                
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: ['bfloat16', 'float16', 'float32']                                                         
                                                                                                           
     Default value: 'bfloat16'                                                                             
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: gradient_accumulation_steps                                                               
     Description: Number of update steps to accumulate before performing a backward/update pass.           
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Interval: 1                                                                                        
        Range: [1, 33]                                                                                     
                                                                                                           
     Default value: 4                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: auto_stop_method                                                                          
     Description: The default value is `None`. This parameter defines the method used to automatically     
     fine-tuning job. Supported values are `WARMUP_60S_STABLE_120S_OR_10_STEPS` and `None`. If set to      
     `WARMUP_60S_STABLE_120S_OR_10_STEPS`, the job stops after spending at least 60 seconds in the         
     warmup phase plus the longer of 120 seconds or the duration of 10 optimization steps. This method     
     excludes the first 60 seconds of training when calculating throughput and system metrics.             
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: ['WARMUP_60S_STABLE_120S_OR_10_STEPS', None]                                               
                                                                                                           
     Default value: None                                                                                   
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: r                                                                                         
     Description: The LORA rank                                                                            
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Interval: 1                                                                                        
        Range: [1, 33]                                                                                     
                                                                                                           
     Default value: 4                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: lora_alpha                                                                                
     Description: LORA Alpha scales the learning weights                                                   
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Interval: 1                                                                                        
        Range: [1, 33]                                                                                     
                                                                                                           
     Default value: 16                                                                                     
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: fast_moe                                                                                  
     Description: Configures the amount of expert parallel sharding. number_gpus must be divisible by      
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Interval: 1                                                                                        
        Range: [0, 33]                                                                                     
                                                                                                           
     Default value: 0                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: fast_kernels                                                                              
     Description: Switches on fast kernels, the value is a list with strings of boolean values for [fa     
     fast_rms_layernorm, fast_rope_embeddings]                                                             
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: [None, ['True', 'True', 'True']]                                                           
                                                                                                           
     Default value: None                                                                                   
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: optim                                                                                     
     Description: The optimizer to use.                                                                    
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: [                                                                                          
            'adamw_hf',                                                                                    
            'adamw_torch',                                                                                 
            'adamw_torch_fused',                                                                           
            'adamw_torch_xla',                                                                             
            'adamw_torch_npu_fused',                                                                       
            'adamw_apex_fused',                                                                            
            'adafactor',                                                                                   
            'adamw_anyprecision',                                                                          
            'adamw_torch_4bit',                                                                            
            'ademamix',                                                                                    
            'sgd',                                                                                         
            'adagrad',                                                                                     
            'adamw_bnb_8bit',                                                                              
            'adamw_8bit',                                                                                  
            'ademamix_8bit',                                                                               
            'lion_8bit',                                                                                   
            'lion_32bit',                                                                                  
            'paged_adamw_32bit',                                                                           
            'paged_adamw_8bit',                                                                            
            'paged_ademamix_32bit',                                                                        
            'paged_ademamix_8bit',                                                                         
            'paged_lion_32bit',                                                                            
            'paged_lion_8bit',                                                                             
            'rmsprop',                                                                                     
            'rmsprop_bnb',                                                                                 
            'rmsprop_bnb_8bit',                                                                            
            'rmsprop_bnb_32bit',                                                                           
            'galore_adamw',                                                                                
            'galore_adamw_8bit',                                                                           
            'galore_adafactor',                                                                            
            'galore_adamw_layerwise',                                                                      
            'galore_adamw_8bit_layerwise',                                                                 
            'galore_adafactor_layerwise',                                                                  
            'lomo',                                                                                        
            'adalomo',                                                                                     
            'grokadamw',                                                                                   
            'schedule_free_adamw',                                                                         
            'schedule_free_sgd'                                                                            
        ]                                                                                                  
                                                                                                           
     Default value: 'adamw_torch'                                                                          
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: bf16                                                                                      
     Description: Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher N     
     bf16 mixed precision support for NPU architecture or using CPU (use_cpu) or Ascend NPU. This is       
     an experimental API and it may change.                                                                
     Domain:                                                                                               
                                                                                                           
        Type: BINARY_VARIABLE_TYPE                                                                         
        Values: [False, True]                                                                              
                                                                                                           
     Default value: 0                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: gradient_checkpointing_use_reentrant                                                      
     Description: Specify whether to use the activation checkpoint variant that requires reentrant aut     
     parameter should be passed explicitly. Torch version 2.5 will raise an exception if use_reentrant     
     is not passed. If use_reentrant=False, checkpoint will use an implementation that does not            
     require reentrant autograd. This allows checkpoint to support additional functionality, such as       
     working as expected with torch.autograd.grad and support for keyword arguments input into the         
     checkpointed function.                                                                                
     Domain:                                                                                               
                                                                                                           
        Type: BINARY_VARIABLE_TYPE                                                                         
        Values: [False, True]                                                                              
                                                                                                           
     Default value: 0                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: dataset_text_field                                                                        
     Description: Training dataset text field containing single sequence. Either the dataset_text_fiel     
     data_formatter_template need to be supplied. For running vision language model tuning pass the        
     column name for text data.                                                                            
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: ['output', 'messages']                                                                     
                                                                                                           
     Default value: 'output'                                                                               
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: dataset_image_field                                                                       
     Description: For running vision language model tuning pass the column name of the image data in t     
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: [None, 'images']                                                                           
                                                                                                           
     Default value: None                                                                                   
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: remove_unused_columns                                                                     
     Description: Remove columns not required by the model when using an nlp.Dataset.                      
     Domain:                                                                                               
                                                                                                           
        Type: BINARY_VARIABLE_TYPE                                                                         
        Values: [True, False]                                                                              
                                                                                                           
     Default value: 1                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: dataset_kwargs_skip_prepare_dataset                                                       
     Description: When True, configures trl to skip preparing the dataset.                                 
     Domain:                                                                                               
                                                                                                           
        Type: BINARY_VARIABLE_TYPE                                                                         
        Values: [True, False]                                                                              
                                                                                                           
     Default value: 0                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: flash_attn                                                                                
     Description: Use Flash attention v2 from transformers                                                 
     Domain:                                                                                               
                                                                                                           
        Type: BINARY_VARIABLE_TYPE                                                                         
        Values: [True, False]                                                                              
                                                                                                           
     Default value: 1                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: gpu_model                                                                                 
     Description: The GPU model to use                                                                     
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: [                                                                                          
            None,                                                                                          
            'NVIDIA-A100-SXM4-80GB',                                                                       
            'NVIDIA-A100-80GB-PCIe',                                                                       
            'Tesla-T4',                                                                                    
            'L40S',                                                                                        
            'Tesla-V100-PCIE-16GB',                                                                        
            'NVIDIA-H100-PCIe',                                                                            
            'NVIDIA-H100-80GB-HBM3'                                                                        
        ]                                                                                                  
                                                                                                           
     Default value: None                                                                                   
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: distributed_backend                                                                       
     Description: Which pytorch backend to use when training with multiple GPU devices                     
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: ['DDP', 'FSDP', 'None']                                                                    
                                                                                                           
     Default value: 'FSDP'                                                                                 
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: number_nodes                                                                              
     Description: If set, actuator distributes tasks on multiple nodes. Each Node will use number_gpus     
     GPUs. Each Node will use 1 process for each GPU it uses                                               
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Interval: 1                                                                                        
        Range: [1, 9]                                                                                      
                                                                                                           
     Default value: 1                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: fms_hf_tuning_version                                                                     
     Description: The version of fms-hf-tuning to use - controls which wrapper to use as well as python     
     dependencies                                                                                          
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: [                                                                                          
            '2.0.1',                                                                                       
            '2.1.0',                                                                                       
            '2.1.1',                                                                                       
            '2.1.2',                                                                                       
            '2.2.1',                                                                                       
            '2.3.1',                                                                                       
            '2.4.0',                                                                                       
            '2.5.0',                                                                                       
            '2.6.0',                                                                                       
            '2.7.1',                                                                                       
            '2.8.2',                                                                                       
            '3.0.0',                                                                                       
            '3.0.0.1',                                                                                     
            '3.1.0'                                                                                        
        ]                                                                                                  
                                                                                                           
     Default value: '2.1.2'                                                                                
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: enable_roce                                                                               
     Description: This setting is only in effect for multi-node runs. It controls whether RDMA over Co     
     Ethernet (RoCE) is switched on or not                                                                 
     Domain:                                                                                               
                                                                                                           
        Type: BINARY_VARIABLE_TYPE                                                                         
        Values: [False, True]                                                                              
                                                                                                           
     Default value: 0                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: fsdp_sharding_strategy                                                                    
     Description: [1] FULL_SHARD (shards optimizer states, gradients and parameters), [2] SHARD_GRAD_O     
     optimizer states and gradients), [3] NO_SHARD (DDP), [4] HYBRID_SHARD (shards optimizer states,       
     gradients and parameters within each node while each node has full copy - equivalent to               
     FULL_SHARD for single-node runs), [5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients       
     within each node while each node has full copy). For more information, please refer the official      
     PyTorch docs.                                                                                         
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: ['FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD', 'HYBRID_SHARD', 'HYBRID_SHARD_ZERO2']          
                                                                                                           
     Default value: 'FULL_SHARD'                                                                           
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: fsdp_state_dict_type                                                                      
     Description: [1] FULL_STATE_DICT, [2] LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT                        
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: ['FULL_STATE_DICT', 'LOCAL_STATE_DICT', 'SHARDED_STATE_DICT']                              
                                                                                                           
     Default value: 'FULL_STATE_DICT'                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: fsdp_use_orig_params                                                                      
     Description: If True, allows non-uniform `requires_grad` during init, which means support for int     
     frozen and trainable parameters. (useful only when `use_fsdp` flag is passed).                        
     Domain:                                                                                               
                                                                                                           
        Type: BINARY_VARIABLE_TYPE                                                                         
        Values: [False, True]                                                                              
                                                                                                           
     Default value: 1                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: accelerate_config_mixed_precision                                                         
     Description: Whether or not to use mixed precision training. Choose from 'no', 'fp16', 'bf16' or      
     requires the installation of transformers-engine.                                                     
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: ['no', 'fp16', 'bf16', 'fp8']                                                              
                                                                                                           
     Default value: 'no'                                                                                   
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: accelerate_config_fsdp_transformer_layer_cls_to_wrap                                      
     Description: List of transformer layer class names (case-sensitive) to wrap, e.g, BertLayer,          
     GraniteDecoderLayer, GPTJBlock, T5Block ... (useful only when using FSDP)                             
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: [                                                                                          
            None,                                                                                          
            'GraniteDecoderLayer',                                                                         
            'LlamaDecoderLayer',                                                                           
            'MistralDecoderLayer',                                                                         
            'GPTJBlock',                                                                                   
            'T5Block'                                                                                      
        ]                                                                                                  
                                                                                                           
     Default value: None                                                                                   
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
                                                                                                           
Outputs:
 ───────────────────────────────────────────────────────────────────────────────────────────────────────── 
   finetune_full_benchmark-v1.0.0-is_valid                                                                 
   finetune_full_benchmark-v1.0.0-dataset_tokens_per_second_per_gpu                                        
   finetune_full_benchmark-v1.0.0-train_runtime                                                            
   finetune_full_benchmark-v1.0.0-dataset_tokens_per_second                                                
   finetune_full_benchmark-v1.0.0-train_samples_per_second                                                 
   finetune_full_benchmark-v1.0.0-train_steps_per_second                                                   
   finetune_full_benchmark-v1.0.0-train_tokens_per_second                                                  
   finetune_full_benchmark-v1.0.0-train_tokens_per_gpu_per_second                                          
   finetune_full_benchmark-v1.0.0-cpu_compute_utilization                                                  
   finetune_full_benchmark-v1.0.0-cpu_memory_utilization                                                   
   finetune_full_benchmark-v1.0.0-gpu_compute_utilization_min                                              
   finetune_full_benchmark-v1.0.0-gpu_compute_utilization_avg                                              
   finetune_full_benchmark-v1.0.0-gpu_compute_utilization_max                                              
   finetune_full_benchmark-v1.0.0-gpu_memory_utilization_min                                               
   finetune_full_benchmark-v1.0.0-gpu_memory_utilization_avg                                               
   finetune_full_benchmark-v1.0.0-gpu_memory_utilization_max                                               
   finetune_full_benchmark-v1.0.0-gpu_memory_utilization_peak                                              
   finetune_full_benchmark-v1.0.0-gpu_power_watts_min                                                      
   finetune_full_benchmark-v1.0.0-gpu_power_watts_avg                                                      
   finetune_full_benchmark-v1.0.0-gpu_power_watts_max                                                      
   finetune_full_benchmark-v1.0.0-gpu_power_percent_min                                                    
   finetune_full_benchmark-v1.0.0-gpu_power_percent_avg                                                    
   finetune_full_benchmark-v1.0.0-gpu_power_percent_max                                                    
 ───────────────────────────────────────────────────────────────────────────────────────────────────────── 
```
<!-- markdownlint-enable line-length -->

You can see the required inputs under the section `Required Inputs` and the
optional inputs under `Optional Inputs and Default Values`. The next section
explains how to use optional properties.

> [!NOTE]
>
> The experiment gives the full domains it supports for each required and
> optional constitutive property. However, when constructing the `entityspace`
> you usually only want to use a sub-domain.

## Parameterizing Experiments

If an experiment has
[optional properties](../core-concepts/actuators.md#optional-properties) you can
define equivalent properties in the entity space. If you don't, the default
value for the property will be used.

In addition, you can define your own custom parameterization of the experiment.
For example, take the following experiment:

<!-- markdownlint-disable line-length -->
```terminaloutput
Identifier: robotic_lab.peptide_mineralization
Description: Measures adsorption of peptide lanthanide combinations

Required Inputs:
                                                                                                           
   Constitutive Properties:                                                                                
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: peptide_identifier                                                                        
     Description: The identifier of the peptide to use                                                     
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: ['test_peptide', 'test_peptide_new']                                                       
                                                                                                           
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: peptide_concentration                                                                     
     Description: The concentration of the peptide                                                         
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Values: [0.1, 0.4, 0.6, 0.8]                                                                       
                                                                                                           
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: lanthanide_concentration                                                                  
     Description: The concentration of lanthanide                                                          
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Values: [0.1, 0.4, 0.6, 0.8]                                                                       
                                                                                                           
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
                                                                                                           
Optional Inputs and Default Values:
                                                                                                           
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: temperature                                                                               
     Description: The temperature at which to execute the experiment                                       
     Domain:                                                                                               
                                                                                                           
        Type: CONTINUOUS_VARIABLE_TYPE                                                                     
        Range: [0, 100]                                                                                    
                                                                                                           
     Default value: 23                                                                                     
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: replicas                                                                                  
     Description: How many replicas to average the adsorption_timeseries over                              
     Domain:                                                                                               
                                                                                                           
        Type: DISCRETE_VARIABLE_TYPE                                                                       
        Interval: 1                                                                                        
        Range: [1, 4]                                                                                      
                                                                                                           
     Default value: 1                                                                                      
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     Identifier: robot_identifier                                                                          
     Description: The identifier of the robot to use to perform the experiment                             
     Domain:                                                                                               
                                                                                                           
        Type: CATEGORICAL_VARIABLE_TYPE                                                                    
        Values: ['harry', 'hermione']                                                                      
                                                                                                           
     Default value: 'hermione'                                                                             
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
                                                                                                           
Outputs:
 ───────────────────────────────────────────────────────────────────────────────────────────────────────── 
   peptide_mineralization-adsorption_timeseries                                                            
   peptide_mineralization-adsorption_plateau_value                                                         
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────
```
<!-- markdownlint-enable line-length -->

It has three optional properties: `temperature`, `robot_identifier` and
`replicas`.

### Example: Customizing an experiment

The default temperature is `23` degrees C, however imagine you want to run this
experiment at `30` degrees C. You can define a `discoveryspace` like:

```yaml
sampleStoreIdentifier: c04713
entitySpace:
  - identifier: peptide_identifier
    propertyDomain:
      values: ["test_peptide"]
  - identifier: peptide_concentration
    propertyDomain:
      values: [0.1, 0.4, 0.6, 0.8]
  - identifier: lanthanide_concentration
    propertyDomain:
      values: [0.1, 0.4, 0.6, 0.8]
experiments:
  - actuatorIdentifier: robotic_lab
    experimentIdentifier: peptide_mineralization
    parameterization:
      - value: 30
        property:
          identifier: "temperature"
metadata:
  description: Space for exploring the absorption properties of test_peptide
```

### Example: Multiple customizations of the same experiment

You can add the multiple custom parameterizations of the same experiment e.g.
one experiment that runs at 30 degrees C and another at 25 degrees.

```yaml
sampleStoreIdentifier: c04713 # PUT REAL ID HERE
entitySpace:
  - identifier: peptide_identifier
    propertyDomain:
      values: ["test_peptide"]
  - identifier: peptide_concentration
    propertyDomain:
      values: [0.1, 0.4, 0.6, 0.8]
  - identifier: lanthanide_concentration
    propertyDomain:
      values: [0.1, 0.4, 0.6, 0.8]
experiments:
  - actuatorIdentifier: robotic_lab
    experimentIdentifier: peptide_mineralization
    parameterization:
      - value: 30
        property:
          identifier: "temperature"
  - actuatorIdentifier: robotic_lab
    experimentIdentifier: peptide_mineralization
    parameterization:
      - value: 25
        property:
          identifier: "temperature"
metadata:
  description: Space for exploring the absorption properties of test_peptide
```

### Example: Using an optional property in the `entityspace`

Finally, if you want to scan a range of temperatures in your discovery space,
the best would be to move this parameter into the `entityspace`:

```yaml
sampleStoreIdentifier: c04713 # PUT REAL ID HERE
entitySpace:
  - identifier: peptide_identifier
    propertyDomain:
      values: ["test_peptide"]
  - identifier: peptide_concentration
    propertyDomain:
      values: [0.1, 0.4, 0.6, 0.8]
  - identifier: lanthanide_concentration
    propertyDomain:
      values: [0.1, 0.4, 0.6, 0.8]
  - identifier: temperature
    propertyDomain:
      domainRange: [20, 30]
      interval: 1
experiments:
  - actuatorIdentifier: robotic_lab
    experimentIdentifier: peptide_mineralization
metadata:
  description: Space for exploring the absorption properties of test_peptide
```

Here entities will be generated with a temperatures property that ranges from 20
to 30 degrees. When the experiment is run on the entity it will retrieve the
value of the temperature from it rather than the Experiment.

Our toy
[example actuator](https://github.com/IBM/ado/tree/main/plugins/actuators/example_actuator)
contains the above examples. You can use it to experiment and explore custom
parameterization.
