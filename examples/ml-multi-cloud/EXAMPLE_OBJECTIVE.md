# Search based on a custom objective function

> [!NOTE]
>
> This example shows how to create and use a custom objective function, an
> experiment which requires the output of another experiment, with `ado`.

## The scenario

Often, experiments will not directly produce the value that you are interested
in. For example, an experiment might measure the run time of an application,
while **the meaningful metric is the associated cost, which requires knowing
information like the cost per hour of the GPUs used**. Another common scenario
involves aggregating data points from one or more experiments into a single
value.

In this example we will install **a custom objective function that calculates a
cost** for the application workload configurations used in the
[taking a random walk example](/ado/examples/random-walk/). When the workload
configuration space is explored using a random walk, both the `wallClockRuntime`
and the `cost`, as defined by the custom function, will be measured.

> [!CAUTION]
>
> The commands below assume you are in the directory `examples/ml-multi-cloud`
> in **the ado source repository**. See
> [the instructions for cloning the repository](/ado/getting-started/install/#__tabbed_1_3).

## Prerequisites

### Install the ray_tune ado operator

If you haven't already installed the ray_tune operator, run:

```commandline
pip install ado-ray-tune
```

Then, execute

```commandline
ado get operators
```

should show an entry for `ray_tune` like below

```commandline
Available operators by type:
┌───────┬─────────────┬─────────┐
│ INDEX │ OPERATOR    │ TYPE    │
├───────┼─────────────┼─────────┤
│ 0     │ random_walk │ explore │
│ 1     │ ray_tune    │ explore │
│ 2     │ rifferla    │ modify  │
└───────┴─────────────┴─────────┘
```

## Installing the custom experiment

The custom experiment is defined in a Python package under
`custom_actuator_function/`. To install it run:

```commandline
pip install custom_experiment/
```

then

```commandline
ado get experiments --details
```

will output something similar to:

<!-- markdownlint-disable line-length -->
```commandline
┌────────────────────┬─────────────────────────┐
│ ACTUATOR ID        │ EXPERIMENT ID           │
├────────────────────┼─────────────────────────┤
│ custom_experiments │ ml-multicloud-cost-v1.0 │
│ mock               │ test-experiment         │
│ mock               │ test-experiment-two     │
└────────────────────┴─────────────────────────┘
```
<!-- markdownlint-enable line-length -->

You can see the custom experiment provided by the package,
**ml-multicloud-cost-v1.0** on the first line. Executing
`ado describe experiment ml-multicloud-cost-v1.0` outputs:

<!-- markdownlint-disable line-length -->
```terminaloutput
Identifier: custom_experiments.ml-multicloud-cost-v1.0

Required Inputs:
                                                                             
   Constitutive Properties:                                                  
    ─────────────────────────────────────────────────────────────────────    
     Identifier: nodes                                                       
     Domain:                                                                 
                                                                             
        Type: DISCRETE_VARIABLE_TYPE                                         
        Interval: 1                                                          
        Range: [0, 1000]                                                     
                                                                             
    ─────────────────────────────────────────────────────────────────────    
    ─────────────────────────────────────────────────────────────────────    
     Identifier: cpu_family                                                  
     Domain:                                                                 
                                                                             
        Type: DISCRETE_VARIABLE_TYPE                                         
        Values: [0, 1]                                                       
                                                                             
    ─────────────────────────────────────────────────────────────────────    
   Observed Properties:                                                      
                                                                             
      op-benchmark_performance-wallClockRuntime                              
                                                                             
                                                                             
Outputs:
 ─────────────────────────────────────────────────────────────────────────── 
   ml-multicloud-cost-v1.0-total_cost                                        
 ───────────────────────────────────────────────────────────────────────────
```
<!-- markdownlint-enable line-length -->

From this, you can see the `ml-multicloud-cost-v1.0` requires an observed
property, i.e. a property measured by another experiment, as input. From the
observed property identifier, the experiment is called `benchmark_performance`
and the property is `wallClockRuntime`.

## Create a discoveryspace that uses the custom experiment

First create a `samplestore` with the `ml-multi-cloud` example data following
[these instructions](/ado/examples/random-walk/#using-pre-existing-data-with-ado).
If you have already completed the
[taking a random walk example](/ado/examples/random-walk/), reuse the
`samplestore` you created there (use `ado get samplestores` if you cannot recall
the identifier).

To use the custom experiment, you must add it in the `experiments` list of a
`discoveryspace`. The `actuatorIdentifier` will be `custom_experiments` and the
`experimentIdentifier` will be the name of your experiment. For this case the
relevant section looks like:

```yaml
experiments:
  - experimentIdentifier: "benchmark_performance"
    actuatorIdentifier: "replay"
  - experimentIdentifier: "ml-multicloud-cost-v1.0"
    actuatorIdentifier: "custom_experiments"
```

The complete `discoveryspace` for this example is given in
`ml_multicloud_space_with_custom.yaml` To create it execute:

```commandline
ado create space -f ml_multicloud_space_with_custom.yaml --set "sampleStoreIdentifier=$SAMPLE_STORE_IDENTIFIER"
```

> [!IMPORTANT]
>
> If an experiment takes the output of another experiment as input
> both experiments must be in the `discoveryspace`. In the above example if the
> entry `benchmark_performance` was omitted the `ado create space` command would
> fail with:
>
> **SpaceInconsistencyError**: MeasurementSpace does not contain an experiment
> measuring an observed property required by another experiment in the space

You view a description of the space using the `ado describe` command:

```commandline
ado describe space --use-latest
```

This will output:

<!-- markdownlint-disable line-length -->
```terminaloutput
Identifier: 'space-19b2de-6da1f4'

Entity Space:
                                                                             
   Number of entities: 48                                                    
                                                                             
   Categorical properties:                                                   
                                                                             
      name       values                                                      
     ────────────────────────────                                            
      provider   ['A', 'B', 'C']                                             
                                                                             
   Discrete properties:                                                      
                                                                             
      name         range   interval   values                                 
     ──────────────────────────────────────────────                          
      cpu_family   None    None       [0, 1]                                 
      vcpu_size    None    None       [0, 1]                                 
      nodes        None    None       [2, 3, 4, 5]                           
                                                                             
                                                                             
Measurement Space:
                                                                             
   Experiments:                                                              
                                                                             
      experiment                                   supported                 
     ────────────────────────────────────────────────────────                
      replay.benchmark_performance                 True                      
      custom_experiments.ml-multicloud-cost-v1.0   True                      
                                                                             
    ─────────────────── replay.benchmark_performance ────────────────────    
     Inputs:                                                                 
                                                                             
        parameter    type       value   parameterized                        
       ───────────────────────────────────────────────                       
        cpu_family   required   None    na                                   
        nodes        required   None    na                                   
        provider     required   None    na                                   
        vcpu_size    required   None    na                                   
                                                                             
     Outputs:                                                                
                                                                             
        target property                                                      
       ──────────────────                                                    
        wallClockRuntime                                                     
        status                                                               
                                                                             
    ─────────────────────────────────────────────────────────────────────    
                                                                             
    ──────────── custom_experiments.ml-multicloud-cost-v1.0 ─────────────    
     Inputs:                                                                 
                                                                             
        parameter                    type       value   parameterized        
       ───────────────────────────────────────────────────────────────       
        nodes                        required   None    na                   
        cpu_family                   required   None    na                   
        benchmark_performance-wal…   required   None    na                   
                                                                             
     Outputs:                                                                
                                                                             
        target property                                                      
       ─────────────────                                                     
        total_cost                                                           
                                                                             
    ─────────────────────────────────────────────────────────────────────    
                                                                             
                                                                             
Sample Store identifier: 6da1f4                                                            
```
<!-- markdownlint-enable line-length -->

## Exploring the `discoveryspace`

To run a `randomwalk` operation on the new space, execute:

```commandline
ado create operation -f randomwalk_ml_multicloud_operation.yaml --use-latest space
```

This produces an output similar to that described in the
[taking a random walk example](/ado/examples/random-walk/#exploring-the-discoveryspace)
and will exit printing the operation identifier. However, in this case there is
additional information related to the dependent experiment.

When it completes, you can get a table of the points visited with:

```commandline
ado show entities operation --use-latest
```

You will see a table similar to the following - note the extra column for the
new cost function:

<!-- markdownlint-disable line-length -->
```commandline
┌───────────────┬──────────────┬───────────────┬───────────────┬────────────┬───────────────┬───────┬──────────┬───────────┬────────────────┬───────────────┬──────────────┬────────────────┬───────────────┬──────────────┬───────┐
│ request_index │ result_index │ identifier    │ experiment_id │ cpu_family │ generatorid   │ nodes │ provider │ vcpu_size │ reason         │ wallClockRun… │ status       │ total_cost     │ request_id    │ entity_index │ valid │
├───────────────┼──────────────┼───────────────┼───────────────┼────────────┼───────────────┼───────┼──────────┼───────────┼────────────────┼───────────────┼──────────────┼────────────────┼───────────────┼──────────────┼───────┤
│ 0             │ 0            │ provider.B-c… │ replay.bench… │ 0.0        │ explicit_gri… │ 3     │ B        │ 1.0       │ Externally     │ not_measured  │ not_measured │ not_measured   │ randomwalk-1… │ 0            │ False │
│               │              │               │               │            │               │       │          │           │ defined        │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ experiments    │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ cannot be      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ applied to     │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ entities:      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ replay.benchm… │               │              │                │               │              │       │
│ 1             │ 0            │ C_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ 92.171414375… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 1             │ 0            │ C_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ 100.97977471… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 1             │ 0            │ C_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.56031706598… │ 9d1c78        │ 0            │ True  │
│ 1             │ 0            │ C_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.80499374204… │ 2b5c63        │ 0            │ True  │
│ 2             │ 0            │ provider.B-c… │ replay.bench… │ 0.0        │ explicit_gri… │ 5     │ B        │ 1.0       │ Externally     │ not_measured  │ not_measured │ not_measured   │ randomwalk-1… │ 0            │ False │
│               │              │               │               │            │               │       │          │           │ defined        │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ experiments    │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ cannot be      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ applied to     │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ entities:      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ replay.benchm… │               │              │                │               │              │       │
│ 3             │ 0            │ C_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ 136.30710506… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 3             │ 0            │ C_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ 135.47050046… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 3             │ 0            │ C_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.78630847401… │ ce622d        │ 0            │ True  │
│ 3             │ 0            │ C_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.76306945747… │ df682d        │ 0            │ True  │
│ 4             │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ 103.90595746… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 4             │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ 112.70569872… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 4             │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ 113.88505148… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 4             │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.44313829806… │ 510954        │ 0            │ True  │
│ 4             │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.56535692678… │ adc911        │ 0            │ True  │
│ 4             │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.58173682623… │ 7dc57c        │ 0            │ True  │
│ 5             │ 0            │ A_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ 105.63729166… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 5             │ 0            │ A_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ 96.847161054… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 5             │ 0            │ A_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.93436921305… │ f422f1        │ 0            │ True  │
│ 5             │ 0            │ A_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.69019891818… │ ad4f2e        │ 0            │ True  │
│ 6             │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 2     │ B        │ 0.0       │ not_measured   │ 346.07099580… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 6             │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 2     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.84523328675… │ 884535        │ 0            │ True  │
│ 7             │ 0            │ provider.B-c… │ replay.bench… │ 1.0        │ explicit_gri… │ 4     │ B        │ 1.0       │ Externally     │ not_measured  │ not_measured │ not_measured   │ randomwalk-1… │ 0            │ False │
│               │              │               │               │            │               │       │          │           │ defined        │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ experiments    │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ cannot be      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ applied to     │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ entities:      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ replay.benchm… │               │              │                │               │              │       │
│ 8             │ 0            │ C_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ C        │ 1.0       │ not_measured   │ 309.84232401… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 8             │ 0            │ C_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.72134624454… │ 0add34        │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ 138.06051611… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ 150.94715046… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.91750716831… │ b22cce        │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 2.09648820095… │ 6af57d        │ 0            │ True  │
│ 10            │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ 202.48239731… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 10            │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ 193.55997109… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 10            │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 4.49960882928… │ 641651        │ 0            │ True  │
│ 10            │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 4.30133269098… │ 93c307        │ 0            │ True  │
│ 11            │ 0            │ C_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 2     │ C        │ 1.0       │ not_measured   │ 363.28567099… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 11            │ 0            │ C_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 2     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 4.03650745550… │ b2acce        │ 0            │ True  │
│ 12            │ 0            │ C_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 4     │ C        │ 1.0       │ not_measured   │ 114.01436853… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 12            │ 0            │ C_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 4     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.53365263409… │ 6a3f5f        │ 0            │ True  │
│ 13            │ 0            │ A_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ 151.58562421… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 13            │ 0            │ A_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ 155.02856159… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 13            │ 0            │ A_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.52642707029… │ 4454cf        │ 0            │ True  │
│ 13            │ 0            │ A_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.58380935986… │ dd3bb4        │ 0            │ True  │
│ 14            │ 0            │ A_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ A        │ 0.0       │ not_measured   │ 335.20851802… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 14            │ 0            │ A_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.86226954460… │ e2fc0a        │ 0            │ True  │
│ 15            │ 0            │ provider.B-c… │ replay.bench… │ 1.0        │ explicit_gri… │ 3     │ B        │ 1.0       │ Externally     │ not_measured  │ not_measured │ not_measured   │ randomwalk-1… │ 0            │ False │
│               │              │               │               │            │               │       │          │           │ defined        │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ experiments    │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ cannot be      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ applied to     │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ entities:      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ replay.benchm… │               │              │                │               │              │       │
│ 16            │ 0            │ A_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ 206.74496150… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 16            │ 0            │ A_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ 236.17150664… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 16            │ 0            │ A_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.44574935833… │ 44cb62        │ 0            │ True  │
│ 16            │ 0            │ A_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.93619177738… │ 490165        │ 0            │ True  │
│ 17            │ 0            │ A_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ 221.51019692… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 17            │ 0            │ A_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ 216.39412736… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 17            │ 0            │ A_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.84591830770… │ bd3d9c        │ 0            │ True  │
│ 17            │ 0            │ A_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.80328439474… │ 12ea47        │ 0            │ True  │
│ 18            │ 0            │ A_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ 135.91092538… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 18            │ 0            │ A_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ 117.94136571… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 18            │ 0            │ A_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.77530348300… │ 8ad2ad        │ 0            │ True  │
│ 18            │ 0            │ A_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.27614904774… │ 1a0042        │ 0            │ True  │
│ 19            │ 0            │ A_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ 84.453469991… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 19            │ 0            │ A_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ 86.230160951… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 19            │ 0            │ A_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.17296486099… │ 96c0ed        │ 0            │ True  │
│ 19            │ 0            │ A_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.19764112432… │ 6cd3dc        │ 0            │ True  │
│ 20            │ 0            │ C_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ 85.679467439… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 20            │ 0            │ C_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ 95.863260507… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 20            │ 0            │ C_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.18999260332… │ b2ddab        │ 0            │ True  │
│ 20            │ 0            │ C_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.33143417371… │ 55db9b        │ 0            │ True  │
│ 21            │ 0            │ B_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 2     │ B        │ 1.0       │ not_measured   │ 298.81930494… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 21            │ 0            │ B_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 2     │ B        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 3.32021449936… │ ffbb95        │ 0            │ True  │
│ 22            │ 0            │ C_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ 598.88346576… │ Timed out.   │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 22            │ 0            │ C_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ 244.33887457… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 22            │ 0            │ C_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 9.98139109611… │ 91632b        │ 0            │ True  │
│ 22            │ 0            │ C_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 4.07231457630… │ 9e378c        │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ 106.07093071… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ 130.30512285… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.47320737110… │ 559d56        │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.80979337294… │ f3a447        │ 0            │ True  │
│ 24            │ 0            │ A_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 2     │ A        │ 0.0       │ not_measured   │ 378.31657004… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 24            │ 0            │ A_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 2     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 4.20351744492… │ 221473        │ 0            │ True  │
│ 25            │ 0            │ A_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 2     │ A        │ 1.0       │ not_measured   │ 291.90445613… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 25            │ 0            │ A_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 2     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 3.24338284598… │ 43333e        │ 0            │ True  │
│ 26            │ 0            │ A_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 4     │ A        │ 0.0       │ not_measured   │ 145.12948369… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 26            │ 0            │ A_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 4     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.61254981888… │ d984de        │ 0            │ True  │
│ 27            │ 0            │ C_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ 154.98134708… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 27            │ 0            │ C_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ 168.34859228… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 27            │ 0            │ C_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.58302245140… │ 63c714        │ 0            │ True  │
│ 27            │ 0            │ C_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.80580987135… │ 3390c8        │ 0            │ True  │
│ 28            │ 0            │ A_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ 168.36590766… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 28            │ 0            │ A_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ 170.15659737… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 28            │ 0            │ A_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.40304923057… │ 409b9e        │ 0            │ True  │
│ 28            │ 0            │ A_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.41797164479… │ 06db5c        │ 0            │ True  │
│ 29            │ 0            │ C_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 4     │ C        │ 1.0       │ not_measured   │ 121.42492485… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 29            │ 0            │ C_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 4     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.34916583167… │ 022af8        │ 0            │ True  │
│ 30            │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ 220.19828414… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 30            │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ 273.71202731… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 30            │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.66997140248… │ 3b4610        │ 0            │ True  │
│ 30            │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 4.56186712185… │ 6bf4dd        │ 0            │ True  │
│ 31            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 0.0       │ not_measured   │ 225.17914223… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 31            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 0.0       │ not_measured   │ 228.14362454… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 31            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.25099523464… │ a302db        │ 0            │ True  │
│ 31            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.26746458080… │ 62e595        │ 0            │ True  │
│ 32            │ 0            │ C_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ 168.91636371… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 32            │ 0            │ C_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ 174.03356242… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 32            │ 0            │ C_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.40763636430… │ 8bfb84        │ 0            │ True  │
│ 32            │ 0            │ C_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.45027968684… │ f339ac        │ 0            │ True  │
│ 33            │ 0            │ provider.B-c… │ replay.bench… │ 1.0        │ explicit_gri… │ 5     │ B        │ 1.0       │ Externally     │ not_measured  │ not_measured │ not_measured   │ randomwalk-1… │ 0            │ False │
│               │              │               │               │            │               │       │          │           │ defined        │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ experiments    │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ cannot be      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ applied to     │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ entities:      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ replay.benchm… │               │              │                │               │              │       │
│ 34            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ 113.87676978… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 34            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ 132.54151201… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 34            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.26529744201… │ 91cc7f        │ 0            │ True  │
│ 34            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.47268346680… │ 16affd        │ 0            │ True  │
│ 35            │ 0            │ B_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 1.0       │ not_measured   │ 184.93504953… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 35            │ 0            │ B_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 1.0       │ not_measured   │ 166.74843192… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 35            │ 0            │ B_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.02741694185… │ 128ae6        │ 0            │ True  │
│ 35            │ 0            │ B_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 0.92638017733… │ a75803        │ 0            │ True  │
│ 36            │ 0            │ C_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ 240.07358503… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 36            │ 0            │ C_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ 269.09066414… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 36            │ 0            │ C_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 2.00061320861… │ 24e5c5        │ 0            │ True  │
│ 36            │ 0            │ C_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 2.24242220123… │ fa537f        │ 0            │ True  │
│ 37            │ 0            │ A_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ A        │ 1.0       │ not_measured   │ 272.99782156… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 37            │ 0            │ A_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.51665456427… │ 1c5976        │ 0            │ True  │
│ 38            │ 0            │ A_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 4     │ A        │ 0.0       │ not_measured   │ 158.70639538… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 38            │ 0            │ A_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 4     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.52680878639… │ 9e4db8        │ 0            │ True  │
│ 39            │ 0            │ C_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 4     │ C        │ 0.0       │ not_measured   │ 177.72359776… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 39            │ 0            │ C_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 4     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.94941328366… │ 70dc0f        │ 0            │ True  │
│ 40            │ 0            │ A_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 4     │ A        │ 1.0       │ not_measured   │ 116.31417059… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 40            │ 0            │ A_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 4     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.58475934664… │ e809c3        │ 0            │ True  │
│ 41            │ 0            │ C_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ C        │ 0.0       │ not_measured   │ 415.82928490… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 41            │ 0            │ C_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 2.31016269392… │ 9995f8        │ 0            │ True  │
│ 42            │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ 141.99024295… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 42            │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ 168.79178500… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 42            │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.94417341550… │ 875849        │ 0            │ True  │
│ 42            │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 4.68866069449… │ a0cf51        │ 0            │ True  │
│ 43            │ 0            │ C_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 2     │ C        │ 0.0       │ not_measured   │ 463.39653873… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 43            │ 0            │ C_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 2     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 5.14885043038… │ 3b6b15        │ 0            │ True  │
│ 44            │ 0            │ C_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 4     │ C        │ 0.0       │ not_measured   │ 188.09087824… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 44            │ 0            │ C_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 4     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 2.08989864720… │ 0be80e        │ 0            │ True  │
│ 45            │ 0            │ A_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 4     │ A        │ 1.0       │ not_measured   │ 106.67012143… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 45            │ 0            │ A_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 4     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.18522357145… │ 4ae232        │ 0            │ True  │
│ 46            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ 153.51639366… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 46            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ 184.44801592… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 46            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ 176.28814435… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 46            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.27930328051… │ 4b56b9        │ 0            │ True  │
│ 46            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.53706679940… │ d8119b        │ 0            │ True  │
│ 46            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.46906786958… │ 02acb3        │ 0            │ True  │
│ 47            │ 0            │ provider.B-c… │ replay.bench… │ 0.0        │ explicit_gri… │ 4     │ B        │ 1.0       │ Externally     │ not_measured  │ not_measured │ not_measured   │ randomwalk-1… │ 0            │ False │
│               │              │               │               │            │               │       │          │           │ defined        │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ experiments    │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ cannot be      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ applied to     │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ entities:      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ replay.benchm… │               │              │                │               │              │       │
└───────────────┴──────────────┴───────────────┴───────────────┴────────────┴───────────────┴───────┴──────────┴───────────┴────────────────┴───────────────┴──────────────┴────────────────┴───────────────┴──────────────┴───────┘
```
<!-- markdownlint-enable line-length -->

## Explore Further

- _Perform an optimization instead of a random walk_: See the
  [search a space with an optimizer example](/ado/examples/best-configuration-search).
- _Modify the objective function_: Try modifying the cost function and creating
  a new space - be careful to change the name of the experiment!
- _Create a custom experiment_: Explore
  [the documentation for writing your own custom experiment](/ado/actuators/creating-custom-experiments/)
- _Break the discoveryspace_: See what happens if you try to create the
  `discoveryspace` without the experiment that provides input to the cost
  function.
- _Examine the requests_: Run `ado show requests operation` to see what is
  replayed (`benchmark_performance`) and what is calculated
  (`ml_multicloud_cost-v1.0`)

## Key Takeaways

- **Dependent experiments**: `ado` allows you to define experiments which
  consume the output of other experiments.
  - There is no limit to the depth of the chain of dependent experiments.
  - Dependent experiments are executed when the required inputs are available.
- **Custom experiments**: You can add your own Python functions as experiments
  using `ado`'s custom experiments feature.
- **Uniform usage pattern**: How you use `ado` to define spaces or perform
  operations does not change if you use custom or dependent experiments.
