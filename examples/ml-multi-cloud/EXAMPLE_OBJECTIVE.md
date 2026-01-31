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
      OPERATOR     TYPE
0  random_walk  explore
1     ray_tune  explore
```

## Installing the custom experiment

The custom experiment is defined in a Python package under
`custom_actuator_function/`. To install it run:

```commandline
pip install custom_experiment/
```

then

```commandline
ado get actuators --details
```

will output something similar to:

<!-- markdownlint-disable line-length -->
```commandline
2          custom_experiments          CustomExperiments                        ml-multicloud-cost-v1.0       True
3         molecule-embeddings                 Embeddings                   calculate-morgan-fingerprint       True
4          molformer-toxicity         molformer-toxicity                               predict-toxicity       True
5                     mordred         Mordred Descriptor                  mordred-descriptor-calculator       True
6                       st4sd                      ST4SD                      toxicity-prediction-opera       True
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
               result_index                                   identifier                        benchmark_performance-wallClockRuntime benchmark_performance-status                           ml-multicloud-cost-v1.0-total_cost                                                                                        reason  valid
request_index
0                         0                               A_f0.0-c0.0-n3                         [221.5101969242096, 216.394127368927]                     [ok, ok]                     [1.8459183077017467, 1.8032843947410584]                                                                                                 True
1                         0                               C_f0.0-c0.0-n5                        [150.9471504688263, 138.0605161190033]                     [ok, ok]                     [2.0964882009559207, 1.9175071683194902]                                                                                                 True
2                         0                               A_f0.0-c0.0-n5                       [106.0709307193756, 130.30512285232544]                     [ok, ok]                      [1.473207371102439, 1.8097933729489646]                                                                                                 True
3                         0                               C_f1.0-c1.0-n5                       [100.97977471351624, 92.17141437530518]                     [ok, ok]                      [2.8049937420421176, 2.560317065980699]                                                                                                 True
4                         0                               C_f1.0-c0.0-n5                       [136.3071050643921, 135.47050046920776]                     [ok, ok]                       [3.786308474010892, 3.763069457477993]                                                                                                 True
5                         0                               A_f0.0-c1.0-n2                                                    272.997822                           ok                                                     1.516655                                                                                                 True
6                         0                               C_f1.0-c1.0-n4                                                    114.014369                           ok                                                     2.533653                                                                                                 True
7                         0                               B_f1.0-c1.0-n2                                                    298.819305                           ok                                                     3.320214                                                                                                 True
8                         0                               C_f0.0-c0.0-n4                                                    188.090878                           ok                                                     2.089899                                                                                                 True
9                         0                               A_f1.0-c1.0-n2                                                    291.904456                           ok                                                     3.243383                                                                                                 True
10                        0                               B_f0.0-c0.0-n3  [184.44801592826843, 153.51639366149902, 176.28814435005188]                 [ok, ok, ok]  [1.537066799402237, 1.2793032805124918, 1.4690678695837658]                                                                                                 True
11                        0                               B_f0.0-c1.0-n2                        [166.74843192100525, 184.935049533844]                     [ok, ok]                     [0.9263801773389181, 1.0274169418546888]                                                                                                 True
12                        0                               C_f0.0-c0.0-n2                                                    415.829285                           ok                                                     2.310163                                                                                                 True
13                        0  provider.B-cpu_family.1-vcpu_size.1-nodes.3                                                                                                                                                          Externally defined experiments cannot be applied to entities: replay.benchmark_performance.   False
14                        0                               A_f0.0-c1.0-n4                                                    106.670121                           ok                                                     1.185224                                                                                                 True
15                        0                               B_f1.0-c0.0-n3                       [273.7120273113251, 220.19828414916992]                     [ok, ok]                       [4.561867121855418, 3.669971402486165]                                                                                                 True
16                        0                               B_f1.0-c0.0-n2                                                    346.070996                           ok                                                     3.845233                                                                                                 True
17                        0                               B_f0.0-c0.0-n5   [113.88505148887634, 103.90595746040344, 112.7056987285614]                 [ok, ok, ok]  [1.5817368262343936, 1.443138298061159, 1.5653569267855751]                                                                                                 True
18                        0                               C_f1.0-c1.0-n2                                                    363.285671                           ok                                                     4.036507                                                                                                 True
19                        0                               C_f0.0-c1.0-n5                        [95.86326050758362, 85.67946743965149]                     [ok, ok]                      [1.331434173716439, 1.1899926033284929]                                                                                                 True
20                        0                               C_f1.0-c1.0-n3                       [154.9813470840454, 168.34859228134155]                     [ok, ok]                      [2.583022451400757, 2.8058098713556925]                                                                                                 True
21                        0                               B_f0.0-c0.0-n2                       [228.14362454414368, 225.1791422367096]                     [ok, ok]                      [1.267464580800798, 1.2509952346483866]                                                                                                 True
22                        0  provider.B-cpu_family.0-vcpu_size.1-nodes.4                                                                                                                                                          Externally defined experiments cannot be applied to entities: replay.benchmark_performance.   False
23                        0                               C_f0.0-c1.0-n4                                                    121.424925                           ok                                                     1.349166                                                                                                 True
24                        0                               A_f0.0-c0.0-n2                                                    335.208518                           ok                                                      1.86227                                                                                                 True
25                        0                               A_f1.0-c1.0-n4                                                    116.314171                           ok                                                     2.584759                                                                                                 True
26                        0                               C_f0.0-c1.0-n2                                                    309.842324                           ok                                                     1.721346                                                                                                 True
27                        0                               A_f1.0-c1.0-n3                      [155.02856159210205, 151.58562421798706]                     [ok, ok]                      [2.5838093598683676, 2.526427070299784]                                                                                                 True
28                        0                               C_f1.0-c0.0-n2                                                    463.396539                           ok                                                      5.14885                                                                                                 True
29                        0                               A_f0.0-c0.0-n4                                                    145.129484                           ok                                                      1.61255                                                                                                 True
30                        0                               A_f1.0-c0.0-n3                       [206.74496150016785, 236.1715066432953]                     [ok, ok]                      [3.4457493583361307, 3.936191777388255]                                                                                                 True
31                        0  provider.B-cpu_family.0-vcpu_size.1-nodes.3                                                                                                                                                          Externally defined experiments cannot be applied to entities: replay.benchmark_performance.   False
32                        0                               C_f0.0-c0.0-n3                       [269.0906641483307, 240.07358503341675]                     [ok, ok]                       [2.242422201236089, 2.000613208611806]                                                                                                 True
33                        0                               B_f0.0-c0.0-n4                       [113.87676978111269, 132.5415120124817]                     [ok, ok]                      [1.265297442012363, 1.4726834668053521]                                                                                                 True
34                        0                               A_f1.0-c0.0-n4                                                    158.706395                           ok                                                     3.526809                                                                                                 True
35                        0                               C_f0.0-c1.0-n3                        [168.9163637161255, 174.0335624217987]                     [ok, ok]                     [1.4076363643010457, 1.4502796868483225]                                                                                                 True
36                        0  provider.B-cpu_family.1-vcpu_size.1-nodes.5                                                                                                                                                          Externally defined experiments cannot be applied to entities: replay.benchmark_performance.   False
37                        0                               B_f1.0-c0.0-n4                      [202.48239731788635, 193.55997109413147]                     [ok, ok]                       [4.499608829286363, 4.301332690980699]                                                                                                 True
38                        0                               C_f1.0-c0.0-n4                                                    177.723598                           ok                                                     3.949413                                                                                                 True
39                        0                               B_f1.0-c0.0-n5                      [168.79178500175476, 141.99024295806885]                     [ok, ok]                       [4.688660694493188, 3.944173415501912]                                                                                                 True
40                        0  provider.B-cpu_family.0-vcpu_size.1-nodes.5                                                                                                                                                          Externally defined experiments cannot be applied to entities: replay.benchmark_performance.   False
41                        0                               C_f1.0-c0.0-n3                       [244.33887457847595, 598.8834657669067]             [ok, Timed out.]                      [4.0723145763079325, 9.981391096115113]                                                                                                 True
42                        0                               A_f1.0-c1.0-n5                        [96.8471610546112, 105.63729166984558]                     [ok, ok]                      [2.6901989181836448, 2.934369213051266]                                                                                                 True
43                        0                               A_f0.0-c1.0-n3                      [170.15659737586975, 168.36590766906738]                     [ok, ok]                     [1.4179716447989146, 1.4030492305755615]                                                                                                 True
44                        0  provider.B-cpu_family.1-vcpu_size.1-nodes.4                                                                                                                                                          Externally defined experiments cannot be applied to entities: replay.benchmark_performance.   False
45                        0                               A_f1.0-c0.0-n2                                                     378.31657                           ok                                                     4.203517                                                                                                 True
46                        0                               A_f0.0-c1.0-n5                        [86.23016095161438, 84.45346999168396]                     [ok, ok]                     [1.1976411243279776, 1.1729648609956105]                                                                                                 True
47                        0                               A_f1.0-c0.0-n5                      [117.94136571884157, 135.91092538833618]                     [ok, ok]                       [3.276149047745599, 3.775303483009338]                                                                                                 True
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
