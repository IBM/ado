# Taking a random walk

> [!NOTE] The scenario
>
> When deploying a workload, you need to configure parameters such as the number
> of CPUs or the type of GPU. **In this example, `ado` is used to explore how
> performance varies across the workload parameter space for a cloud
> application.**
>
> Exploring a workload parameter space with `ado` involves:
>
> 1. Defining the values of the workload parameters to test and how to measure
>    them using a `discoveryspace`
> 2. Exploring the `discoveryspace` by creating an `operation` that samples
>    points and measures them
> 3. Getting the results of the `operation`

<!-- markdownlint-disable-next-line MD028 -->

> [!IMPORTANT] Prerequisites
>
> - Get the example files
>
> ```commandline
> git clone https://github.com/IBM/ado.git
> cd ado/examples/ml-multi-cloud
> ```
>
> - Install the following Python package locally:
>
> ```bash
> pip install ado-core
> ```
<!-- markdownlint-disable-next-line MD028 -->

<!-- markdownlint-disable line-length -->
> [!TIP] TL;DR
>
> To create the `discoveryspace` and explore it with a random walk execute:
>
> ```bash
> : # Create the space to explore
> ado create space -f ml_multicloud_space.yaml --with store=ml_multicloud_sample_store.yaml
> : # Explore!
> ado create op -f randomwalk_ml_multicloud_operation.yaml --use-latest space
> ```
>
<!-- markdownlint-enable line-length -->

## Using pre-existing data with `ado`

For this example we will use some **pre-existing data**. This makes the example
simpler and quicker to execute but can also be useful in other situations. The
data is in the file `ml_export.csv` and consists of results of running a
benchmark on different cloud hardware configurations from different providers.

In `ado` such configurations are called `entities`, and are stored, along with
the results of measurements executed on them, in a
[`samplestore`](/ado/resources/sample-stores). Let's start by copying the data
in `ml_export.csv` into a new `samplestore`.

To do this execute,

```commandline
ado create store -f ml_multicloud_sample_store.yaml
```

and it will report that a `samplestore` has been created:

```commandline
Success! Created sample store with identifier $SAMPLE_STORE_IDENTIFIER
```

You can see all available sample stores using `ado get samplestores`.

!!! info end
    <!-- markdownlint-disable-next-line code-block-style -->
    You only need to create this `samplestore` once.
    It can be reused in multiple `discoveryspaces`
    or examples that require the `ml_export.csv` data.

## Creating a `discoveryspace` for the `ml-multi-cloud` data

A `discoveryspace` describes a set of points and how to measure them. Here we
will create a `discoveryspace` to describe the space explored in
`ml_export.csv`.

Execute:

```commandline
ado create space -f ml_multicloud_space.yaml --use-latest samplestore
```

This will confirm the creation of the `discoveryspace` with:

```commandline
Success! Created space with identifier: $DISCOVERY_SPACE_IDENTIFIER
```

You can now describe the `discoveryspace` with:

```commandline
ado describe space --use-latest
```

This will output:

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

> [!NOTE]
>
> The set of points is defined by the properties in the `Entity Space` - here
> '_cpu_family_', '_provider_', '_vcpu_size_' and '_nodes_' - and the values
> those properties can take.
<!-- markdownlint-disable-next-line no-blanks-blockquote -->
> [!TIP]
> Consider why the size of the entityspace is 48. Compare this to the
> number of rows in `ml_export.csv`.

## Exploring the `discoveryspace`

Next we will run an operation that will "explore" the `discoveryspace` we just
created. Since we already have the data, `ado` will transparently identify and
reuse it. An example operation file is given in
`randomwalk_ml_multicloud_operation.yaml`. The contents are:

```yaml
{% include "./randomwalk_ml_multicloud_operation.yaml" %}
```

To run the operation execute:

```commandline
ado create operation -f randomwalk_ml_multicloud_operation.yaml --use-latest space
```

This will output a lot of information as it samples all the entities. Typically,
you will see the following lines for each entity (point in the entity space)
sampled and measured:

<!-- markdownlint-disable line-length -->
```commandline
(RandomWalk pid=14797) Continuous batching: SUBMIT EXPERIMENT. Submitting experiment replay.benchmark_performance for provider.B-cpu_family.1-vcpu_size.1-nodes.4
(RandomWalk pid=14797)
(RandomWalk pid=14797) Continuous batching: SUMMARY. Entities sampled and submitted: 2. Experiments completed: 1 Waiting on 1 active requests. There are 0 dependent experiments
(RandomWalk pid=14797) Continuous Batching: EXPERIMENT COMPLETION. Received finished notification for experiment in measurement request in group 1: request-randomwalk-0.9.6.dev91+884f713b.dirty-c5ed4b-579021-experiment-benchmark_performance-entities-provider.B-cpu_family.1-vcpu_size.1-nodes.4 (explicit_grid_sample_generator)-requester-randomwalk-0.9.6.dev91+884f713b.dirty-c5ed4b-time-2025-07-29 20:03:00.976809+01:00
```
<!-- markdownlint-enable line-length -->

The first line, "SUBMIT EXPERIMENT", indicates the entity -
`provider.B-cpu_family.1-vcpu_size.1-nodes.4` - and experiment -
`replay.benchmark_performance` submitted. The next line gives a summary of what
has happened so far: this is the second entity sampled and submitted; one
experiment has completed; and the sampler is waiting on one active experiment
before submitting a new one. Finally, the "EXPERIMENT COMPLETION" line indicates
the experiment has finished.

The operation will end with information like:

```yaml
config:
  operation:
    module:
      moduleClass: RandomWalk
      moduleName: orchestrator.modules.operators.randomwalk
      modulePath: .
      moduleType: operation
    parameters:
      batchSize: 1
      numberEntities: 48
      samplerConfig:
        mode: sequential
        samplerType: generator
  spaces:
    - space-65cf33-a8df39
created: "2025-06-20T13:03:46.763154Z"
identifier: randomwalk-0.9.4.dev30+564196d4.dirty-b8a233
kind: operation
metadata:
  entities_submitted: 48
  experiments_requested: 74
operationType: search
operatorIdentifier: randomwalk-0.9.4.dev30+564196d4.dirty
status:
  - event: created
    recorded_at: "2025-06-20T13:03:40.267005Z"
  - event: added
    recorded_at: "2025-06-20T13:03:46.764750Z"
  - event: started
    recorded_at: "2025-06-20T13:03:46.769169Z"
  - event: finished
    exit_state: success
    recorded_at: "2025-06-20T13:03:48.369516Z"
  - event: updated
    recorded_at: "2025-06-20T13:03:48.374765Z"
version: v1
```

The identifier operation is stored in the `identifier` field: in the output
above, it is `randomwalk-0.9.4.dev30+564196d4.dirty-b8a233`.

> [!NOTE]
>
> The operation "reuses" existing measurements: this is an `ado` feature called
> **memoization**.
>
> `ado` transparently executes experiments or memoizes data as appropriate - so
> the operator does not need to know if a measurement needs to be performed at
> the time it requests it, or if previous data can be reused.
<!-- markdownlint-disable-next-line no-blanks-blockquote -->
> [!TIP]
>
> Operations are **domain agnostic**. If you look in
> `randomwalk_ml_multicloud_operation.yaml` you will see there is no reference
> to characteristics of the discoveryspace we created. Indeed, this operation
> file could work on any discoveryspace.
>
> This shows that operators, like randomwalk, don't have to know domain specific
> details. All information about what to explore and how to measure is captured
> in the `discoveryspace`.

## Looking at the `operation` output

The command

```commandline
ado show entities operation --use-latest
```

displays the results of the operation i.e. the entities sampled and the
measurement results. You will see something like the following (the sampling is
random so the order can be different):

<!-- markdownlint-disable line-length -->
```text
               result_index                                   identifier                        benchmark_performance-wallClockRuntime benchmark_performance-status                                                                                        reason  valid
request_index
0                         0                               C_f1.0-c1.0-n4                                                    114.014369                           ok                                                                                                 True
1                         0                               A_f0.0-c0.0-n2                                                    335.208518                           ok                                                                                                 True
2                         0  provider.B-cpu_family.0-vcpu_size.1-nodes.5                                                                                             Externally defined experiments cannot be applied to entities: replay.benchmark_performance.   False
3                         0                               C_f1.0-c0.0-n4                                                    177.723598                           ok                                                                                                 True
4                         0                               B_f1.0-c0.0-n5                      [168.79178500175476, 141.99024295806885]                     [ok, ok]                                                                                                 True
5                         0                               A_f1.0-c1.0-n4                                                    116.314171                           ok                                                                                                 True
6                         0                               C_f1.0-c1.0-n2                                                    363.285671                           ok                                                                                                 True
7                         0                               A_f0.0-c0.0-n5                       [106.0709307193756, 130.30512285232544]                     [ok, ok]                                                                                                 True
8                         0                               C_f0.0-c0.0-n5                        [150.9471504688263, 138.0605161190033]                     [ok, ok]                                                                                                 True
9                         0                               B_f1.0-c0.0-n4                      [202.48239731788635, 193.55997109413147]                     [ok, ok]                                                                                                 True
10                        0                               C_f0.0-c0.0-n2                                                    415.829285                           ok                                                                                                 True
11                        0                               B_f0.0-c0.0-n4                       [113.87676978111269, 132.5415120124817]                     [ok, ok]                                                                                                 True
12                        0                               C_f1.0-c0.0-n2                                                    463.396539                           ok                                                                                                 True
13                        0                               A_f1.0-c1.0-n5                        [96.8471610546112, 105.63729166984558]                     [ok, ok]                                                                                                 True
14                        0                               A_f0.0-c0.0-n3                         [221.5101969242096, 216.394127368927]                     [ok, ok]                                                                                                 True
15                        0                               B_f1.0-c1.0-n2                                                    298.819305                           ok                                                                                                 True
16                        0                               C_f1.0-c1.0-n3                       [154.9813470840454, 168.34859228134155]                     [ok, ok]                                                                                                 True
17                        0                               C_f0.0-c1.0-n2                                                    309.842324                           ok                                                                                                 True
18                        0  provider.B-cpu_family.1-vcpu_size.1-nodes.3                                                                                             Externally defined experiments cannot be applied to entities: replay.benchmark_performance.   False
19                        0                               B_f0.0-c0.0-n5   [113.88505148887634, 103.90595746040344, 112.7056987285614]                 [ok, ok, ok]                                                                                                 True
20                        0                               C_f0.0-c1.0-n3                        [168.9163637161255, 174.0335624217987]                     [ok, ok]                                                                                                 True
21                        0                               B_f0.0-c0.0-n2                       [228.14362454414368, 225.1791422367096]                     [ok, ok]                                                                                                 True
22                        0                               B_f0.0-c1.0-n2                        [166.74843192100525, 184.935049533844]                     [ok, ok]                                                                                                 True
23                        0  provider.B-cpu_family.1-vcpu_size.1-nodes.5                                                                                             Externally defined experiments cannot be applied to entities: replay.benchmark_performance.   False
24                        0                               B_f1.0-c0.0-n2                                                    346.070996                           ok                                                                                                 True
25                        0                               C_f0.0-c0.0-n4                                                    188.090878                           ok                                                                                                 True
26                        0                               A_f1.0-c1.0-n2                                                    291.904456                           ok                                                                                                 True
27                        0                               C_f1.0-c0.0-n3                       [244.33887457847595, 598.8834657669067]             [ok, Timed out.]                                                                                                 True
28                        0                               A_f0.0-c1.0-n2                                                    272.997822                           ok                                                                                                 True
29                        0  provider.B-cpu_family.1-vcpu_size.1-nodes.4                                                                                             Externally defined experiments cannot be applied to entities: replay.benchmark_performance.   False
30                        0                               A_f1.0-c1.0-n3                      [155.02856159210205, 151.58562421798706]                     [ok, ok]                                                                                                 True
31                        0                               A_f0.0-c1.0-n4                                                    106.670121                           ok                                                                                                 True
32                        0                               A_f1.0-c0.0-n3                       [206.74496150016785, 236.1715066432953]                     [ok, ok]                                                                                                 True
33                        0                               C_f0.0-c0.0-n3                       [269.0906641483307, 240.07358503341675]                     [ok, ok]                                                                                                 True
34                        0                               A_f1.0-c0.0-n2                                                     378.31657                           ok                                                                                                 True
35                        0  provider.B-cpu_family.0-vcpu_size.1-nodes.3                                                                                             Externally defined experiments cannot be applied to entities: replay.benchmark_performance.   False
36                        0                               B_f1.0-c0.0-n3                       [273.7120273113251, 220.19828414916992]                     [ok, ok]                                                                                                 True
37                        0                               A_f1.0-c0.0-n4                                                    158.706395                           ok                                                                                                 True
38                        0                               A_f0.0-c0.0-n4                                                    145.129484                           ok                                                                                                 True
39                        0                               A_f0.0-c1.0-n3                      [170.15659737586975, 168.36590766906738]                     [ok, ok]                                                                                                 True
40                        0                               A_f0.0-c1.0-n5                        [86.23016095161438, 84.45346999168396]                     [ok, ok]                                                                                                 True
41                        0                               B_f0.0-c0.0-n3  [184.44801592826843, 153.51639366149902, 176.28814435005188]                 [ok, ok, ok]                                                                                                 True
42                        0                               C_f1.0-c1.0-n5                       [100.97977471351624, 92.17141437530518]                     [ok, ok]                                                                                                 True
43                        0  provider.B-cpu_family.0-vcpu_size.1-nodes.4                                                                                             Externally defined experiments cannot be applied to entities: replay.benchmark_performance.   False
44                        0                               C_f1.0-c0.0-n5                       [136.3071050643921, 135.47050046920776]                     [ok, ok]                                                                                                 True
45                        0                               C_f0.0-c1.0-n4                                                    121.424925                           ok                                                                                                 True
46                        0                               A_f1.0-c0.0-n5                      [117.94136571884157, 135.91092538833618]                     [ok, ok]                                                                                                 True
47                        0                               C_f0.0-c1.0-n5                        [95.86326050758362, 85.67946743965149]                     [ok, ok]                                                                                                 True
```
<!-- markdownlint-enable line-length -->

> [!TIP] Some things to note and consider:
>
> - The table is in the order the points were measured
> - Some points have multiple measurements c.f. size of entityspace versus the
>   number of rows in `ml_export.csv`.
> - Some points were not measured - these are points in the discoveryspace for
>   which no data was present to replay.

## Exploring Further

Here are a variety of commands you can try after executing the example above:

### Viewing entities

There are multiple ways to view the entities related to a `discoveryspace`. Try:

```commandline
ado show entities space --use-latest
ado show entities space --use-latest --aggregate mean
ado show entities space --use-latest --include unmeasured
ado show entities space --use-latest --property-format target
```

Also,

```commandline
ado show details space --use-latest
```

will give you a summary of what has been measured.

> [!NOTE]
>
> If you want to run these commands with the latest space created
> use the `--use-latest` flag as above

### Resource provenance

The `related` sub-command shows resource provenance:

```commandline
ado show related operation --use-latest
```

### Operation timeseries

The following commands give more details of the operation timeseries:

```commandline
ado show results operation --use-latest
ado show requests operation --use-latest
```

### Resource templates

Another helpful command is `template` which will output a default example of a
resource YAML along with an (optional) description of its fields. Try:

```commandline
ado template operation --include-schema --operator-name random_walk
```

### Rerun

An interesting thing to try is to run the operation again and compare the output
of `show entities operation` for the two operations, and `show entities space`.

## Takeaways

- **create-explore-view pattern**: A common pattern in `ado` is to create a
  `discoveryspace` to describe a set of points to measure, create `operations`
  on it to explore or analyse it, and then view the results.
- **entity space and measurement space**: A `discoveryspace` consists of an
  `entityspace` - the set of points to measure - and a `measurementspace` - the
  set of experiments to apply to them.
- **operations are domain agnostic**: `ado` enables operations to run on
  multiple different domains without modification.
- **memoization**: By default `ado` will identify if a measurement has already
  been completed on an entity and reuse it
- **provenance**: `ado` stores the relationship between the resources it creates.
- **results viewing**: `ado show entities` outputs the data in a
  `discoveryspace` or measured in an `operation`.
- **measurement timeseries**: The sequence (timeseries) of measurements,
  successful or not, of each `operation` is preserved.
- **`discoveryspace` views**: By default `ado show entities space` only shows
  successfully measured entities, but you can see what has not been measured if
  you want.
