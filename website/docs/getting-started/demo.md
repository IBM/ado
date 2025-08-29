---
status: published #Status can be draft, reviewed or published.
---
The following videos give a glimpse of using ado to benchmark fine-tuning performance across a range of fine-tuning workload configurations. 

## List actuators and experiments

First we list the experiments given by the `SFTTrainer` [actuator](../actuators/working-with-actuators.md), which provides fine-tuning benchmarking capabilities.
We can use `ado` to get the details of one of the experiments `finetune_full_benchmark-v1.0.0` and see what it requires as input and what it measures.

<video controls preload="auto" poster="../videos/step1_trimmed_thumbnail.png">
<source src="../videos/step1_trimmed.mp4" type="video/mp4">
</video>


## Create a `discoveryspace` to explore fine-tuning performance

Next we create a `discoveryspace` that represents a fine-tuning benchmarking campaign.
For quick start we use `ado`s `template` functionality to create a default configuration space
for `lora` and `full` fine-tuning benchmark experiments.

<video controls preload="auto" poster="../videos/step2_trimmed_thumbnail.png">
<source src="../videos/step2_trimmed.mp4" type="video/mp4">
</video>



## Explore the `discoveryspace` with a RandomWalk

This clip shows looking at the available operators and then creating a [RandomWalk](../operators/random-walk.md) [operation](../resources/operation.md) to explore
the discovery space created above. 
The operation is configured to sample all 40 of the configurations, a.k.a. [entities](../core-concepts/entity-spaces.md), in the `discoveryspace`. 
After the operation is finished we can look results at a summary of the operation and get the results as a CSV file. 

<video controls preload="auto" poster="../videos/step3_trimmed_thumbnail.png">
<source src="../videos/step3_trimmed.mp4" type="video/mp4">
</video>


## Examine spaces collaborators have created

`ado` enables multiple distributed users can work on the same project. 
Here another user can query the `discoveryspaces` created by their colleagues, including in this case the one created above.
Resources, like `discoveryspaces`, can be tagged with custom metadata. 
For example, in this clip the user requests a summary of all spaces tagged with `exp=ft`.
They then apply a custom export operator to the data which in this case integrates new data with an external store in a rigorous and repeatable way. 

<video controls preload="auto" poster="../videos/step4_trimmed_thumbnail.png">
<source src="../videos/step4_trimmed.mp4" type="video/mp4">
</video>