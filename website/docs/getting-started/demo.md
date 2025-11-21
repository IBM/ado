<!-- markdownlint-disable-next-line first-line-h1 -->
The following videos provide an overview of using ado to benchmark fine-tuning
performance across a range of fine-tuning workload configurations.

## List actuators and experiments

We begin by listing the experiments provided by the `SFTTrainer`
[actuator](../actuators/working-with-actuators.md), which provides fine-tuning
benchmarking capabilities. We can use `ado` to get the details of one of the
experiments `finetune_full_benchmark-v1.0.0` and see what it requires as input
and what it measures.

<!-- markdownlint-disable no-inline-html -->
<video controls preload="auto" poster="../videos/step1_trimmed_thumbnail.png">
<source src="../videos/step1_trimmed.mp4" type="video/mp4">
</video>
<!-- markdownlint-enable no-inline-html -->

## Create a `discoveryspace` to explore fine-tuning performance

Next we create a `discoveryspace` that represents a fine-tuning benchmarking
campaign. To get started quickly, we use `ado`s `template` functionality to
create a default configuration space for `lora` and `full` fine-tuning benchmark
experiments.

<!-- markdownlint-disable no-inline-html -->
<video controls preload="auto" poster="../videos/step2_trimmed_thumbnail.png">
<source src="../videos/step2_trimmed.mp4" type="video/mp4">
</video>
<!-- markdownlint-enable no-inline-html -->

## Explore the `discoveryspace` with a RandomWalk

This clip demonstrates how to view the available operators and then creating a
[RandomWalk](../operators/random-walk.md) [operation](../resources/operation.md)
to explore the discovery space created above. The operation is configured to
sample all 40 of the configurations, a.k.a.
[entities](../core-concepts/entity-spaces.md), in the `discoveryspace`. After
the operation is finished we can look results at a summary of the operation and
get the results as a CSV file.

<!-- markdownlint-disable no-inline-html -->
<video controls preload="auto" poster="../videos/step3_trimmed_thumbnail.png">
<source src="../videos/step3_trimmed.mp4" type="video/mp4">
</video>
<!-- markdownlint-enable no-inline-html -->

## Examine spaces collaborators have created

`ado` enables multiple distributed users to collaborate on the same project. Here
another user can query the `discoveryspaces` created by their colleagues,
including the one created earlier. Resources, like `discoveryspaces`,
can be annotated with custom metadata. For example, in this clip the user requests
a summary of all spaces tagged with `exp=ft`. They then apply a custom export
operator to the data which in this case integrates new data with an external
store in a rigorous and repeatable way.

<!-- markdownlint-disable no-inline-html -->
<video controls preload="auto" poster="../videos/step4_trimmed_thumbnail.png">
<source src="../videos/step4_trimmed.mp4" type="video/mp4">
</video>
<!-- markdownlint-enable no-inline-html -->