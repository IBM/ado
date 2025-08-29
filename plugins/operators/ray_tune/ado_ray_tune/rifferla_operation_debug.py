# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import orchestrator.core.discoveryspace.space
import orchestrator.metastore.project
from orchestrator.modules.operators.collections import modify

# datacol_lhs_v0
# on spark DB (sadly)
# STATE ID: space-3bfcf2-3870e2
# OPERATION ID: raytune-test-0.0.10.dev22+gca92ad0.d20230220-988e47
# ENTITY STORAGE: sqlsource_3870e2
# space_id = 'space-3bfcf2-3870e2'
# space_id = 'space-f06a38-9e5744'
# space_id = 'space-9be45b-94744d'
# space_id = 'space-f06a38-9e5744'  # ear caikit lhs v0
# space_id = 'space-2709e6-5092cb'  # ear caikit lhs v1 run 0
# space_id = 'space-664ed8-3fce0d'  # ear caikit lhs v1 run 1

space_id = "space-664ed8-3fce0d"  # ear caikit v2 atc
project = "caikit-testharness"
# space_id = 'space-635518-c9d490'
# project = 'spark'


if __name__ == "__main__":

    project_context = orchestrator.metastore.project.ProjectContext(
        project=project
    )  # from environment as default

    inputSpace = (
        orchestrator.core.discoveryspace.space.DiscoverySpace.from_stored_configuration(
            project_context=project_context, space_identifier=space_id
        )
    )

    metric = "latency95"
    # metric = 'wallClockRuntime'
    failed_metric = "valid_experiment"
    failed_value = 0.0
    mode = "min"
    # max_concurrent_trials = 1
    # mi_diff_limit = 0.2
    # samples_below_limit = 5
    # consider_pareto_front_convergence = True
    # min_mi_threshold = 0.8
    # min_mi_threshold = 0.5
    min_mi_threshold = 0.9
    newSpace = modify.reduce_space_with_mutual_information_analysis(
        inputSpace,
        metric=metric,
        failed_metric=failed_metric,
        failed_value=failed_value,
        mode=mode,
        min_mi_threshold=min_mi_threshold,
        # find_valid_intersection=True
    )

    print("placeholder")
