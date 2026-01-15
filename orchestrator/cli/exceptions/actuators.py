# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


class NoActuatorWithExperimentError(Exception):
    """
    This error is raised when there is no actuator associated with an experiment.
    """


class ActuatorDoesNotHaveExperimentError(Exception):
    """
    This error is raised when the actuator id provided by the user does not
    implement the experiment the user was looking for.
    """

    actuators_with_experiments: set

    def __init__(self, actuators: set) -> None:
        self.actuators_with_experiments = actuators


class TooManyActuatorsWithExperimentError(Exception):
    """
    This error is raised when multiple actuators are found with the experiment
    the user is looking for but the user did not specify which one to use.
    """

    actuators_with_experiments: set

    def __init__(self, actuators: set) -> None:
        self.actuators_with_experiments = actuators
