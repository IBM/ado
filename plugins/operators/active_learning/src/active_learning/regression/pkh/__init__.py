# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Predictive kernel herding (PKH): a regression-based active-learning strategy.

References
----------
.. [1] E. Aydar, C. Pinto, S. Venugopal, and D. Chatzopoulos, "Sampling
       Where It Matters: Predicting LLM Serving Performance with
       Predictive Kernel Herding," in Proceedings of the Sixth European
       Workshop on Machine Learning and Systems (EuroMLSys '26), ACM,
       2026, pp. 13-22. doi: 10.1145/3805621.3807633.
"""

from active_learning.regression.pkh.operator import pkh
from active_learning.regression.pkh.parameters import (
    PKHOperatorParameters,
    PKHParameters,
)
from active_learning.regression.pkh.sampler import PKHSampleSelector

__all__ = ["PKHOperatorParameters", "PKHParameters", "PKHSampleSelector", "pkh"]
