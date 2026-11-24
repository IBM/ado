# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Feature encoding and discovery-space data helpers for PKH."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from active_learning.data import (
    candidate_entities,
    fit_feature_encoder,
    measured_labels,
    target_label,
    validate_target_output,
)
from ado.core.discoveryspace.space import DiscoverySpace, Entity
from ado.schema.domain import VariableTypeEnum

__all__ = [
    "candidate_entities",
    "encode_feature_matrix",
    "measured_labels",
    "target_label",
    "validate_target_output",
]


def encode_feature_matrix(
    discovery_space: DiscoverySpace,
    entities: list[Entity],
) -> tuple[pd.DataFrame, np.ndarray, ColumnTransformer]:
    """Encode constitutive values in the discovery space's declared feature order."""
    properties = discovery_space.entitySpace.constitutiveProperties
    raw_columns = {
        prop.identifier: [
            entity.valueForConstitutivePropertyIdentifier(prop.identifier).value
            for entity in entities
        ]
        for prop in properties
    }
    categorical_columns = [
        prop.identifier
        for prop in properties
        if prop.propertyDomain.variableType
        == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE
        or any(not isinstance(v, (int, float)) for v in raw_columns[prop.identifier])
    ]
    frame = pd.DataFrame(
        {
            prop.identifier: (
                pd.Series(
                    [str(value) for value in raw_columns[prop.identifier]],
                    dtype="category",
                )
                if prop.identifier in categorical_columns
                else pd.Series(raw_columns[prop.identifier])
            )
            for prop in properties
        }
    )
    categorical_encoder = make_pipeline(
        FunctionTransformer(
            pd.DataFrame.map,
            kw_args={"func": str},
            feature_names_out="one-to-one",
        ),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    encoder = ColumnTransformer(
        [("categorical", categorical_encoder, categorical_columns)],
        remainder="passthrough",
    )
    return fit_feature_encoder(frame, encoder)
