# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""AutoGluon preprocessing for PKH categorical features."""

import pandas as pd
from autogluon.common.features.types import R_OBJECT
from autogluon.features.generators import AutoMLPipelineFeatureGenerator


class PKHCategoricalFeatureGenerator(AutoMLPipelineFeatureGenerator):
    """Apply AutoGluon's feature transformations to PKH's category strings."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert raw category values to their training strings before prediction."""
        columns = self.feature_metadata_in.get_features(valid_raw_types=[R_OBJECT])
        frame = X.copy()
        values = frame[columns].astype(object)
        frame[columns] = values.where(values.notna(), None).map(str)
        return super().transform(frame)
