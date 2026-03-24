# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""
Test Data Documentation for High-Dimensional Sampling Tests

This module documents the structure and purpose of test dataframes used in
test_high_dimensional_sampling.py. Each test dataframe is designed to validate
specific aspects of the sampling and ordering functionality.
"""

# %%
import numpy as np
import pandas as pd


def create_simple_2d_dataframe() -> pd.DataFrame:
    """
    Create a simple 2D dataframe for basic testing.

    Structure:
    - Constitutive properties: ['param_a', 'param_b']
    - param_a: 3 unique values [0, 1, 2]
    - param_b: 2 unique values ['x', 'y']
    - Total configurations: 3 x 2 = 6
    - Additional column: 'metric' (float values)

    Use case:
    - Basic functionality testing
    - Verifying correct ordering and sampling
    - Testing all sampling strategies on small dataset

    Returns:
        pd.DataFrame with 6 rows representing all combinations
    """
    data = {
        "param_a": [0, 0, 1, 1, 2, 2],
        "param_b": ["x", "y", "x", "y", "x", "y"],
        "metric": [10.5, 20.3, 15.7, 25.1, 18.9, 22.4],
    }
    return pd.DataFrame(data)


def create_3d_dataframe() -> pd.DataFrame:
    """
    Create a 3D dataframe for testing higher dimensionality.

    Structure:
    - Constitutive properties: ['batch_size', 'learning_rate', 'optimizer']
    - batch_size: 4 unique values [16, 32, 64, 128]
    - learning_rate: 3 unique values [0.001, 0.01, 0.1]
    - optimizer: 2 unique values ['adam', 'sgd']
    - Total configurations: 4 x 3 x 2 = 24
    - Additional columns: 'accuracy', 'loss'

    Use case:
    - Testing realistic ML hyperparameter space
    - Verifying sampling strategies on moderate-sized space
    - Testing numeric and categorical mixed types

    Returns:
        pd.DataFrame with 24 rows representing all combinations
    """
    import itertools

    batch_sizes = [16, 32, 64, 128]
    learning_rates = [0.001, 0.01, 0.1]
    optimizers = ["adam", "sgd"]

    combinations = list(itertools.product(batch_sizes, learning_rates, optimizers))

    data = {
        "batch_size": [c[0] for c in combinations],
        "learning_rate": [c[1] for c in combinations],
        "optimizer": [c[2] for c in combinations],
        "accuracy": np.random.RandomState(42).uniform(0.7, 0.95, len(combinations)),
        "loss": np.random.RandomState(43).uniform(0.1, 0.5, len(combinations)),
    }
    return pd.DataFrame(data)


def create_dataframe_with_duplicates() -> pd.DataFrame:
    """
    Create a dataframe with duplicate configurations.

    Structure:
    - Constitutive properties: ['param_x', 'param_y']
    - param_x: 3 unique values [1, 2, 3]
    - param_y: 2 unique values ['a', 'b']
    - Total unique configurations: 3 x 2 = 6
    - Total rows: 10 (includes 4 duplicates)
    - Additional column: 'result' (different values for duplicates)

    Use case:
    - Testing duplicate removal functionality
    - Verifying that only first occurrence is kept
    - Testing warning messages for duplicates

    Returns:
        pd.DataFrame with 10 rows (6 unique + 4 duplicate configurations)
    """
    data = {
        "param_x": [1, 1, 2, 2, 3, 3, 1, 2, 1, 3],
        "param_y": ["a", "b", "a", "b", "a", "b", "a", "b", "b", "a"],
        "result": [100, 200, 150, 250, 180, 280, 105, 155, 205, 185],
    }
    return pd.DataFrame(data)


def create_sparse_dataframe() -> pd.DataFrame:
    """
    Create a dataframe with missing configurations (sparse coverage).

    Structure:
    - Constitutive properties: ['dim1', 'dim2']
    - dim1: should have 4 values [0, 1, 2, 3]
    - dim2: should have 3 values [0, 1, 2]
    - Expected total: 4 x 3 = 12
    - Actual rows: 8 (missing 4 configurations)
    - Additional column: 'value'

    Use case:
    - Testing handling of incomplete configuration spaces
    - Verifying injection of missing rows with NaN
    - Testing order_df_for_get_index_list_nn_high_dimensional

    Returns:
        pd.DataFrame with 8 rows (incomplete coverage)
    """
    data = {
        "dim1": [0, 0, 1, 1, 2, 2, 3, 3],
        "dim2": [0, 1, 0, 2, 1, 2, 0, 1],
        "value": [10, 20, 30, 40, 50, 60, 70, 80],
    }
    return pd.DataFrame(data)


def create_large_4d_dataframe() -> pd.DataFrame:
    """
    Create a larger 4D dataframe for stress testing.

    Structure:
    - Constitutive properties: ['p1', 'p2', 'p3', 'p4']
    - p1: 5 unique values [0, 1, 2, 3, 4]
    - p2: 4 unique values [0, 1, 2, 3]
    - p3: 3 unique values [0, 1, 2]
    - p4: 2 unique values [0, 1]
    - Total configurations: 5 x 4 x 3 x 2 = 120
    - Additional columns: 'output1', 'output2'

    Use case:
    - Testing performance on larger datasets
    - Verifying sampling strategies scale correctly
    - Testing memory efficiency

    Returns:
        pd.DataFrame with 120 rows representing all combinations
    """
    import itertools

    p1_vals = list(range(5))
    p2_vals = list(range(4))
    p3_vals = list(range(3))
    p4_vals = list(range(2))

    combinations = list(itertools.product(p1_vals, p2_vals, p3_vals, p4_vals))

    data = {
        "p1": [c[0] for c in combinations],
        "p2": [c[1] for c in combinations],
        "p3": [c[2] for c in combinations],
        "p4": [c[3] for c in combinations],
        "output1": np.random.RandomState(100).randn(len(combinations)),
        "output2": np.random.RandomState(101).randn(len(combinations)),
    }
    return pd.DataFrame(data)


def create_single_dimension_dataframe() -> pd.DataFrame:
    """
    Create a dataframe with only one constitutive property.

    Structure:
    - Constitutive properties: ['param']
    - param: 10 unique values [0, 1, 2, ..., 9]
    - Total configurations: 10
    - Additional column: 'score'

    Use case:
    - Testing edge case of 1D sampling
    - Verifying behavior reduces to 1D sampling correctly
    - Testing boundary conditions

    Returns:
        pd.DataFrame with 10 rows
    """
    data = {
        "param": list(range(10)),
        "score": np.random.RandomState(50).uniform(0, 100, 10),
    }
    return pd.DataFrame(data)


def create_mixed_type_dataframe() -> pd.DataFrame:
    """
    Create a dataframe with mixed data types.

    Structure:
    - Constitutive properties: ['int_param', 'float_param', 'str_param']
    - int_param: 3 unique integers [1, 2, 3]
    - float_param: 2 unique floats [0.5, 1.5]
    - str_param: 2 unique strings ['low', 'high']
    - Total configurations: 3 x 2 x 2 = 12
    - Additional columns: 'result_a', 'result_b'

    Use case:
    - Testing handling of mixed data types
    - Verifying sorting works with different types
    - Testing type preservation through sampling

    Returns:
        pd.DataFrame with 12 rows
    """
    import itertools

    int_params = [1, 2, 3]
    float_params = [0.5, 1.5]
    str_params = ["low", "high"]

    combinations = list(itertools.product(int_params, float_params, str_params))

    data = {
        "int_param": [c[0] for c in combinations],
        "float_param": [c[1] for c in combinations],
        "str_param": [c[2] for c in combinations],
        "result_a": np.random.RandomState(60).randn(len(combinations)),
        "result_b": np.random.RandomState(61).uniform(0, 1, len(combinations)),
    }
    return pd.DataFrame(data)


def create_unbalanced_dimensions_dataframe() -> pd.DataFrame:
    """
    Create a dataframe with highly unbalanced dimension sizes.

    Structure:
    - Constitutive properties: ['small_dim', 'large_dim']
    - small_dim: 2 unique values [0, 1]
    - large_dim: 20 unique values [0, 1, 2, ..., 19]
    - Total configurations: 2 x 20 = 40
    - Additional column: 'measurement'

    Use case:
    - Testing sampling strategies with unbalanced dimensions
    - Verifying Latin Hypercube properties hold
    - Testing edge cases in dimension handling

    Returns:
        pd.DataFrame with 40 rows
    """
    import itertools

    small_vals = [0, 1]
    large_vals = list(range(20))

    combinations = list(itertools.product(small_vals, large_vals))

    data = {
        "small_dim": [c[0] for c in combinations],
        "large_dim": [c[1] for c in combinations],
        "measurement": np.random.RandomState(70).exponential(2.0, len(combinations)),
    }
    return pd.DataFrame(data)


# Summary of all test dataframes
TEST_DATAFRAMES = {
    "simple_2d": {
        "creator": create_simple_2d_dataframe,
        "constitutive_props": ["param_a", "param_b"],
        "dimensions": [3, 2],
        "total_configs": 6,
        "description": "Basic 2D test case",
    },
    "3d_ml_hyperparams": {
        "creator": create_3d_dataframe,
        "constitutive_props": ["batch_size", "learning_rate", "optimizer"],
        "dimensions": [4, 3, 2],
        "total_configs": 24,
        "description": "Realistic ML hyperparameter space",
    },
    "with_duplicates": {
        "creator": create_dataframe_with_duplicates,
        "constitutive_props": ["param_x", "param_y"],
        "dimensions": [3, 2],
        "total_configs": 6,
        "description": "Contains duplicate configurations",
    },
    "sparse_coverage": {
        "creator": create_sparse_dataframe,
        "constitutive_props": ["dim1", "dim2"],
        "dimensions": [4, 3],
        "total_configs": 12,
        "description": "Missing some configurations",
    },
    "large_4d": {
        "creator": create_large_4d_dataframe,
        "constitutive_props": ["p1", "p2", "p3", "p4"],
        "dimensions": [5, 4, 3, 2],
        "total_configs": 120,
        "description": "Large 4D space for stress testing",
    },
    "single_dimension": {
        "creator": create_single_dimension_dataframe,
        "constitutive_props": ["param"],
        "dimensions": [10],
        "total_configs": 10,
        "description": "Edge case: 1D sampling",
    },
    "mixed_types": {
        "creator": create_mixed_type_dataframe,
        "constitutive_props": ["int_param", "float_param", "str_param"],
        "dimensions": [3, 2, 2],
        "total_configs": 12,
        "description": "Mixed data types (int, float, string)",
    },
    "unbalanced_dimensions": {
        "creator": create_unbalanced_dimensions_dataframe,
        "constitutive_props": ["small_dim", "large_dim"],
        "dimensions": [2, 20],
        "total_configs": 40,
        "description": "Highly unbalanced dimension sizes",
    },
}
# %%

if __name__ == "__main__":
    """Print summary of all test dataframes."""
    print("=" * 80)
    print("TEST DATAFRAME DOCUMENTATION")
    print("=" * 80)

    for name, info in TEST_DATAFRAMES.items():
        print(f"\n{name.upper()}")
        print("-" * 40)
        print(f"Description: {info['description']}")
        print(f"Constitutive Properties: {info['constitutive_props']}")
        print(f"Dimensions: {info['dimensions']}")
        print(f"Total Configurations: {info['total_configs']}")

        # Create and show sample
        df = info["creator"]()
        print(f"Actual DataFrame rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 3 rows:")
        print(df.head(3))
        print()
