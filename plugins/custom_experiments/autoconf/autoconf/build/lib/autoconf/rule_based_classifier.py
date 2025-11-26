import pandas as pd


def to_series(x: pd.Series | pd.DataFrame | dict) -> pd.Series:
    if isinstance(x, pd.Series):
        return x

    if isinstance(x, pd.DataFrame):
        if len(x) != 1:
            raise ValueError(f"DataFrame must have exactly 1 row, got {len(x)}")
        return x.iloc[0]

    if isinstance(x, dict):
        s = pd.Series(x)
        if s.empty:
            raise ValueError("Config from dict cannot be empty")
        return s

    raise TypeError(f"Expected Series, DataFrame, or dict, got {type(x).__name__}")


def is_row_valid(config:  pd.Series | pd.DataFrame | dict, raise_error : bool = False, err_prefix :str = 'Rule-based classifier error: ')->tuple[bool, list[str]]:
    """
    Applies to rows a rule-based classification
    """
    errors = []
    config = to_series(config)


    # Rule 1
    if config['batch_size'] % config['number_gpus'] != 0:
        errors.append(err_prefix + "batch_size must be evenly divisible by number_gpus.")

    # Rule 2 (example)
    # if config['number_gpus'] > 0 and config['number_gpus'] % config['number_nodes'] != 0:
    #     errors.append("number_gpus must be evenly divisible by number_nodes.")

    if raise_error:
        if errors:
            raise ValueError("Configuration Errors:\n" + "\n".join(errors))
    else:
        if errors:
            return False, errors
        
        else:
            return True, errors


# %%

def filter_valid_with_hard_logic(df: pd.DataFrame):
    # OPTIONAL: you can create a pydantic model that does this validation
    print(f'l before {len(df)}')
    valid_indeces = []
    for i,config in df.iterrows():
        # Add other logics here
        if is_row_valid(config, raise_error=False)[0] == True:
            valid_indeces.append(i)
    df_filtered = df.loc[valid_indeces].copy()
    print(f'l after {len(df_filtered)}')
    return df_filtered