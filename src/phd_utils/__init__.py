import pandas as pd

def rename_signal_ids_to_var_ids(df: pd.DataFrame, vars_config: dict) -> pd.DataFrame:
    """
    Rename signal ids to var ids in a dataframe.
    :param df: Dataframe to rename signal ids to var ids in.
    :param vars_config: Dictionary with variables configuration, should contain var_id
    and signal_id for each variable.
    :return: None
    """
    var_ids, signal_ids = zip(*[(var_info['var_id'], var_info['signal_id']) for var_info in vars_config.values()])

    return df.rename(columns=dict(zip(signal_ids, var_ids)))


