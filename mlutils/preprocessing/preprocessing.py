import numpy as np
import pandas as pd


def set_time_index(df: pd.DataFrame, col_name: str, time_format: str = None) -> pd.DataFrame:
    if col_name not in df.columns:
        raise Exception("Column {} is not in dataset: {}".format(col_name, df.columns))
    try:
        if time_format is not None:
            df.index = pd.to_datetime(df[col_name], format=time_format)
        else:
            df.index = pd.to_datetime(df[col_name])
    except Exception as e:
        print("Unable to set {} as time index: {}".format(col_name, e))
    return df


def resample_df(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    # Time interval is the number of seconds between each misuration
    time_interval = df.index.to_series().diff().dt.seconds.mode()[0]
    # Take only the integer part
    time_interval = str(int(time_interval))
    df = df.resample(time_interval + "S").mean().ffill().interpolate()
    assert (df.index.to_series().diff().dt.seconds > float(time_interval)).sum() == 0
    df = df.resample("{}".format(interval) + "S").mean().ffill().interpolate()
    assert (df.index.to_series().diff().dt.seconds > float(interval)).sum() == 0
    return df


def split_sequence(sequence: np.array, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
