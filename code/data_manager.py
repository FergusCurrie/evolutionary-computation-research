import pandas as pd
import numpy as np


def get_data(fn: str) -> pd.DataFrame:
    data = pd.read_csv(fn)
    return data


def get_cleveland() -> pd.DataFrame:
    """
    Get cleveland dataset

    Returns:
        pd.DataFrame: dataset
    """
    index = set()
    data = pd.read_csv("data/datasets/cleveland/processed.cleveland.data")
    # Really brute forced drop for testing
    for col in data.columns:
        if "?" in list(data[col]):
            for i, e in enumerate(list(data[col])):
                if e == "?":
                    index.add(i)
    index = list(index)
    data = data.drop(index)
    return data


def pandas_to_numpy_x_y(df: pd.DataFrame) -> tuple[np.array, np.array]:
    """
    Get pandas numpy and return as xy

    Args:
        df (pd.DataFrame): data

    Returns:
        tuple[np.array, np.array]: x array and y array
    """
    datanp = df.to_numpy().astype(float)
    dataX = datanp[:, :-1]
    dataY = datanp[:, -1]
    return dataX, dataY
