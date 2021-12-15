import pandas as pd
import numpy as np



def _get_cleveland() -> pd.DataFrame:
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

def _get_credit_default() -> pd.DataFrame:
    return 0

def _df_clean_qm(df : pd.DataFrame) -> np.array:
    cleaned = []
    for row in df.to_numpy():
        if '?' in list(row):
            continue
        else:
            cleaned.append(row)
    return np.array(cleaned)

def _standard_load(fn) -> np.array:
    df = pd.read_csv(fn, header=None)
    X = _df_clean_qm(df)
    return X

def _ecoli_load(fn : str) -> pd.DataFrame:
    df = pd.read_csv(fn, header=None, delim_whitespace=True)
    df = _df_clean_qm(df)
    return df 


def _to_numpy_x_y(df: np.array) -> tuple([np.array, np.array]):
    """
    Get pandas numpy and return as xy

    Args:
        df (pd.DataFrame): data

    Returns:
        tuple[np.array, np.array]: x array and y array
    """
    datanp = df.astype(float)
    dataX = datanp[:, :-1]
    dataY = datanp[:, -1]
    return dataX, dataY


def _load_file(name):
    """Basically a switch for getting the right dataset

    Args:
        name (str) : name of dataset 

    Returns:
        X,y : np.array
    """
    # about wehrre firwt file iw run from in terms f relative path 
    if name == 'cleveland':
        return _standard_load('../data/datasets/cleveland/processed.cleveland.data')
    if name == 'credit_default':
        return _standard_load('../data/datasets/credit_default/dcc.csv')
    if name == 'ecoli':
        return _ecoli_load('../data/datasets/ecoli/ecoli.data')
    if name == 'iris':
        return _standard_load('../data/datasets/iris/iris.data')
    if name == 'mammo_graphic':
        return _standard_load('data/datasets/mammo_graphic/mammographic_masses.data')
    if name == 'wiscoinsin_breast_cancer':
        return _standard_load('../data/datasets/wiscoinsin_breast_cancer/breast-cancer-wiscoinsin.data')
    print('INCORRECT NAME')


def get_data(name) -> np.array:
    df = _load_file(name)
    X, y = _to_numpy_x_y(df)
    return X, y
        