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

def _ionosphere_load(fn) -> np.array:
    # Load and clean the 'b' 'g' target to 1/0
    df = pd.read_csv(fn, header=None, names=['X'+str(x) for x in range(34)])
    y = np.array([1 if x == 'g' else 0 for x in df['X33']])
    df['X33'] = y
    X = _df_clean_qm(df)
    return X

def _cleveland_load(fn : str) -> np.array:
    df = pd.read_csv(fn, header=None, names=['X'+str(x) for x in range(14)])
    df['X13'] = df['X13'].replace([1,2,3,4], 1)
    X = _df_clean_qm(df)
    return X

def _wisconsin_load(fn : str) -> np.array:
    df = pd.read_csv(fn, header=None, names=['X'+str(x) for x in range(10)])
    df['X9'] = df['X9'].replace([4], 1)
    df['X9'] = df['X9'].replace([2], 0)
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
        return _cleveland_load('data/datasets/cleveland/processed.cleveland.data')
    if name == 'ionosphere':
        return _ionosphere_load('data/datasets/ionosphere/ionosphere.data')
    if name == 'credit_default':
        return _standard_load('../data/datasets/credit_default/dcc.csv')
    if name == 'ecoli':
        return _ecoli_load('../data/datasets/ecoli/ecoli.data')
    if name == 'iris':
        return _standard_load('../data/datasets/iris/iris.data')
    if name == 'mammo_graphic':
        return _standard_load('data/datasets/mammo_graphic/mammographic_masses.data')
    if name == 'wisconsin_breast_cancer':
        return _wisconsin_load('data/datasets/wisconsin_breast_cancer/breast-cancer-wisconsin.data')
    print('INCORRECT NAME')


def get_data(name) -> np.array:
    df = _load_file(name)
    X, y = _to_numpy_x_y(df)
    return X, y
