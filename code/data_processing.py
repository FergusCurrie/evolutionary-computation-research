import pandas as pd
import numpy as np
from sklearn import preprocessing

#########################################################################################

def _get_credit_default() -> pd.DataFrame:
    return 0

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
    label_encoder = preprocessing.LabelEncoder()
    df = pd.read_csv(fn, header=None, delim_whitespace=True)
    df = df.iloc[:, 1:]
    df[8]= label_encoder.fit_transform(df[8]) 
    df = _df_clean_qm(df)
    return df 

def _iris_load(fn : str) -> pd.DataFrame:
    label_encoder = preprocessing.LabelEncoder()
    df = pd.read_csv(fn, header=None)
    df[4]= label_encoder.fit_transform(df[4]) 
    df = _df_clean_qm(df)
    return df 

def _australia_load(fn : str) -> np.array:
    df = pd.read_csv(fn, header=None, delim_whitespace=True)
    X = _df_clean_qm(df)
    return X

def _dress_load(fn : str) -> np.array:
    df = pd.read_csv(fn, header=None)
    X = _df_clean_qm(df)
    return X

def _post_operative_load(fn : str) -> np.array:
    df = pd.read_csv(fn, header=None)
    X = _df_clean_qm(df)
    return X
    
def _post_operative_load(fn : str) -> np.array:
    label_encoder = preprocessing.LabelEncoder()
    df = pd.read_csv(fn, header=None)
    for i in range(len(df.columns)):
        df[i]= label_encoder.fit_transform(df[i]) 
    X = _df_clean_qm(df)
    return X

def _titanic_load(fn : str) -> np.array:
    df = pd.read_csv(fn, header=None)
    X = _df_clean_qm(df)
    return X

def _yeast_load(fn : str) -> np.array:
    label_encoder = preprocessing.LabelEncoder()
    df = pd.read_csv(fn, header=None, delim_whitespace=True)
    df = df.iloc[:, 1:]
    df[9] = label_encoder.fit_transform(df[9]) 
    X = _df_clean_qm(df)
    return X

def _spec_load(fn : str) -> np.array:
    # Load training 
    df1 = pd.read_csv(fn+'.TRAIN', header=None)

    # Load test
    df2 = pd.read_csv(fn+'.TEST', header=None)

    # pd concat
    df = pd.concat([df1, df2], ignore_index=True) 
    X = _df_clean_qm(df) 
    return X


#########################################################################################

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

#########################################################################################
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
        return _standard_load('data/datasets/credit_default/dcc.csv')
    if name == 'ecoli':
        return _ecoli_load('data/datasets/ecoli/ecoli.data')
    if name == 'iris':
        return _iris_load('data/datasets/iris/iris.data')
    if name == 'mammo_graphic':
        return _standard_load('data/datasets/mammo_graphic/mammographic_masses.data')
    if name == 'wisconsin_breast_cancer':
        return _wisconsin_load('data/datasets/wisconsin_breast_cancer/breast-cancer-wisconsin.data')
    if name == 'australia':
        return _australia_load('data/datasets/australia/australian.dat')
    if name == 'dresses':
        return _dress_load('data/datasets/australia/australian.dat')
    if name == 'postop':
        return _post_operative_load('data/datasets/post_operative/post-operative.data')
    if name == 'titanic':
        return _titanic_load('data/datasets/titanic/gender_submission.csv')
    if name == 'yeast':
        return _yeast_load('data/datasets/yeast/yeast.data')
    if name == 'spec':
        return _spec_load('data/datasets/spec/SPECT')
    print('INCORRECT NAME')

def get_data(name) -> np.array:
    df = _load_file(name)
    X, y = _to_numpy_x_y(df)
    return X, y


def get_all_datasets():
    datasets = [
        'cleveland', 
        'ionosphere', 
        'ecoli', 
        'iris', 
        'mammo_graphic', 
        'wisconsin_breast_cancer',
        'australia',
        'postop',
        'yeast',
        'spec']
    # 'titanic' - dataset just has passenger id 
    # 'dresses' - rar file 
    # 'credit_default' - slow to load 
    #return datasets
    return ['spec']