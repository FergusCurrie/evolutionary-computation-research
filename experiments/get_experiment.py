'''
Wrapper for getting an experiment dictionary from experiment name 
'''

from typing import Callable
from experiments.experiment_bagboost import get_experiment_bagboost_experiment

from experiments.experiment_MOGP_div import get_experiment__mogpdiv_experiment

from experiments.experiment_test_fastbag import get_fast_baggp_experiment

from experiments.experiment_bagboost_fast import get_fast_bagboost_experiment

def is_experiment(name : str) -> bool:
    if name in ['bagboost_experiment', 'divmogp_experiment', 'fast_bagboost_experiment', 'test_fastbag']:
        return True
    return False

def get_experiment(name : str) -> dict:
    """
    Returns the dictionary containing the experiment specified by name.

    Args:
        name (str): Name of the experiment to get
    """
    assert(is_experiment(name)) # confirm name is valid

    # Return correct function 
    if name == 'bagboost_experiment':
        return get_experiment_bagboost_experiment()
    if name == 'divmogp_experiment':
        return get_experiment__mogpdiv_experiment()
    if name == 'fast_bagboost_experiment':
        return get_fast_bagboost_experiment()
    if name == 'test_fastbag':
        return get_fast_baggp_experiment()
    print('Experiment name error ')
    assert(1 == 0) # fail