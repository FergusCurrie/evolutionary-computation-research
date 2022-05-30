'''
Wrapper for getting an experiment dictionary from experiment name 
'''

from typing import Callable
from experiments.experiment_bagboost import get_experiment_bagboost_experiment

from experiments.experiment_MOGP_div import get_experiment__mogpdiv_experiment

from experiments.experiment_fastbag import get_fast_baggp_experiment
from experiments.experiment_m3gpbag import get_m3gpbag_experiment


def is_experiment(name : str) -> bool:
    if name in ['bagboost_experiment', 'divmogp_experiment', 'fastbag','m3gp_bag']:
        return True
    print(f'\n\n {name} \nn\n')
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
    if name == 'fastbag':
        return get_fast_baggp_experiment()
    if name == 'm3gp_bag':
        return get_m3gpbag_experiment()
    print('Experiment name error ')
    assert(1 == 0) # fail