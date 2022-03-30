'''
Wrapper for getting an experiment dictionary from experiment name 
'''

from typing import Callable
from experiments.experiment_bagboost import get_experiment_bagboost_experiment
from experiments.experiment_MOGP import get_experiment__mogp_experiment
from experiments.experiment_ORMOGP import get_experiment__ORmogp_experiment
from experiments.experiment_GP import get_experiment__gp_experiment
from experiments.experiment_full import get_experiment__full_experiment
from experiments.experiment_MOGP_div import get_experiment__mogpdiv_experiment

def is_experiment(name : str) -> bool:
    if name in ['mogp_experiment', 'gp_experiment', 'ORmogp_experiment', "full_experiment", 'bagboost_experiment', 'divmogp_experiment']:
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
    if name == 'mogp_experiment':
        return get_experiment__mogp_experiment()
    if name == 'ORmogp_experiment':
        return get_experiment__ORmogp_experiment()
    if name == 'gp_experiment':
        return get_experiment__gp_experiment()
    if name == 'full_experiment':
        return get_experiment__full_experiment()
    if name == 'bagboost_experiment':
        return get_experiment_bagboost_experiment()
    if name == 'divmogp_experiment':
        return get_experiment__mogpdiv_experiment()
    print('Experiment name error ')
    assert(1 == 0) # fail