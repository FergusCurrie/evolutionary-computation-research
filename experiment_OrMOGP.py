"""
Experiment testing baselines. 
"""
import sys
sys.path.append("/")

from code.learners.EC.OrMOGP import gp_ormo_member_generation
from code.member_selection.offEEL import offEEL
from code.data_processing import get_data
from model import Model
from code.metrics.classification_metrics import binary_metric
from code.decision_fusion.voting import binary_voting
import numpy as np


def GP_predict(learner, X):
    """GP learner is simply a lambda. However it takes 5 arguments. 

    Args:
        learner (lambda): [description]
        X ([type]): [description]

    Returns:
        np.array  : (n_datapoints, )
    """
    result = []
    for x in X:
        z = learner(*x)
        if z >= 0:
            result.append(1)
        else:
            result.append(0)
    result = np.array(result)
    assert(result.shape[0] == X.shape[0])
    return np.array(result)


def get_experiment():
    # name
    exp_name = "ORmogp_experiment"

    # Datasets
    datasets = {'ionosphere':get_data("ionosphere"), 
                'mammo_graphic' : get_data("mammo_graphic"), 
                'cleveland' : get_data('cleveland'), 
                'wisconsin' : get_data('wisconsin_breast_cancer')}

    # Metrics
    metrics = [binary_metric]

    # MODELS ###############################################################################################################
    # ORMOGP
    ORMOGP_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 50, "verbose": False}
    ORMOGP_params = [ORMOGP_params_1]
    ORMOGP_model = Model(
        member_generation_func=gp_ormo_member_generation,
        member_selection_func=offEEL, # offEEL
        decision_fusion_func=binary_voting,
        params=ORMOGP_params,
        pred_func=GP_predict,
    )

    # Combine models into list
    models = [ORMOGP_model]
    ########################################################################################################################

    # Calculat number of tasks
    n_tasks = 0
    for model in models:
        n_tasks += len(datasets) * len(model.params)

    return {"datasets": datasets, "metrics": metrics, "models": models, "n_tasks": n_tasks, "name": exp_name}
    #return {"datasets": [datasets[0]], "metrics": [metrics[0]], "models": [models[0]], "n_tasks": 1, "name": [exp_name[0]]}

