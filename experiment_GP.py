"""
Experiment testing baselines. 
"""
import sys
sys.path.append("/")

from code.learners.EC.MOGP import gp_mo_member_generation
from code.learners.EC.GP import gp_member_generation
from code.member_selection.offEEL import offEEL
from code.learners.randomforest.randomforests import adaboost_classifier_member_generation
from code.learners.randomforest.randomforests import random_forest_classifier_member_generation
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


def forest_predict(learner, X):
    """Learner is a DecisionTreeClassifier from sklearn. Can just pass the 2d array. Need to make sure 
    shape is consistent with decision tree. 

    Args:
        learner (DecisionTreeClassifier): Decision tree. Use .predict() method from sklearn
        X (np.array):  2d dataset

    Returns:
        np.array : (n_datapoints, )
    """
    res = learner.predict(X) 
    assert(res.shape[0] == X.shape[0])
    return res


def get_experiment():
    # name
    exp_name = "gp_experiment"

    # Datasets
    datasets = {'ionosphere':get_data("ionosphere"), 
                'mammo_graphic' : get_data("mammo_graphic"), 
                'cleveland' : get_data('cleveland'), 
                'wisconsin' : get_data('wisconsin_breast_cancer')}

    # Metrics
    metrics = [binary_metric]

    # MODELS ###############################################################################################################
    # GP
    GP_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 50, "verbose": True, "t_size": 7}

    GP_params = [GP_params_1]
    GP_model = Model(
        member_generation_func=gp_member_generation,
        member_selection_func=None, # offEEl
        decision_fusion_func=binary_voting,
        params=GP_params,
        pred_func=GP_predict,
    )

    # Combine models into list
    models = [GP_model]
    ########################################################################################################################

    # Calculat number of tasks
    n_tasks = 0
    for model in models:
        n_tasks += len(datasets) * len(model.params)

    return {"datasets": datasets, "metrics": metrics, "models": models, "n_tasks": n_tasks, "name": exp_name}
    #return {"datasets": [datasets[0]], "metrics": [metrics[0]], "models": [models[0]], "n_tasks": 1, "name": [exp_name[0]]}

