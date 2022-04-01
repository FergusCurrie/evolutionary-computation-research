"""
Experiment testing baselines. 
"""
import sys
sys.path.append("/")

from model import Model
from code.data_processing import get_data
from code.learners.EC.MOGP import gp_mo_member_generation
from code.learners.EC.NCLMOGP import nclmo_member_generation
from code.learners.EC.PFMOGP import pfmo_member_generation
from code.learners.EC.GP import gp_member_generation
from code.member_selection.offEEL import offEEL
from code.decision_fusion.voting import binary_voting
from code.learners.EC.deap_extra import GP_predict
from code.metrics.classification_metrics import binary_metric

import numpy as np

def get_experiment__mogpdiv_experiment():
    # name
    exp_name = "divmogp_experiment"

    # Datasets
    datasets = {'ionosphere':get_data("ionosphere"), 
                'mammo_graphic' : get_data("mammo_graphic"), 
                'cleveland' : get_data('cleveland'), 
                'wisconsin' : get_data('wisconsin_breast_cancer')}


    # Metrics
    metrics = [binary_metric]

    # MODELS ###############################################################################################################
    # GP
    GP_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 50, "verbose": False, "t_size": 7}

    GP_params = [GP_params_1]
    GP_model = Model(
        member_generation_func=gp_member_generation,
        member_selection_func=None, # offEEl
        decision_fusion_func=binary_voting,
        params=GP_params,
        pred_func=GP_predict,
    )
    
    # MOGP
    MOGP_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 50, "verbose": False}
    MOGP_params = [MOGP_params_1]
    MOGP_model = Model(
        member_generation_func=gp_mo_member_generation,
        member_selection_func=offEEL, # offEEL
        decision_fusion_func=binary_voting,
        params=MOGP_params,
        pred_func=GP_predict,
    )

    # NCLMOGP
    NCLMOGP_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 50, "verbose": False}
    NCLMOGP_params = [NCLMOGP_params_1]
    NCLMOGP_model = Model(
        member_generation_func=nclmo_member_generation,
        member_selection_func=offEEL, # offEEL
        decision_fusion_func=binary_voting,
        params=NCLMOGP_params,
        pred_func=GP_predict,
    )

    # PFMOGP
    #PFMOGP_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 50, "verbose": False}
    PFMOGP_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 50, "verbose": False}
    PFMOGP_params = [PFMOGP_params_1]
    PFMOGP_model = Model(
        member_generation_func=pfmo_member_generation,
        member_selection_func=offEEL, # offEEL
        decision_fusion_func=binary_voting,
        params=PFMOGP_params,
        pred_func=GP_predict,
    )


    # Combine models into list
    models = [GP_model, MOGP_model, NCLMOGP_model, PFMOGP_model]
    ########################################################################################################################

    # Calculat number of tasks
    n_tasks = 0
    for model in models:
        n_tasks += len(datasets) * len(model.params)

    return {"datasets": datasets, "metrics": metrics, "models": models, "n_tasks": n_tasks, "name": exp_name}
    #return {"datasets": [datasets[0]], "metrics": [metrics[0]], "models": [models[0]], "n_tasks": 1, "name": [exp_name[0]]}

