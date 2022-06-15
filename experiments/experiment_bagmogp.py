"""
Experiment testing baselines. 
"""
import sys
from code.learners.EC.ORMOGP import gp_ormo_member_generation
from code.learners.EC.bagmogp import bag_mogp_member_generation
sys.path.append("/")

from model import Model
from code.data_processing import get_all_datasets, get_data
from code.learners.EC.MOGP import gp_mo_member_generation
from code.learners.EC.NCLMOGP import nclmo_member_generation
from code.learners.EC.PFMOGP import pfmo_member_generation
from code.learners.EC.GP import gp_member_generation
from code.member_selection.offEEL import offEEL
from code.decision_fusion.voting import binary_voting, majority_voting, winner_takes_all
from code.learners.EC.deap_extra import GP_predict, raw_bag_GP_predict
from code.metrics.classification_metrics import binary_metric, multi_class_metric

import numpy as np

def get_experiment__bagmogp():
    # name
    exp_name = "bagmogp"

    # Datasets
    all_datasets = get_all_datasets()
    datasets = {}

    for d in all_datasets:
        datasets[d] = get_data(d)

    # Metrics
    metrics = [multi_class_metric]

    # MODELS ###############################################################################################################
    
    # MOGP
    # 250 50 
    bagMOGP_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 50, "verbose": False}
    bagMOGP_params = [bagMOGP_params_1]
    bagMOGP_model = Model(
        member_generation_func=bag_mogp_member_generation,
        member_selection_func=None, # offEEL
        decision_fusion_func=winner_takes_all,
        params=bagMOGP_params,
        pred_func=raw_bag_GP_predict,
        model_name='bagMOGP500'
    )

    bagMOGP_params_2 = {"p_size": 250, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 50, "verbose": False}
    bagMOGP_params2 = [bagMOGP_params_2]
    bagMOGP_model2 = Model(
        member_generation_func=bag_mogp_member_generation,
        member_selection_func=None, # offEEL
        decision_fusion_func=winner_takes_all,
        params=bagMOGP_params2,
        pred_func=raw_bag_GP_predict,
        model_name='bagMOGP250'
    )




    # Combine models into list
    models = [bagMOGP_model, bagMOGP_model2]
    ########################################################################################################################

    # Calculat number of tasks
    n_tasks = 0
    for model in models:
        n_tasks += len(datasets) * len(model.params)     

    return {"datasets": datasets, "metrics": metrics, "models": models, "n_tasks": n_tasks, "name": exp_name}
    #return {"datasets": [datasets[0]], "metrics": [metrics[0]], "models": [models[0]], "n_tasks": 1, "name": [exp_name[0]]}

