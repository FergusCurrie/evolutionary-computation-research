"""
Experiment testing baselines. 
"""
import sys
from code.learners.EC.ORMOGP import gp_ormo_member_generation
from code.learners.EC.bagmogp import bag_mogp_member_generation
sys.path.append("/")

from model import Model
from code.data_processing import get_data
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
    datasets = {}
    for ds in ['cleveland', 'ionosphere', 'mammo_graphic', 'wisconsin_breast_cancer', 'australia', 'postop', 'spec']:
        datasets[ds] = get_data(ds)

    # Metrics
    metrics = [multi_class_metric]

    # MODELS ###############################################################################################################
    
    # MOGP
    bagMOGP_params_1 = {"p_size": 5, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 3, "verbose": False}
    bagMOGP_params = [bagMOGP_params_1]
    bagMOGP_model = Model(
        member_generation_func=bag_mogp_member_generation,
        member_selection_func=None, # offEEL
        decision_fusion_func=winner_takes_all,
        params=bagMOGP_params,
        pred_func=raw_bag_GP_predict,
        model_name='bagMOGP'
    )




    # Combine models into list
    models = [bagMOGP_model]
    ########################################################################################################################

    # Calculat number of tasks
    n_tasks = 0
    for model in models:
        n_tasks += len(datasets) * len(model.params)     

    return {"datasets": datasets, "metrics": metrics, "models": models, "n_tasks": n_tasks, "name": exp_name}
    #return {"datasets": [datasets[0]], "metrics": [metrics[0]], "models": [models[0]], "n_tasks": 1, "name": [exp_name[0]]}

