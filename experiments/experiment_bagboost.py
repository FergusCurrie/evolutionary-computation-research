"""
Experiment testing baselines. 
"""
import sys
from code.member_selection.greedyEnsemble import greedyEnsemble
sys.path.append("/")

from model import Model
from code.data_processing import get_data
from code.learners.EC.DivBaggingGP import divbagging_member_generation
from code.learners.EC.DivNicheGP import divnichegp_member_generation

from code.decision_fusion.voting import binary_voting
from code.learners.EC.deap_extra import GP_predict
from code.metrics.classification_metrics import binary_metric
from code.member_selection.offEEL import offEEL

def get_experiment_bagboost_experiment():
    # name
    exp_name = "bagboost_experiment"

    # Datasets
    datasets = {'ionosphere':get_data("ionosphere"), 
                'mammo_graphic' : get_data("mammo_graphic"), 
                'cleveland' : get_data('cleveland'), 
                'wisconsin' : get_data('wisconsin_breast_cancer')}

    # Metrics
    metrics = [binary_metric]

    # MODELS ###############################################################################################################
    # BaggingGP
    bag_params_1 = {"p_size": 500, "max_depth": 5, "pc": 0.6, "pm": 0.4, "ngen": 20, "verbose": False, "t_size": 7, 'ncycles':5, 'batch_size':100}
    bag_params = [bag_params_1]
    bag_model = Model(
        member_generation_func=divbagging_member_generation,
        member_selection_func=None, # offEEl
        decision_fusion_func=binary_voting,
        params=bag_params,
        pred_func=GP_predict,
    )


    # NichingGP
    nich_params_1 = {
        "p_size": 50,  # 500
        "max_depth": 5, 
        "pc": 0.6, 
        "pm": 0.4, 
        "ngen": 5,  # 100
        "verbose": False, 
        "t_size": 7, 
        'batch_size':100,
        'radius': 1, # radius of the niche
        'capacity': 1 # number of winners in a niche 
    }

    nich_params = [nich_params_1,nich_params_1]
    nich_model = Model(
        member_generation_func=divnichegp_member_generation,
        member_selection_func=greedyEnsemble, # offEEl
        decision_fusion_func=binary_voting,
        params=nich_params,
        pred_func=GP_predict,
    )

    # Combine models into list
    #models = [bag_model, nich_model]
    models = [nich_model]
    ########################################################################################################################

    # Calculat number of tasks
    n_tasks = 0
    for model in models:
        n_tasks += len(datasets) * len(model.params)

    return {"datasets": datasets, "metrics": metrics, "models": models, "n_tasks": n_tasks, "name": exp_name}


