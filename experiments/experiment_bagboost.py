"""
Experiment testing baselines. 
"""
import sys
from code.learners.EC.CCGP import ccgp_member_generation
from code.learners.EC.GP import gp_member_generation
from code.member_selection.greedyEnsemble import greedyEnsemble
sys.path.append("/")

from model import Model
from code.data_processing import get_all_datasets, get_data
from code.learners.EC.DivBaggingGP import divbagging_member_generation
from code.learners.EC.DivNicheGP import divnichegp_member_generation

from code.decision_fusion.voting import binary_voting
from code.learners.EC.deap_extra import GP_predict
from code.metrics.classification_metrics import binary_metric, multi_class_metric
from code.member_selection.offEEL import offEEL

def get_experiment_bagboost_experiment():
    # nam
    exp_name = "bagboost_experiment"
    # print(datasets.keys())
    all_datasets = get_all_datasets()
    datasets = {}
    for d in all_datasets:
        datasets[d] = get_data(d)

    # Metrics
    metrics = [multi_class_metric]

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
        model_name='GP'
    )
    
    # BaggingGP
    bag_params_1 = {"p_size": 500, "max_depth": 5, "pc": 0.6, "pm": 0.4, "ngen": 20, "verbose": False, "t_size": 7, 'ncycles':5, 'batch_size':100}
    bag_params = [bag_params_1]
    bag_model = Model(
        member_generation_func=divbagging_member_generation,
        member_selection_func=None, # offEEl
        decision_fusion_func=binary_voting,
        params=bag_params,
        pred_func=GP_predict,
        model_name = 'baggp'
    )


    # NichingGP
    nich_params_1 = {
        "p_size": 500,  # 500
        "max_depth": 5, 
        "pc": 0.6, 
        "pm": 0.4, 
        "ngen": 100,  # 100
        "verbose": False, 
        "t_size": 7, 
        'batch_size':100,# bs?
        'radius': 0.5, # radius of the niche
        'capacity': 1 # number of winners in a niche 
    }
    nich_params = [nich_params_1]
    nich_model = Model(
        member_generation_func=divnichegp_member_generation,
        member_selection_func=greedyEnsemble, # offEEl
        decision_fusion_func=binary_voting,
        params=nich_params,
        pred_func=GP_predict,
        model_name = 'nichgp'
    )


    # CCGP
    ccgp_params_1 = {
        "max_depth": 5, 
        "pc": 0.6, 
        "pm": 0.4, 
        "ngen": 100,  # 10
        "verbose": False, 
        "t_size": 7,
        "nspecies": 5,
        'species_size': 100,
    }
    ccgp_model = Model(
        member_generation_func=ccgp_member_generation,
        member_selection_func=None, # offEEl
        decision_fusion_func=binary_voting,
        params=[ccgp_params_1],
        pred_func=GP_predict,
        model_name = 'ccgp'
    )


    # Combine models into list
    models = [GP_model, bag_model, nich_model, ccgp_model]
    #models = [ccgp_model]

    ########################################################################################################################

    # Calculat number of tasks
    n_tasks = 0
    for model in models:
        n_tasks += len(datasets) * len(model.params)

    return {"datasets": datasets, "metrics": metrics, "models": models, "n_tasks": n_tasks, "name": exp_name}


