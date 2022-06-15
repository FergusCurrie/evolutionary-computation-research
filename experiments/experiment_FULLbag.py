"""
Experiment testing baselines. 
"""
import sys
from code.learners.EC.CCGP import ccgp_member_generation
from code.learners.EC.GP import gp_member_generation
from code.learners.EC.m3gpbag import m3gpbag_member_generation
from code.member_selection.greedyEnsemble import greedyEnsemble
sys.path.append("/")

from model import Model
from code.data_processing import get_all_datasets, get_data
from code.learners.EC.DivBaggingGP import divbagging_member_generation
from code.learners.EC.DivNicheGP import divnichegp_member_generation

from code.decision_fusion.voting import binary_voting, majority_voting
from code.learners.EC.deap_extra import GP_predict
from code.metrics.classification_metrics import binary_metric, multi_class_metric
from code.member_selection.offEEL import offEEL
import pandas as pd 
import numpy as np


def m3gp_predict(learner, X, n_classes):
    """_summary_

    Args:
        learner (_type_): Individual object from the m3gp class
        X (_type_): the dataset
        n_classes (_type_): number of unique classes - only needs for range selection
    """
    # Predict only accepts a dataframe
    dfX = pd.DataFrame(data=X) 
    pred = learner.predict(dfX) # returns a list of floats
    return np.array(pred)


def get_bag_model_1():
    bag_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 20, "verbose": False, "t_size": 7, 'ncycles':5}
    bag_params = [bag_params_1]
    bag_model = Model(
        member_generation_func=divbagging_member_generation,
        member_selection_func=None, # offEEl
        decision_fusion_func=majority_voting,
        params=bag_params,
        pred_func=GP_predict,
        model_name = 'baggp'
    )
    return bag_model

def get_niche_model():# NichingGP
    nich_params_1 = {
        "p_size": 500,  # 500
        "max_depth": 8, 
        "pc": 0.6, 
        "pm": 0.4, 
        "ngen": 100,  # 100
        "verbose": False, 
        "t_size": 7, 
        'batch_size':'N',# bs?
        'radius': 0.05, # radius of the niche
        'capacity': 1 # number of winners in a niche 
    }
    nich_params = [nich_params_1]
    nich_model = Model(
        member_generation_func=divnichegp_member_generation,
        member_selection_func=greedyEnsemble, # offEEl
        decision_fusion_func=majority_voting,
        params=nich_params,
        pred_func=GP_predict,
        model_name = 'nichgp'
    )
    return nich_model


def get_ccgp_model(): # CCGP
    ccgp_params_1 = {
        "max_depth": 8, 
        "pc": 0.6, 
        "pm": 0.4, 
        "ngen": 100,  # 10
        "verbose": False, 
        "t_size": 7,
        "nspecies": 5,
        'species_size': 'N',
    }
    ccgp_model = Model(
        member_generation_func=ccgp_member_generation,
        member_selection_func=None, # offEEl
        decision_fusion_func=majority_voting,
        params=[ccgp_params_1],
        pred_func=GP_predict,
        model_name = 'ccgp'
    )
    return ccgp_model

def get_m3gp():
    m3gpbag_params_1 = {"p_size": 500, "max_depth": 5, "pc": 0.6, "pm": 0.4, "ngen": 20, "verbose": False, "t_size": 7, 'ncycles':5, 'batch_size':100}
    m3gpbag_params = [m3gpbag_params_1]
    m3gpbag_model = Model(
        member_generation_func=m3gpbag_member_generation,
        member_selection_func=None, # offEEl
        decision_fusion_func=majority_voting,
        params=m3gpbag_params,
        pred_func=m3gp_predict,
        model_name = 'm3gp_bag'
    )
    return m3gpbag_model

def get_experiment_full_bag():
    # nam
    exp_name = "fullbag"
    # print(datasets.keys())
    all_datasets = get_all_datasets()
    datasets = {}
    for d in all_datasets:
        datasets[d] = get_data(d)

    # Metrics
    metrics = [multi_class_metric]

    models = []

    # MODELS ###############################################################################################################
    # BaggingGP
    models.append(get_bag_model_1())
    models.append(get_niche_model())
    models.append(get_ccgp_model())
    


    ########################################################################################################################

    # Calculat number of tasks
    n_tasks = 0
    for model in models:
        n_tasks += len(datasets) * len(model.params)

    return {"datasets": datasets, "metrics": metrics, "models": models, "n_tasks": n_tasks, "name": exp_name}


