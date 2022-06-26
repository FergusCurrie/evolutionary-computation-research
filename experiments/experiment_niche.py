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

from code.decision_fusion.voting import binary_voting, majority_voting
from code.learners.EC.deap_extra import GP_predict
from code.metrics.classification_metrics import binary_metric, multi_class_metric
from code.member_selection.offEEL import offEEL

def get_experiment_niche():
    # nam
    exp_name = "niche"
    # print(datasets.keys())
    all_datasets = get_all_datasets()
    datasets = {}
    datasets['cleveland'] = get_data('cleveland')

    # Metrics
    metrics = [multi_class_metric]

    # p_size = 2

    # MODELS ###############################################################################################################

    # NichingGP
    nich_params_1 = {
        "p_size": 500,  # 500
        "max_depth": 5, 
        "pc": 0.6, 
        "pm": 0.4, 
        "ngen": 100,  # 100
        "verbose": True, 
        "t_size": 7, 
        'batch_size':100,# bs?
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



    # Combine models into list
    models = [nich_model]
    #models = [bag_model]

    ########################################################################################################################

    # Calculat number of tasks
    n_tasks = 0
    for model in models:
        n_tasks += len(datasets) * len(model.params)

    return {"datasets": datasets, "metrics": metrics, "models": models, "n_tasks": n_tasks, "name": exp_name}


