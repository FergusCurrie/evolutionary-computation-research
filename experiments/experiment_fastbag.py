"""
Experiment testing baselines. 
"""
import sys

from code.learners.EC.FastGPBag_RF import gp_rf_bagging_member_generation
sys.path.append("/")

from model import Model
from code.data_processing import get_all_datasets, get_data

from code.decision_fusion.voting import  majority_voting
from code.learners.EC.deap_extra import GP_predict
from code.metrics.classification_metrics import multi_class_metric

def get_fast_baggp_experiment():
    # name
    exp_name = "fastbag"
    # Datasets
    all_datasets = get_all_datasets()
    datasets = {}

    for d in all_datasets:
        datasets[d] = get_data(d)

    # Metrics
    metrics = [multi_class_metric]

    # Fast bag gp with lin alg norm
    fast_bag_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 20, "verbose": False, "t_size": 7, 'ncycles':5, 'batch_size':'N', 'use_hamming':False}
    #fast_bag_params_1 = {"p_size": 5, "max_depth": 5, "pc": 0.6, "pm": 0.4, "ngen": 2, "verbose": False, "t_size": 7, 'ncycles':5, 'batch_size':100, 'use_hamming':False}
    fast_bag_params= [fast_bag_params_1]
    fast_bag_model1 = Model(
        member_generation_func=gp_rf_bagging_member_generation,
        member_selection_func=None, # offEEl
        decision_fusion_func=majority_voting,
        params=fast_bag_params,
        pred_func=GP_predict,
        model_name = 'fastbaggp'
    )

    # Fast bag gp with hamming distance
    fast_bag_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 20, "verbose": False, "t_size": 7, 'ncycles':5, 'batch_size':'N', 'use_hamming':True}
    #fast_bag_params_1 = {"p_size": 5, "max_depth": 5, "pc": 0.6, "pm": 0.4, "ngen": 2, "verbose": False, "t_size": 7, 'ncycles':5, 'batch_size':100, 'use_hamming':True}
    fast_bag_params= [fast_bag_params_1]
    fast_bag_model2 = Model(
        member_generation_func=gp_rf_bagging_member_generation,
        member_selection_func=None, # offEEl
        decision_fusion_func=majority_voting,
        params=fast_bag_params,
        pred_func=GP_predict,
        model_name = 'ham_fastbaggp'
    )

    models = [fast_bag_model1, fast_bag_model2]
    ########################################################################################################################

    # Calculat number of tasks
    n_tasks = 0
    for model in models:
        n_tasks += len(datasets) * len(model.params)

    return {"datasets": datasets, "metrics": metrics, "models": models, "n_tasks": n_tasks, "name": exp_name}


