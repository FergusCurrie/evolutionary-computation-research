"""
Experiment testing baselines. 
"""
import sys
sys.path.append("/")

from model import Model
from code.data_processing import get_data
from code.learners.EC.ORMOGP import gp_ormo_member_generation
from code.member_selection.offEEL import offEEL
from code.decision_fusion.voting import binary_voting
from code.learners.EC.deap_extra import GP_predict
from code.metrics.classification_metrics import accuracy, binary_metric, ave


def get_experiment__ORmogp_experiment():
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
    ORMOGP_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 50, "verbose": False, "obj1":ave}
    ORMOGP_params_2 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 50, "verbose": False, "obj1":accuracy}
    ORMOGP_params = [ORMOGP_params_1, ORMOGP_params_1]
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

