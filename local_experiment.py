

from code.data_processing import get_data
from code.metrics.classification_metrics import accuracy
from run_a_task import run

from experiments.get_experiment import get_experiment

from code.learners.EC.ORMOGP import test


ORMOGP_params_1 = {"p_size": 500, "max_depth": 8, "pc": 0.6, "pm": 0.4, "ngen": 50, "verbose": False, "obj1":accuracy}
X, y = get_data("ionosphere")



#test(X, y , ORMOGP_params_1)



if True:
    #name = 'full_experiment'
    name = 'ORmogp_experiment'

    experiment = get_experiment(name)
    print(f'number of tasks : {experiment["n_tasks"]}')
    for i in range(experiment['n_tasks']):
        run(1, i+1, name)
