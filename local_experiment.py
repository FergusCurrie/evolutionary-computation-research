

from experiment_OrMOGP import get_experiment
from run_a_task import run


experiment = get_experiment()

for i in range(experiment['n_tasks']):
    run(1, i+1)