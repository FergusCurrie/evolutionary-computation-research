

from run_a_task import run

from experiments.get_experiment import get_experiment

name = 'ORmogp_experiment'
experiment = get_experiment(name)

for i in range(experiment['n_tasks']):
    run(1, i+1, name)