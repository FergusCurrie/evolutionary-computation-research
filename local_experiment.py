

from run_a_task import run

from experiments.get_experiment import get_experiment

name = 'full_experiment'
experiment = get_experiment(name)
print(f'number of tasks : {experiment["n_tasks"]}')
for i in range(experiment['n_tasks']):
    run(1, i+1, name)