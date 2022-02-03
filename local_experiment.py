

from experiment_MOGP import get_experiment
from run_a_task import run
import subprocess

experiment = get_experiment()

for i in range(experiment['n_tasks']):
    run(i, 1)