from run_a_task import run
import sys
from experiments.get_experiment import get_experiment


jobid = int(sys.argv[1])
taskid = int(sys.argv[2])
xpr_name = str(sys.argv[3])
experiment = get_experiment(xpr_name)
run(jobid, taskid, experiment['name'])
