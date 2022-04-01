

from code.data_processing import get_data
from code.metrics.classification_metrics import accuracy
from run_a_task import run

from experiments.get_experiment import get_experiment

from code.learners.EC.ORMOGP import test
from code.learners.EC.DivBaggingGP import divbagging_member_generation
from code.learners.EC.DivNicheGP import divnichegp_member_generation


from code.learners.EC.NCLMOGP import nclmo_member_generation
from code.learners.EC.PFMOGP import pfmo_member_generation
import random 


if True: # normal experiment 
    #name = 'full_experiment'
    name = 'divmogp_experiment'

    experiment = get_experiment(name)   
    print(f'number of tasks : {experiment["n_tasks"]}')

    jobid= random.random()

    #for i in range(experiment['n_tasks']):
        #run(jobid, i+1, name, nseeds=1)
