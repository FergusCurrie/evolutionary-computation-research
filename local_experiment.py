

from re import L
from code.data_processing import get_all_datasets, get_data
from code.metrics.classification_metrics import accuracy
from run_a_task import run

from experiments.get_experiment import get_experiment

from code.learners.EC.ORMOGP import test
from code.learners.EC.DivBaggingGP import divbagging_member_generation
from code.learners.EC.DivNicheGP import divnichegp_member_generation
from code.learners.EC.DivNicheBoostGP import divnicheboostgp_member_generation
from code.learners.EC.CCGP import ccgp_member_generation

from code.learners.EC.NCLMOGP import nclmo_member_generation
from code.learners.EC.PFMOGP import pfmo_member_generation
import random 


# MODELS ###############################################################################################################

#xpr_name = 'fast_bagboost_experiment'
xpr_name = 'test_fastbag'
X = get_experiment(xpr_name)
#X = get_experiment('fast_bagboost_experiment')

s = random.randint(0,9999)
for i in range(X['n_tasks']):
    run(s, i+1, xpr_name)



