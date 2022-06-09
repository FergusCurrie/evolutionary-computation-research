
from code.data_processing import get_data
from code.learners.EC.MOGP import gp_mo_member_generation
from code.member_selection.offEEL import offEEL
from code.metrics.classification_metrics import *
from deap import gp
from deap import creator, base, tools
from deap.algorithms import varAnd
import numpy as np
import operator
import random
from code.metrics.classification_metrics import *
from code.learners.EC.deap_extra import GP_predict, get_pset
import pandas as pd 
from code.decision_fusion.voting import majority_voting
from model import Model
from code.learners.learner import Learner


def transform_for_class(y, cls):
    mask = y == cls
    yt = np.array(y, copy=True)  
    yt[mask == False] = 0 
    yt[mask == True] = 1 
    return yt



def bag_mogp_member_generation(X, y, params, seed): # this is going to call the innergp a few times. 
    ensemble = []
    ensemble_strings = []
    classes = np.unique(y)
    rdf = None 
    sum_history = np.ones((params['ngen'], 2))
    for cls in classes:
        # Transform class
        yt = transform_for_class(y, cls)

        # MOGP member generation
        compiled_pop, df, str_pop = gp_mo_member_generation(X, yt, params, seed)
        rdf = df

        # Convert to learners object and perform OFFEEL
        learners = [Learner(member, GP_predict) for member in compiled_pop]
        selected_learners = offEEL(learners, X, yt, majority_voting, params)

        # Convert back to functions 
        selected_as_functions = [l.get_function() for l in selected_learners]
        ensemble.append(selected_as_functions)
        
        
        #sum_history += min_history

        #ensemble.append(compiled_best) # complied lambda
        #ensemble_strings.append(str_best) # str of member 
    


    return ensemble, pd.DataFrame(data=(sum_history/len(classes))), ['']#ensemble_strings # temporarily only saving the first 

