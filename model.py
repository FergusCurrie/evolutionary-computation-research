"""
Class to define an algorithm. 

An algorithm is made of three steps : member generation, member selection, decision fusion. 


"""
from code.learners.learner import Learner
import numpy as np

class Model: 
    def __init__(self, member_generation_func, member_selection_func, decision_fusion_func, params, pred_func):
        self.member_generation_func = member_generation_func
        self.member_selection_func = member_selection_func
        self.decsion_fusion_func = decision_fusion_func
        self.params = params
        self.active_param = None # ?
        self.ensemble = None #self.run(X, y)
        self.pred_func = pred_func
        self.history = None

    def member_generation(self, X, y):
        # Generate ensemble 
        self.ensemble, self.history = self.member_generation_func(X, y, self.active_param) # an ensemble should be a list of functions 


        # Convert ensemble into learner obkjects 
        self.ensemble = [Learner(member, self.pred_func) for member in self.ensemble]

    def member_selection(self, X : np.array, y : np.array):
        # Select members from ensemble
        if self.member_selection_func == None:
            return

        self.ensemble = self.member_selection_func(self.ensemble, X, y, self.decsion_fusion_func)

    def ensemble_evaluation(self, X : np.array, y: np.array, metrics : list) -> list:

        # First calculate raw predicitons
        raw_ypred = np.array([learner.predict(X) for learner in self.ensemble])

        # Then calculate true predictions with decision function
        ypred = self.decsion_fusion_func(raw_ypred)

        # Run metrics 
        results = [metric(y, ypred) for metric in metrics]

        return results[0] # change this latter 





    
