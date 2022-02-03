import numpy as np
from typing import Callable

class Learner:
    def __init__(self, learner, pred_func : Callable):
        self.learner = learner
        self.pred_func = pred_func
    
    def predict(self, X : np.array) -> float:
        return self.pred_func(self.learner, X)
    