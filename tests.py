"""
Automated python tests.     

"""

from code.learners.randomforest.randomforests import *
from code.data_processing import get_data
import numpy as np


def test_forests():
    X,y = get_data("mammo_graphic")
    params = {
        'n_estimators':100
    }
    rf = random_forest_classifier_member_generation(X, y, params)

    raw = np.sum(np.array([dtc.predict(X) for dtc in rf]), axis=0) / 50
    raw[raw >= 0.5] = 1
    raw[raw < 0.5] = 0

    print(raw)
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    x = clf.predict(X)
    #print(x)

    



test_forests()
