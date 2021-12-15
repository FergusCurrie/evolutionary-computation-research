from code.learners.EC.MOGP import gp_mo_member_generation
from code.learners.EC.GP import gp_member_generation
from code.data_processing import get_data
from sklearn.model_selection import train_test_split
from code.metrics.classification_metrics import *
import numpy as np




X,y = get_data('mammo_graphic')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


if 0:
    ensemble = gp_mo_member_generation(X_train,y_train, p_size=500, max_depth=8, pc=0.6, pm=0.4, ngen=50, verbose=True)

    ypred = ensemble_voting(X_train, ensemble)
    confusion_matrix = calculate_confusion_matrix(y_train, ypred)
    acc = accuracy(confusion_matrix)
    print(f'final train accuracy is = {acc}')


    ypred = ensemble_voting(X_test, ensemble)
    confusion_matrix = calculate_confusion_matrix(y_test, ypred)
    acc = accuracy(confusion_matrix)
    print(f'final test accuracy is = {acc}')

if 1:
    ensemble = gp_member_generation(X_train,y_train, p_size=500, max_depth=8, pc=0.6, pm=0.4, ngen=50, t_size=7,verbose=True)

    ypred = ensemble_voting(X_train, ensemble)
    confusion_matrix = calculate_confusion_matrix(y_train, ypred)
    acc = accuracy(confusion_matrix)
    print(f'final train accuracy is = {acc}')


    ypred = ensemble_voting(X_test, ensemble)
    confusion_matrix = calculate_confusion_matrix(y_test, ypred)
    acc = accuracy(confusion_matrix)
    print(f'final test accuracy is = {acc}')

