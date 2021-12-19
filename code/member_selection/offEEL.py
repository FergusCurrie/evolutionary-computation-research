"""
From an ensemble selec
From this paper : https://dl.acm.org/doi/10.1145/1276958.1277317

"""
from code.metrics.classification_metrics import *



def offEEL(population, toolbox, X, y):
    # First sort population 
    sorted_pop = sorted(population, key=lambda e : member_evaluation(individual=e, toolbox=toolbox,X=X,y=y), reverse=True) # DESCENDING 

    # Now loop through an 
    L = []
    best = [-1, -1] # ind, acc
    for i in range(len(sorted_pop)-1):
        L = sorted_pop[0:i+1]
        F = [toolbox.compile(expr=individual) for individual in L]
        ypred = []
        for x in X:
            _yp = 0
            preds = np.array([single_pred(x, f) for f in F])
            if len(preds[preds == 0]) < len(preds[preds == 1]):
                _yp = 1
            ypred.append(_yp)
        confusion_matrix = calculate_confusion_matrix(y, ypred)
        acc = accuracy(confusion_matrix)
        if best[1] < acc:
            best = [i, acc]
    return population[0:best[0]+1]