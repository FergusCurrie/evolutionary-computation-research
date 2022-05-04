"""
Run a task. This means 1 model with 1 set of parameters on 1 dataset. 
Args should be passed. jobid and taskid 
"""

from sklearn.model_selection import train_test_split
from experiments.get_experiment import get_experiment

import pandas as pd
import sys
import time
import os 


def select_task(jobid : int, taskid : int, experiment : dict):
    """
    Method for selecting current task out of all tasks in job.

    Args:
        taskid (int): Which task within experiment. A task is a combination of a model, a set of parameters and a dataset. 
        experiment (dict): Dictionary containing experiment data. 

    Returns:
        Model, str, dict: The model, dataset and parameters that define this task
    """
    i = 0
    for model in experiment["models"]:
        for param in model.params:
            for dataset_name in experiment["datasets"].keys():
                #print(taskid)
                if i == int(taskid)-1: # minus 1 for index
                    # now write to file 
                    f = open(f"results_file/{jobid}/{taskid}/{jobid}_{taskid}_info.txt", "a")
                    f.write(f"{taskid} : {model.model_name} , {dataset_name} , {param} \n")
                    f.close()
                    return model, dataset_name, param
                i += 1
    return None, None, None

def evaluation(X_train, y_train, X_test, y_test, model, mgen, metrics, seed, time):
    training_results = model.ensemble_evaluation(X_train, y_train, metrics)
    test_results = model.ensemble_evaluation(X_test, y_test, metrics) # comes back as [[training, seed , tpr, ..]]
    a = [mgen] + [True] + [seed] + [time] + training_results
    b = [mgen] + [False] + [seed] + [time] + test_results
    return [a, b]


def run(jobid : int, taskid : int, name : str, nseeds = 30): 
    """
    Run a single task. 

    Careful. Grid can't handle a task id of 1. Therefore we refer to a task from 1, but is index from 0. 

    Args:
        jobid (int): Which job. A job is the number corresponding to the experiment (differnet grid runs of same exp have diff job ids). 
        taskid (int): Which task within experiment. A task is a combination of a model, a set of parameters and a dataset. 
    """
    # first make file system
    if not os.path.isdir(f'results_file/{jobid}'):
        os.mkdir(f'results_file/{jobid}')
    d = f'results_file/{jobid}/{taskid}/'
    os.mkdir(d)
    os.mkdir(f'{d}/scoring')
    os.mkdir(f'{d}/lisp_model')

    

    results = []

    # Load Experiment
    print(f'Running {name}')
    experiment = get_experiment(name) # experiment is now a dict
    metrics = experiment["metrics"]

    # Select correct task
    # Careful. Grid can't handle a task id of 1. Therefore we refer to a task from 1, but is index from 0. 
    model, dataset_name, param = select_task(jobid, taskid, experiment)
    dataset = experiment["datasets"][dataset_name] # none on dataset name ? 

    # Select the active parameter
    model.active_param = param

    # Split the data
    X = dataset[0]
    y = dataset[1]

    # Run for 30 seeds
    for i in range(nseeds):
        seed = 169 * i
        print(f'Run number {i}/{nseeds}  ... seed = {seed} of {dataset_name}')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y) # NOTICE STRATIFICATION

        # Member generation 
        start = time.time()
        model.member_generation(X_train, y_train, seed)
        end = time.time()
        results = results + evaluation(X_train, y_train, X_test, y_test, model=model, metrics=metrics, mgen=True, time=start - end, seed=seed)

        # Member Selection
        start = time.time()
        model.member_selection(X_train, y_train)
        end = time.time()
        results = results + evaluation(X_train, y_train, X_test, y_test, model=model, metrics=metrics, mgen=True, time=start - end, seed=seed)

        # Compare members on scoring. 
        raw_ypreds = model.get_member_ypreds(X_train, y_train)
        df = pd.DataFrame(data=raw_ypreds.T, columns = [f'member_{x}' for x in range(model.get_number_selected())])
        df.to_csv(f'{d}/scoring/SCORING_{name}_job_{jobid}_task_{taskid}_{dataset_name}_seed_{seed}.csv', index=False)

        # Save the model str
        lisp_trees = model.get_member_strings()
        for ff, tree in enumerate(lisp_trees):
            f = open(f'{d}/lisp_model/member_{ff}_lisp.txt', "a")
            f.write(tree)
            f.close()



    # Saving result - and History
    df = pd.DataFrame(data=results, columns = ['member_generation','training', 'seed', 'time', 'full_acc', 'majority_acc', 'minority_acc', 'tn', 'fp', 'fn', 'tp'])
    df.to_csv(f"{d}{name}_job_{jobid}_task_{taskid}_{dataset_name}.csv", index=False)


