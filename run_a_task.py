"""
Run a task. This means 1 model with 1 set of parameters on 1 dataset. 
Args should be passed. jobid and taskid 
"""

from sklearn.model_selection import train_test_split
from experiment_MOGP import get_experiment
import pandas as pd
import sys
import time


def select_task(experiment, taskid):
    i = 0
    for model in experiment["models"]:
        for param in model.params:
            for dataset_name in experiment["datasets"].keys():
                if i == int(taskid):
                    return model, dataset_name, param
                i += 1
    return None, None, None


# Unpack arguments

nseeds = 30

def run(jobid, taskid): 
    """

    Careful. Grid can't handle a task id of 1. Therefore we refer to a task from 1, but is index from 0. 

    Args:
        jobid ([type]): [description]
        taskid ([type]): [description]
    """

    results = []

    # Load Experiment
    experiment = get_experiment()
    name = experiment["name"]
    print(f'Running {name}')

    # Select correct task
    # Careful. Grid can't handle a task id of 1. Therefore we refer to a task from 1, but is index from 0. 
    model, dataset_name, param = select_task(experiment, taskid-1)
    dataset = experiment["datasets"][dataset_name]

    # Select the active parameter
    model.active_param = param

    # Split the data
    X = dataset[0]
    y = dataset[1]

    # Run for 30 seeds
    for i in range(2):
        start = time.time()
        seed = 169 * i
        print(f'Run number {i}/{30}  ... seed = {seed} of {dataset_name}')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

        # Train it... member_generation -> member_selection
        model.member_generation(X_train, y_train)
        model.member_selection(X_train, y_train)

        # Evaluation
        metrics = experiment["metrics"]
        training_results = model.ensemble_evaluation(X_train, y_train, metrics)
        test_results = model.ensemble_evaluation(X_test, y_test, metrics) # comes back as [[training, seed , tpr, ..]]

        end = time.time()

        results.append([True] + [seed] + [end - start] + training_results)
        results.append([False] + [seed] + [end - start] + test_results)

        # Save history
        model.history.to_csv(f'task_store/history_{i}_{name}_job_{jobid}_task_{taskid}_{dataset_name}.csv')

    # Saving result - and History
    df = pd.DataFrame(data=results, columns = ['training', 'seed', 'time', 'full_acc', 'majority_acc', 'minority_acc', 'tn', 'fp', 'fn', 'tp'])
    df.to_csv(f"task_store/{name}_job_{jobid}_task_{taskid}_{dataset_name}.csv", index=False)


