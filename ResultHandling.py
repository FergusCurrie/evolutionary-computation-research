'''
Code for handling results data. 
'''
import pandas as pd 
import numpy as np
import os 
import sys 
from code.data_processing import get_all_datasets
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from scipy.stats import wilcoxon

class ResultHandling:
    def __init__(self, job,height, width, box_label_size=5) -> None:
        self.job = job
        self.model_names = ['gp','bag', 'nich', 'ccgp']
        self.datasets = get_all_datasets()
        self.box_label_size = box_label_size
        
        self.width = width
        self.height = height

        # Initalisisatoin pipeline
        self.make_taskid_to_string() # dictionary for task_id -> 'model_dataset' as a string
        self.load_data() # load data into a dictionary mapping task id -> dataframe of resluts
 
    def make_taskid_to_string(self):
        """Generates a dictionry of taskid -> str('model_dataset')
        """
        self.index_to_model = {} 
        for i,model in enumerate(self.model_names):
            for j, dataset in enumerate(self.datasets):
                self.index_to_model[(i * len(self.datasets)) + j + 1] = f'{model}_{dataset}'
    
    def load_data(self) -> None:
        self.data = {}
        for task in os.listdir(f'results_file/{self.job}/'):
            if len(os.listdir(f'results_file/{self.job}/{task}')) <= 1:
                continue 
            fn = [x for x in os.listdir(f'results_file/{self.job}/{task}/') if '.csv' in x][0] # currently these files only have one entry
            self.data[int(task)] = pd.read_csv(f'results_file/{self.job}/{task}/{fn}', index_col=False)

    def get_taskid_by_model(self, mn) -> list:
        '''
        Return a chunked list of taskids. Each chunk should all taskids of a specific model 
        i = 0, 1, 2 or 3
        '''
        if mn == 'all':
            return list(self.data.keys())
        i = self.model_names.index(mn)
        sorted_keys =sorted(self.data.keys())
        model_chunks = [sorted_keys[0:10], sorted_keys[10:20], sorted_keys[20:30], sorted_keys[30:40]]
        return model_chunks[i]

    
    def get_task_ids_of_dataset(self, dn : str) -> list:
        '''Gets the task_ids for dataset
        (dn) : str  : dataset name 
        '''
        if dn == 'all':
            return list(self.data.keys())
        i = self.datasets.index(dn)
        dataset_keys = [i, i+10, i+20, i+30]
        return dataset_keys

    # method to efficently grab correct subsections from 
    def get_data(self, task, member_generation=False, training=False, numpy=False):
        x = self.data[task]
        x = x[x['member_generation'] == member_generation] 
        if numpy:
            x = x[x['training'] == training].to_numpy() # [final]
        else:
            x = x[x['training'] == training] # [final]
        return x
    
    def get_cond_data(self, mgen, train, dataset_name= 'all', model_name='all', col='full_acc'):
        dataset_taskids = self.get_task_ids_of_dataset(dataset_name)
        model_taskids = self.get_taskid_by_model(model_name)
        X = np.array([self.get_data(z, member_generation=mgen, training=train)['full_acc'] for z in self.data.keys() if z in dataset_taskids and z in model_taskids])
        X = X.flatten()
        return X

    def box_plot(self, boxes, labels, save=True, draw=False, figname=''):
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.boxplot(boxes)
        ax.set_xlim(0.5, len(boxes) + 0.5)
        ax.set_xticklabels(labels,rotation=45, fontsize=15)
        if save:
            if figname != '':
                plt.savefig(f'prepared_results/{figname}.png')
        if draw:
            plt.show()
        else:
            plt.close()

    def wilcoxon_test(self):
        tests = []
        X = [self.get_cond_data(mgen=False, train=False, dataset_name='all', model_name=model) for model in self.model_names]
        for i in range(len(X)):
            for j in range(len(X)):
                if i != j and i > j: # No statistical comparison with itself
                    tests.append([self.model_names[i],self.model_names[j],ranksums(x=X[i], y=X[j], alternative='less')])
        for t in tests:
            print(t)