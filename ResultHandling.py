'''
Code for handling results data. 
'''
import pandas as pd 
import numpy as np
import os 
import sys
from regex import F

from sklearn import datasets 
from code.data_processing import get_all_datasets
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from scipy.stats import wilcoxon

from experiments.get_experiment import get_experiment

class ResultHandling:
    def __init__(self, job, xp_name, height=10, width=10, box_label_size=15, target_dir='results_file') -> None:
        self.job = job
        self.jobid = self.get_jobname_to_id(job)
        self.box_label_size = box_label_size
        self.width = width
        self.height = height
        self.target_dir = target_dir

        # Get experiment data 
        exp = get_experiment(xp_name)
        self.datasets = list(exp["datasets"].keys())
        self.model_names = [m.model_name for m in exp["models"]]


        # Initalisisatoin pipeline
        self.make_taskid_to_string() # dictionary for task_id -> 'model_dataset' as a string
        self.load_data() # load data into a dictionary mapping task id -> dataframe of resluts

    def get_jobname_to_id(self, name):
        dict = {
            'bagboost' : '4083559',
            'fastbag_ham' : '4083560',
            'ORMOGP' : '4082885',
            'bagmogp' : '4083783',
            'm3gp' : '4082937',
            'pca' : '4083806',
            'fastbag_ham_new' : '4084319',
            'm3gpbag' : '4082937',
            'mogp500' : '4083864'
        }

        if name in dict.keys():
            return dict[name]
        return name
 
    def make_taskid_to_string(self):
        """Generates a dictionry of taskid -> str('model_dataset')
        """
        self.index_to_model = {} 
        for i,model in enumerate(self.model_names):
            for j, dataset in enumerate(self.datasets):
                self.index_to_model[(i * len(self.datasets)) + j + 1] = f'{model}_{dataset}'
    
    def load_data(self) -> None:
        self.data = {}
        for task in os.listdir(f'{self.target_dir}/{self.job}/'):
            if len(os.listdir(f'{self.target_dir}/{self.job}/{task}')) <= 1:
                continue 
            fn = [x for x in os.listdir(f'{self.target_dir}/{self.job}/{task}/') if '.csv' in x and 'HISTORY' not in x][0] # currently these files only have one entry
            self.data[int(task)] = pd.read_csv(f'{self.target_dir}/{self.job}/{task}/{fn}', index_col=False)
            if self.data[int(task)].isnull().values.any():
                print('WARNING CONTAINS NAN')


    def get_taskid_by_model(self, mn) -> list:
        '''
        Return a chunked list of taskids. Each chunk should all taskids of a specific model 
        i = 0, 1, 2 or 3
        '''
        if mn == 'all':
            return list(self.data.keys())
        f = self.model_names.index(mn)
        #taskids = [f + (i*len(self.model_names)) for i in range(len(self.datasets))]
        starting_index = len(self.datasets) * f
        taskids = [starting_index + i + 1 for i in range(len(self.datasets))]
        return taskids

    
    def get_task_ids_of_dataset(self, dn : str) -> list:
        '''Gets the task_ids for dataset
        (dn) : str  : dataset name 
        '''
        if dn == 'all':
            return list(self.data.keys())
        f = self.datasets.index(dn)
        taskids = [f + (i*len(self.datasets)) +1 for i in range(len(self.model_names))]
        return taskids

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
        if dataset_name not in self.datasets and not (dataset_name == 'all'):

            return np.array([])
        dataset_taskids = self.get_task_ids_of_dataset(dataset_name)
        model_taskids = self.get_taskid_by_model(model_name)
        #print(dataset_taskids)
        #print(model_taskids)
        #print(self.data.keys())
        X = np.array([self.get_data(z, member_generation=mgen, training=train)['full_acc'] for z in self.data.keys() if z in dataset_taskids and z in model_taskids])
        X = X.flatten()
        return X

    '''def box_plot(self, boxes, labels, save=True, draw=False, figname=''):
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        #print(boxes)
        #print(labels)

        ax.boxplot(boxes)
        ax.set_xlim(0.5, len(boxes) + 0.5)
        ax.set_xticklabels(labels,rotation=45, fontsize=self.box_label_size)
        #plt.yticks(np.arange(0.5, 1.1, 0.1))
        plt.yticks(fontsize=self.box_label_size)
        if save:
            if figname != '':
                plt.savefig(f'prepared_results/{self.job}/{figname}.png')
        if draw:
            plt.show()
        else:  
            plt.close()'''

    def get_model_vs_dataset_dataframe(self, training) -> pd.DataFrame:
        assert(type(training)==bool)
        avgs = []
        for ds in self.datasets:
            temp = []
            for mn in self.model_names:
                X = self.get_cond_data(mgen=False, train=training, dataset_name=ds, model_name=mn)
                temp.append(float(np.mean(X)))
            avgs.append(temp)
        df = pd.DataFrame(data=avgs, columns=self.model_names)
        df.index = self.datasets
        return df

    