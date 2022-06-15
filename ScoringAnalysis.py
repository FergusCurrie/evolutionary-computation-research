from code.data_processing import get_all_datasets
from code.data_processing import get_data
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 
import os

from code.decision_fusion.voting import majority_voting 
import matplotlib.pyplot as plt
import seaborn as sns

"""

This code does scoring analysis on a single task. 


"""

class ScoringAnalysis:
    ####################################################################################################################
    def __init__(self, job, task, training) -> None:
        # Load parameters 
        self.pth = 'results_finished'
        self.job = job
        self.task = task
        self.training = training

        # Methods for loading 
        self.load_datasets()
        self.load_all_scoring(f'{self.pth}/{self.job}/{self.task}/scoring/') # {seed} -> np.array(nmembers, nobs)
    
    ####################################################################################################################

    def get_jobname_to_id(self, name):
        dict = {
            'bagboost' : '4083559',
            'fastbag_ham' : '4083560',
            'ORMOGP' : '4082885'
        }

        if name in dict.keys():
            return dict[name]
        return name

    def get_y(self, seed):
        """Get a y vector using a seed and training variable 

        Args:
            seed (_type_): seed in index format 0...29?
            training (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        dns = get_all_datasets()
        idd = self.get_jobname_to_id(self.job)
        with open(f'{self.pth}/{self.job}/{self.task}/{idd}_{self.task}_info.txt') as f:
            data = f.read()
        print(data)
        for dn in dns:
            if dn in data:
                X, y = get_data(dn)
                _, _, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
                if self.training:
                    return y_train
                else:
                    return y_test

    def load_datasets(self):
        all_datasets = get_all_datasets()
        self.datasets = {}
        for d in all_datasets:
            self.datasets[d] = get_data(d)  

    def load_scoring(self, fn : str) -> np.array: 
        assert('.csv' in fn)
        x = pd.read_csv(fn).transpose().to_numpy()
        assert(len(x.shape) == 2)
        return x 

    def seed_from_x(self, x):
        return int(x.split('_seed_')[1].split('_')[0])


    def load_all_scoring(self, path : str) -> np.array: 
        self.scoring = {}
        for x in os.listdir(path):
            if (not '.csv' in x):
                continue 
            if self.training:
                if (not 'training' in x):
                    continue
            else:
                if (not 'test' in x):
                    continue
            seed = self.seed_from_x(x)
            self.scoring[seed] = self.load_scoring(path+'/'+x)
                    
        #self.scoring = np.array(F)  # (nseeds, nmembers, nobs)


    ####################################################################################################################

    def agreement(self, v1 : np.array, v2 : np.array) -> np.array:
        agreement = []
        for i in range(len(v1)):
            if v1[i] == v2[i]:
                agreement.append(1)
            else:
                agreement.append(0)
        return np.array(agreement)

    def pairwise_agreement(self, X : np.array, seed : int):
        assert(len(X.shape) == 2)
        assert(X.shape[0] > X.shape[1]) # make sure columns are on the y
        
        # Create large numpy vector with (n_members + 1 + 1, n_datapoints) where thes 1s correspond to ypred and ens_pred
        temp = np.ones((X.shape[0], X.shape[1] + 2))
        ens_pred = majority_voting(X.transpose())
        
        y = self.get_y(seed)
        print(y.shape)
        temp[:, :X.shape[1]] = X # up to third room from end is input
        temp[:, X.shape[1]] = ens_pred # second row from end is ens pred 
        temp[:, X.shape[1]+1] = y # final row from end is ens pred 
        
        # Pairwise
        full = []
        by_member = {}
        for t1 in temp.transpose():
            t1agree = []
            for t2 in temp.transpose():
                #full.append(self.agreement(t1, t2))
                t1agree.append(np.mean(t1 == t2))
            full.append(np.array(t1agree))
        
        return np.array(full)
    
    ####################################################################################################################
    # Drawing 
    ####################################################################################################################

    def agreement_heatmap(self, seed, draw, save):
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        dsnp = self.scoring[seed].transpose() # look at first seed - tranpose bit icky
        agre = self.pairwise_agreement(dsnp, seed).reshape(7,7)
        mask = np.triu(np.ones_like(agre, dtype=bool))
        T= ['m1','m2','m3','m4','m5','ensemble','ytrue']

        f, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(agre, cmap='Greens', center=0, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5},xticklabels=T,yticklabels=T)
        #plt.savefig(f'prepared_results/agreement_heatmap/heatmap_job{job}_task{task}.png')
    
    def perinstance_heatmap(self, seed, draw, save):
        dsnp = self.scoring[seed].transpose() # look at first seed - tranpose bit icky

        f, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(dsnp, cmap='YlGn')
        #plt.savefig(f'prepared_results/agreement_heatmap/heatmap_job{job}_task{task}.png')