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

        # Cache all seeds 
        self.y_seed = {}
        self.X_seed = {}
        self.oracle_seed = {}

        for seed in range(30):
            self.y_seed[seed * 169] = self.get_y(seed * 169)
            self.X_seed[seed * 169] = self.get_x(seed * 169)
        for seed in range(30):
            self.oracle_seed[seed * 169] = self.get_oracle(seed * 169)

        # load table
        self.load_table()
    ####################################################################################################################



    def get_oracle(self, seed):
        y = self.y_seed[seed]
        #print(self.scoring)
        assert(self.scoring[seed].shape[1] == y.shape[0])
        return np.array([np.equal(x, y) for x in self.scoring[seed]]) # nmembver x nobs (right / wrong)

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

    def extract_data(self):
        idd = self.get_jobname_to_id(self.job)
        with open(f'{self.pth}/{self.job}/{self.task}/{idd}_{self.task}_info.txt') as f:
            data = f.read()
        return data
    
    def get_y(self, seed):
        """Get a y vector using a seed and training variable 

        Args:
            seed (_type_): seed in index format 0...29?
            training (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        dns = get_all_datasets()
        data = self.extract_data()
        for dn in dns:
            if dn in data:
                X, y = get_data(dn)
                _, _, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
                if self.training:
                    return y_train
                else:
                    return y_test

    def get_x(self, seed):
        """Get a x vector using a seed and training variable 

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
        for dn in dns:
            if dn in data:
                X, y = get_data(dn)
                x_train, x_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
                if self.training:
                    return x_train
                else:
                    return x_test

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
        
        y = self.y_seed[seed]

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
    
    
    
    def coefficients(self, oracle, i, j):
        # Return abcd ratios 
        assert(oracle.shape[0] < oracle.shape[1])
        A = np.asarray(oracle[i], dtype=bool)
        B = np.asarray(oracle[j], dtype=bool)
        a = np.sum(A * B)           # A right, B right
        b = np.sum(~A * B)          # A wrong, B right
        c = np.sum(A * ~B)          # A right, B wrong
        d = np.sum(~A * ~B)         # A wrong, B wrong
        return a, b, c, d

    ####################################################################################################################
    ##### Non Pairwise
    ####################################################################################################################

    def kuncheva_entropy_measure(self, seed):
        oracle = self.oracle_seed[seed]
        L = oracle.shape[1]
        tmp = np.sum(oracle, axis=1)
        tmp = np.minimum(tmp, L - tmp)
        e = np.mean((1.0 / (L - np.ceil(0.5 * L))) * tmp)
        return e
    
    def new_entropy(self, seed):
        oracle = self.oracle_seed[seed]
        P = np.sum(oracle, axis=2) / oracle.shape[1]
        P = - P * np.log(P + 10e-8)
        entropy = np.mean(np.sum(P, axis=1))
        return entropy

    def kuncheva_kw(self, seed):
        oracle = self.oracle_seed[seed]
        L = oracle.shape[1]
        tmp = np.sum(oracle, axis=1)
        tmp = np.multiply(tmp, L - tmp)
        kw = np.mean((1.0 / (L**2)) * tmp)
        return kw

    def kohavi_wolpert_variance(self, seed):
        X = self.X_seed[seed]
        y = self.y_seed[seed]
        factor = 0
        for j in range(y.shape[0]): # j is instance index
            right, wrong = 0, 0
            for member in self.scoring[0]:
                c = member[j]
                if c == y[j]:
                    right = right + 1
                else:
                    wrong = wrong + 1
    
            factor = factor + right * wrong
    
        kw = (1.0 / (len(X) * (self.scoring[seed].shape[0]**2))) * factor
        return kw

    def entropy_measure_e(self, seed):
        X = self.X_seed[seed]
        y = self.y_seed[seed]

        factor = 0
        for j in range(y.shape[0]):
            right, wrong = 0, 0
            for member in self.scoring[seed]:
                c = member[j]
                if c == y[j]:
                    right = right + 1
                else:
                    wrong = wrong + 1
    
            factor = factor + min(right, wrong)
    
        e = (1.0 / len(X)) * (1.0 / (self.scoring[seed].shape[0] - self.scoring[seed].shape[0] / 2)) * factor
 
        return e


    ####################################################################################################################
    ##### Pairwise
    ####################################################################################################################
    def kuncheva_q_statistics(self, seed): # max diversity at 0? orcale i nmember, nobs if correct.
        oracle = self.oracle_seed[seed]
        L = oracle.shape[0]
        div = np.zeros(int((L * (L - 1)) / 2))
        div_i = 0
    
        for i in range(L):
            for j in range(i+1, L):
                a, b, c, d = self.coefficients(oracle, i, j)
                div[div_i] = float(a * d - b * c) / ((a * d + b * c) + 10e-24)
                div_i += 1
    
        return np.mean(div)
    
    def kuncheva_correlation_coefficient_p(self, seed):
        oracle = self.oracle_seed[seed]
        L = oracle.shape[0]
        div = np.zeros(int((L * (L - 1)) / 2))
        div_i = 0
    
        for i in range(L):
            for j in range(i+1, L):
                a, b, c, d = self.coefficients(oracle, i, j)
                div[div_i] = float((a * d - b * c)) / \
                    (np.sqrt((a + b) * (c + d) * (a + c) * (b + d)))
                div_i += 1
        return np.mean(div)
    
 
    def kuncheva_disagreement_measure(self, seed):
        oracle = self.oracle_seed[seed]
        L = oracle.shape[0]
        div = np.zeros(int((L * (L - 1)) / 2))
        div_i = 0
    
        for i in range(L):
            for j in range(i+1, L):
                a, b, c, d = self.coefficients(oracle, i, j)
                div[div_i] = float(b + c) / (a + b + c + d)
                div_i += 1
        return np.mean(div)
    
    
    
    def kuncheva_double_fault_measure(self, seed):
        oracle = self.oracle_seed[seed]
        L = oracle.shape[0]
        div = np.zeros(int((L * (L - 1)) / 2))
        div_i = 0
    
        for i in range(L):
            for j in range(i+1, L):
                a, b, c, d = self.coefficients(oracle, i, j)
                div[div_i] = float(d) / (a + b + c + d)
                div_i += 1
    
        return np.mean(div)
    
    ####################################################################################################################
    ##### Table generation
    ####################################################################################################################

    def save_table(self, name):
        with open(f'tables/scoring/{name}_{self.training}.tex','w') as file:
            file.write(self.metric_table.to_latex())


    def load_table(self):
        data = []
        for i in range(30):
            seed = i * 169
            metrics = [
                #self.kuncheva_entropy_measure(seed),
                #self.kuncheva_kw(seed),
                self.kohavi_wolpert_variance(seed),
                self.entropy_measure_e(seed),
                self.kuncheva_double_fault_measure(seed),
                self.kuncheva_disagreement_measure(seed),
                self.kuncheva_correlation_coefficient_p(seed),
                self.kuncheva_q_statistics(seed),
                #self.new_entropy(seed)
            ]
            data.append(metrics)
        self.metric_table = pd.DataFrame(data=data, columns=['entropy', 'kw', 'kohavi_wolpert_variance', 'entropy2', 'double fault',
        'disagreement', 'corr coeff', 'q statistic'])

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
        sns.heatmap(agre, cbar=False, cmap='Greens', center=0, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5},xticklabels=T,yticklabels=T)
        #plt.savefig(f'prepared_results/agreement_heatmap/heatmap_job{job}_task{task}.png')
    
    def perinstance_heatmap(self, seed, draw, save):
        dsnp = self.scoring[seed].transpose() # look at first seed - tranpose bit icky

        f, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(dsnp, cmap='YlGn')
        #plt.savefig(f'prepared_results/agreement_heatmap/heatmap_job{job}_task{task}.png')
    
    def cohen_kappa_heatmap(self, seed, draw, save):
        dsnp = self.scoring[seed].transpose() # look at first seed - tranpose bit icky
        f, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(dsnp, cmap='YlGn')