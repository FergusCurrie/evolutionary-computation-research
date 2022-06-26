from ResultHandling import ResultHandling
from experiments.get_experiment import get_experiment
import pandas as pd
import numpy as np
from code.data_processing import get_all_datasets
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from ScoringAnalysis import ScoringAnalysis
import os 

class TaskAnalysis:
    ####################################################################################################################
    def __init__(self, experiments, save_name, datasets=get_all_datasets()) -> None:
        """
        Stores data for the graphing of one experiment. Should combine an arbitarily large number of experiments 
        and the selected models to graph from each experiment 

        experiment_name = ''
        name_in_results_finished = ''
        jobid = ''
        save_name = ''
        models_exlude = '
        """

        # might be a good idea to have specific loading variables for each wierd sub category naming shit
        # and an experiment saving name. Any tables / figures produced should be from this class. (at task level.)
        # can use results handling for a lower level basisl... e.g. conditioning 
        self.experiments = experiments
        self.save_name = save_name

        self.box_label_size = 30
        self.width = 10
        self.height = 10

        # Check all required fields in each experiment 
        for experiment in self.experiments:
            assert('experiment_name' in experiment)
            assert('name_in_results_finished' in experiment)
            assert('jobid' in experiment)
            assert('models_exclude' in experiment)

        # Load the results handling objects
        for experiment in self.experiments:
            experiment['RH'] = ResultHandling(experiment['name_in_results_finished'], experiment['experiment_name'], height=10, width = 10, box_label_size=30, target_dir='results_finished')

        self.datasets = datasets
                
    def my_round(self, x):
        x = np.array(x, dtype=np.float)
        x[x == None] = -9999
        #print(x)
        x = np.round_(x, decimals=3)
        x[x == -9999] = None # np.nan
        return x

    ####################################################################################################################
    # Bolding sifnificant values 
    ####################################################################################################################

    def wilcoxon_test(self):
        tests = []
        X = [self.get_cond_data(mgen=False, train=False, dataset_name='all', model_name=model) for model in self.model_names]
        for i in range(len(X)):
            for j in range(len(X)):
                if i != j and i > j: # No statistical comparison with itself
                    tests.append([self.model_names[i],self.model_names[j],ranksums(x=X[i], y=X[j], alternative='less')])
        for t in tests:
            pass
            
    def generate_significance_mask(self, S, direction='less'):
        """Using tensor of rank 3 generate a n_datasets x n-Models true/false matrix. 

        Args:
            S (_type_): _description_

        Returns:
            _type_: _description_
        """
        n_datasets, n_models, n_seeds = S.shape
        mask = np.zeros((n_datasets, n_models)) 
        for dataset_index in range(n_datasets):
            # Pairwise model comparison
            for model_i in range(n_models):
                if model_i == 0: # baseline doesnt significance test itself
                    continue
                stat, pvalue = ranksums(x=S[dataset_index, 0, :], y=S[dataset_index, model_i, :], alternative=direction)
                if pvalue < 0.05:
                    mask[dataset_index, model_i] = 1

                ''' X = []
                for model_j in range(n_models):
                    rs = ranksums(x=S[dataset_index, model_i, :], y=S[dataset_index, model_j, :], alternative='less')
                    X.append(rs)
                #premask[dataset_index, model_i] = np.array(X) / len(np.array(X))
                if len(np.array(X) / len(np.array(X)) < 0.05) > 0:
                    mask[dataset_index, model_i] = 1
                '''
        return mask

    def decompose_latex_table(self, df):
        """
        From a latex table string, decompose into head, mid and tail. mid being the only important part to edit
        """
        spl = df.split('\n')
        head = spl[:4]
        mid = spl[4:len(spl)-3]
        tail = spl[len(spl)-3:]
        return head, mid, tail

    def df_to_matrix(self, df):
        """
        Take a latex table string and decomposes into a matrix so elements can be indexed
        """
        _, spl, _ = self.decompose_latex_table(df)
        ar  = []
        for line in spl:
            x = line.split()
            ar.append(x)
        return ar

    
    def matrix_to_df(self, matrix, df):
        """
        From a matrix decomposed by 'df_to_matrix' stich it back together into a string
        """
        # print(matrix)# fine here 
        head, mid, tail = self.decompose_latex_table(df)
        r = ''
        for vec in matrix:
            vec[-1] = vec[-1] + '\n'
            r += ' '.join(vec)
        # print(r) fucked her e
        return '\n'.join(head) + '\n' + r + '\n'.join(tail)

    def list_tranpose(self, z):
        r = []
        for i in range(len(z[0])): # row x col. len col 2x3
            r.append([])
        for row in z:
            for i,ele in enumerate(row):
                r[i].append(ele)
        return r

    def joiner(self, Q):
        r = []
        for i in range(0, len(Q), 2):
            r.append(Q[i] + ' ' +Q[i+1])
        return r
        

    def tranpose_latex_matrix(self, df):

        # dcompose 
        spl = df.split('\n')
        head = spl[:2]
        mid = spl[4:len(spl)-3]
        tail = spl[len(spl)-3:]
        mid.insert(0,spl[2])

        # Fix head 
        head = '\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n'
        
        matrix  = []
        for line in mid:
            x = line.split()
            matrix.append(x)

        self.matrix = matrix

        matrix = [self.joiner(x) for x in matrix]
        self.joined = matrix

        matrix = self.list_tranpose(matrix)

        # Fix tranpose 
        matrix[-1] = [m.replace('\\\\', '&') for m in matrix[-1]]
        for i,m in enumerate(matrix):
            matrix[i][-1] = m[-1].replace('&', '\\\\')

        # Repair and return 
        r = ''
        for vec in matrix:
            vec[-1] = vec[-1] + '\n'
            r += ' '.join(vec)

        final = head + '\n' + r + '\n'.join(tail)
        self.final = final
        return final

    def bold_value(self, matrix, row_index, col_index, direction):
        """
        Bold index i, j of a single element 
        """
        adjusted_col_index = (col_index * 2) + 2
        value = matrix[row_index][adjusted_col_index]
        #bolded = '   \\textbf{' + value + '}'
        if direction == 'less':
            bolded = '   \\textbf{+' + value + '}'
            #bolded = '+' + value
        if direction == 'greater':
            bolded = '   \\textbf{-' + value + '}'
            #bolded = '-' + value
        matrix[row_index][adjusted_col_index] = bolded

    def bold_latex_string(self, df, mask, direction):
        X = self.df_to_matrix(df)
        for i, row in enumerate(mask):
            for j, cell in enumerate(row):
                if cell:
                    self.bold_value(X, i, j, direction)
        return self.matrix_to_df(X, df)
    ####################################################################################################################



    def create_parameter_table(self):
        # Create parameter table
        params_stack = {}
        for experiment in self.experiments:
            exp = get_experiment(experiment['experiment_name'])
            for mi, model in enumerate(exp['models']):
                mn = model.model_name
                if mn not in experiment['models_exclude']:
                    #print(mn)
                    #print(params_stack.keys())
                    while mn in params_stack.keys():
                        mn = '2'+mn
                    save_mn = mn
                    if 'name_as' in experiment:
                        mn = experiment['name_as'][mi]
                    params_stack[mn] = model.params[0] # currently multiple params per model is not used
        df = pd.DataFrame.from_dict(params_stack)

        # Save table 
        df.to_latex().replace('NaN', '-')
        with open(f'tables/params_{self.save_name}.tex','w') as file:
            file.write(df.to_latex())

    def box_plot(self, boxes, labels, save=True, draw=False, figname=''):


        fig, ax = plt.subplots(figsize=(self.width, self.height))

        ax.boxplot(boxes)
        ax.set_xlim(0.5, len(boxes) + 0.5)
        ax.set_xticklabels(labels,rotation=0, fontsize=self.box_label_size)
        #plt.yticks(np.arange(0.5, 1.1, 0.1))
        plt.yticks(fontsize=self.box_label_size)
        if save:
            if figname != '':
                plt.savefig(f'prepared_results/{self.save_name}/{figname}.png')
        if draw:
            plt.show()
        else:  
            plt.close()
    
    def save_tables(self):
        for train in [True, False]:
            skipped_datasets = []
            latex_data = [] # at the end hsould be (ndatasets, nmodels, nseeds)
            training_string = 'train' if train else 'test'
            columns = [mn for experiment in self.experiments for mn in experiment['RH'].model_names if mn not in experiment['models_exclude']]
            columns = []
            significance = []

            for ds in self.datasets:
                # Per model train
                plot_labels, plots = [], []
                significance_ds = []

                for experiment in self.experiments:
                    if experiment['experiment_name'] == 'dr': # Scuffed pca only approacgh
                        for i in range(4):
                            j = i + 1
                            X = experiment['RH'].get_cond_data(mgen=False, train=train, model_name='pcabag', dataset_name=f'{ds}_{j}',col='full_acc')
                            plots.append(np.mean(X))
                            significance_ds.append(X)
                            columns = []
                            plot_labels.append(f'pca_{j}')
                    else:
                        for mi, model_nam in enumerate(experiment['RH'].model_names):
                            if model_nam not in experiment['models_exclude']:
                                
                                X = experiment['RH'].get_cond_data(mgen=False, train=train, model_name=model_nam, dataset_name=ds,col='full_acc') # should this be 30. already averaging?

                                if X.size == 0: # check for empty array
                                    plots.append(None)
                                    significance_ds.append(np.empty(30))
                                else:
                                    plots.append(np.mean(X))
                                    significance_ds.append(X)
                                if 'name_as' in experiment:
                                    plot_labels.append(experiment['name_as'][mi])
                                else:
                                    plot_labels.append(f'{model_nam}')
                if columns == []:
                    columns = plot_labels
                latex_data.append(plots) # latex data is (n_datasets, nmodels)
                significance.append(np.array(significance_ds))

            # Prepare table
            np_signifiance = np.array(significance) # number datasets, num models, num seeds                 

            np_latex_data = np.array(latex_data)
            np_latex_data = self.my_round(np_latex_data)
            df = pd.DataFrame(data=np_latex_data, columns=columns)
            df.index = [ds[:4] for ds in get_all_datasets() if ds not in skipped_datasets]

            latec = df.to_latex()
            mask_better_than_base = self.generate_significance_mask(np_signifiance, direction='less')
            mask_worse_than_base = self.generate_significance_mask(np_signifiance, direction='greater')
            st = self.bold_latex_string(latec, mask_better_than_base, direction='less')
            st = self.bold_latex_string(st, mask_worse_than_base, direction='greater')
            #st = self.tranpose_latex_matrix(st)
            with open(f'tables/{self.save_name}_{training_string}.tex','w') as file:
                file.write(st)
    

    def save_boxplots(self):
        for train in [True, False]:
            training_string = 'train' if train else 'test'
            
            for ds in self.datasets:
                plot_labels, plots = [], []
                for ei, experiment in enumerate(self.experiments):
                    if experiment['experiment_name'] == 'dr': # Scuffed pca only approacgh
                        for i in range(4):
                            j = i + 1
                            plots.append(experiment['RH'].get_cond_data(mgen=False, train=train, model_name='pcabag', dataset_name=f'{ds}_{j}',col='full_acc'))
                            if 'name_as' in experiment:
                                plot_labels.append(experiment['name_as'][i])
                            else:
                                plot_labels.append(f'pca_{j}')
                    else:
                        for mi, model_nam in enumerate(experiment['RH'].model_names):
                            if model_nam not in experiment['models_exclude']:
                                X = experiment['RH'].get_cond_data(mgen=False, train=train, model_name=model_nam, dataset_name=ds,col='full_acc') # should this be 30. already averaging?
                                if X.size == 0: # check for empty array
                                    #print(f'{ei} {mi}')
                                    pass
                                else:
                                    plots.append(X)
                                    if 'name_as' in experiment:
                                        plot_labels.append(experiment['name_as'][mi])
                                    else:
                                        plot_labels.append(f'{model_nam}')
                self.box_plot(boxes=plots, labels=plot_labels, save=True, draw=False, figname=f'{self.save_name}_{ds}_{training_string}')

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

    def extract_data(self, job, task):
        idd = self.get_jobname_to_id(job)
        with open(f'results_finished/{job}/{task}/{idd}_{task}_info.txt') as f:
            data = f.read()
        return data

    def save_diversity_table(self):
        columns = []
        metrics = []
        for experiment in self.experiments:
            job = experiment['name_in_results_finished']
            # fast bag is double dipping 
            #n_models = len([mn for mn in experiment['RH'].model_names]) - len(experiment['models_exclude'])
            for mi, model_name in enumerate(experiment['RH'].model_names):
                    
                if model_name not in experiment['models_exclude'] and not model_name == 'GP':
                    if 'name_as' in experiment:
                        columns.append(experiment['name_as'][mi])
                    else:
                        columns.append(model_name)
                    
                    X = np.empty((len(self.datasets), 30, 8)) # datasets x seeds x metrics 
                    for task in os.listdir(f'results_finished/{job}'):

                        data = self.extract_data(job, task)
                        if model_name == data.split()[2]:
                            sa = ScoringAnalysis(job=job, task=task, training=False)
                            sa.load_table()
                            #tasks.append(sa.metric_table)  #
                            int_task= int(task)
                            while int_task >= 10:
                                int_task -= 10
                            X[int(int_task)] = sa.metric_table


                    # Only adding good tasks 
                    metrics.append(X)


        npmetrics = np.array(metrics)    # (n_models, n_datasets, n_seeds, n_metrics)
        npmetrics = npmetrics.reshape(npmetrics.shape[3], npmetrics.shape[0], npmetrics.shape[1] * npmetrics.shape[2])
        avg = np.nanmean(npmetrics, axis=2)
        avg = self.my_round(avg)
        df = pd.DataFrame(data=avg, columns=columns)
        metric_names = ['entropy', 'kw', 'entropy_2', 'kohavi_wolpert', 'double_fault', 'disagreement', 'corr_coeff_p', 'q_stat']
        df.index = metric_names

        latec = df.to_latex()
        mask_better_than_base = self.generate_significance_mask(npmetrics, direction='less')
        mask_worse_than_base = self.generate_significance_mask(npmetrics, direction='greater')
        st = self.bold_latex_string(latec, mask_better_than_base, direction='less')
        st = self.bold_latex_string(st, mask_worse_than_base, direction='greater')
        with open(f'tables/diversity_{self.save_name}.tex','w') as file:
            file.write(st)
    
