import ResultHandling

class Analysis:
    ####################################################################################################################
    def __init__(self, experiments) -> None:
        """
        Stores data for the graphing of one experiment. Should combine an arbitarily large number of experiments 
        and the selected models to graph from each experiment 
        """
        self.RHs = []
        for experiment in experiments:
            experiment_name, list_exclude = experiment
            assert(type(list_exclude == list))
            assert(type(experiment_name) == str)
            
            # Load experiment 
            

    
    ####################################################################################################################