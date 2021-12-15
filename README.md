Collection of work for my AIML research. Multi-objective ensembles. Where I can i'll use my own implmentation.


# File Structure
|---data
    |---datasets
    |---frames
    |---models
|--code
    |---experiments
    |---neuralnetwork
        |---architechtures
        |---losses
        |---training
    |---visualisation
    |---notebooks
|--venv

# Experiements and subexperiments
subexperiments use identical architechture and simply tune hyper-parameters
experiments use different architechture. 
should error catch trying to rerun experiments that have already been run

# Naming conventions 

models : model_{experiement}_{subexperiment}_{n}
experiement info : expinfo_{experiement}
subexperiment info : expinfo_{experiement}_{subexperiment}

# Visualisation
all visualisation should save to a temp folder. it's so easy to load and generate. 



# New Enviroment 

This code is going to mainly be run on the university grid. As installing onto uni systems is difficult
it makes sense to write this repo in thier python version (3.8.12) and only using their libraries. I use a conda
enviroment with python3.8.12, and have a package list 'package.txt' of all packages on uni machine. If I ever have 
the need to run on another computer I store my environment in enviroment.txt. 



https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

conda list --explicit > environment.txt

conda list --explicit

conda install <package_name>=<version> e.g. conda install numpy=1.20.3

numpy=1.20.3
pandas=1.2.4
matplotlib=3.4.2
scikit-learn=0.22.1




