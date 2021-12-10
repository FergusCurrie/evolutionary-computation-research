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



# New Enviroment ... CONDA
This code is going to mainly be run on the university grid. This means the enviroment needs to be runnable from there,
as they are on an older version the enviroment system needs to be able to control this. This code base now uses 
Python3.8.12


https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

conda list --explicit > environment.txt



# Enviroment Stuff (OLD): 

https://towardsdatascience.com/virtual-environments-104c62d48c54#:~:text=A%20virtual%20environment%20is%20a,a%20system%2Dwide%20Python).

1. Create $ python3 -m venv venv/

2. Activate $ source venv/bin/activate 
    2. Deactivate 

3. Show packages $ pip3 list

4. Install packages $ pip3 install 

5. Freeze current package state $ pip3 freeze > requirements.txt
    5. Now we have a text file of the enviroment (which we can install from)

6. And loading enviroment with $ pip install -r requirements.txt

Any issues with enviroment, re-install it: 

$ rm -r venv/                           # Nukes the old environment
$ python3 -m venv venv/                 # Makes a blank new one
$ pip install -r requirements.txt       # Re-installs dependencies