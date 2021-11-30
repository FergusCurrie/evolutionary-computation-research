Collection of work for my AIML research. Multi-objective ensembles. Where I can i'll use my own implmentation.


# File Structure
|---data
    |---raw
    |---processed
|--models
|--notebooks
|--references
|--reporting
|--src
|--venv


# Enviroment Stuff: 

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