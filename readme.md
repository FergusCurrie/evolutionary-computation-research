Code base for masters. 

Contains lots of wrapper code for running an ensemble on the grid.

An ensemble has member generation, member selection and decision fusion defined. 

Implemented so far:
Member generaton:
    - DivBagging
    - DivNiching
    - GP
    - MOGP
    - OrMOGP
Member Selection:
    - greedyEnsemble
    - offEEL
DecisionFusion:
    -Voting


Experiments are defined in experiments/, they define a model, datasets and parameter setting.
A model is a member generation function, member selection function and decision fusion function. 

To run on grid : 
 1. Edit 'grid_run_a_task_wapper.py' and make sure the correct experiment is being run.
 2. Push to github
 3. Clone onto ECS in /vol/grid-solar/sgeusers/currieferg
 4. Determien number of tasks (datasets * parameters * models etc. shouldnt be too hard its in code )
 5. Run submission command : $qsub -t 1-n_tasks:1 -M currieferg@ecs.vuw.ac.nz -m be submission_script-task_array.sh 
 6. When its done push back

