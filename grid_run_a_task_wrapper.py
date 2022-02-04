from run_a_task import run
import sys

jobid = int(sys.argv[1])
taskid = int(sys.argv[2])
run(jobid, taskid)