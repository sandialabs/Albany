from mpi4py import MPI
from PyAlbany import Utils
import glob
import os
import sys
import subprocess

def getLogName(filename, postfix):
    return "log_"+filename.split('.')[0]+postfix

def performGradientChecks(filename, parallelEnv, after):

    # Create an Albany problem:
    paramList = Utils.createParameterList(
        filename, parallelEnv
    )

    paramList.sublist("Piro").sublist("Analysis").sublist("ROL").set("Check Derivatives", True)
    paramList.sublist("Piro").sublist("Analysis").sublist("ROL").set("Perform Optimization", after)

    problem = Utils.createAlbanyProblem(paramList, parallelEnv)
    if after:
        print('Perform gradient checks for file ' + filename + ' after optimization')
    else:
        print('Perform gradient checks for file ' + filename + ' before optimization')
    problem.performAnalysis()

if len(sys.argv) == 1:

    # All the transient optimization inputs of this directory
    filenames = sorted(glob.glob("transient_opt_*.yaml"))

    my_env = os.environ.copy()
    for filename in filenames:
        my_cmd = ['python', 'performGradientChecks.py', filename, 'before']
        with open(getLogName(filename, "_0.txt"), "w") as outfile:
            subprocess.run(my_cmd, stdout=outfile, env=my_env)
        my_cmd = ['python', 'performGradientChecks.py', filename, 'after']
        with open(getLogName(filename, "_1.txt"), "w") as outfile:
            subprocess.run(my_cmd, stdout=outfile, env=my_env)

else:
    parallelEnv = Utils.createDefaultParallelEnv()
    myGlobalRank = MPI.COMM_WORLD.rank
    performGradientChecks(sys.argv[1], parallelEnv, sys.argv[2]=='after')
