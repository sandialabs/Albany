from PyTrilinos import Tpetra
from PyTrilinos import Teuchos
import numpy as np
from PyAlbany import Utils

def main(parallelEnv):
    filename = 'input_conductivity_dist_paramT.yaml'
    problem = Utils.createAlbanyProblem(filename, parallelEnv)
    
    # Now that the Albany problem is constructed, we can solve
    # it and evaluate the response:
    problem.performSolve()
    response = problem.getResponse(0)
    print(response)

if __name__ == "__main__":
    comm = Teuchos.DefaultComm.getComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    main(parallelEnv)
