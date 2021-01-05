from PyTrilinos import Tpetra
from PyTrilinos import Teuchos
import numpy as np
from PyAlbany import Utils

def main(parallelEnv):
    comm = Teuchos.DefaultComm.getComm()
    filename = 'input_conductivity_dist_paramT.yaml'
    problem = Utils.createAlbanyProblem(filename, parallelEnv)

    # We can get from the Albany problem the map of a distributed parameter:
    parameter_map = problem.getParameterMap(0)

    # This map can then be used to construct an RCP to a Tpetra::Multivector:
    m_directions = 4
    directions = Tpetra.MultiVector(parameter_map, m_directions, dtype="d")

    # Numpy operations, such as assignments, can then be performed on the local entries:
    directions[0,:] = 1.        # Set all entries of v_0 to   1
    directions[1,:] = -1.       # Set all entries of v_1 to  -1
    directions[2,:] = 3.        # Set all entries of v_2 to   3
    directions[3,:] = -3.       # Set all entries of v_3 to  -3

    # Now that we have an RCP to the directions, we provide it to the Albany problem:
    problem.setDirections(0, directions)

    # Finally, we can solve the problem (which includes applying the Hessian to the directions) 
    # and get the Hessian-vector products:
    problem.performSolve()
    hessian = problem.getReducedHessian(0, 0)

if __name__ == "__main__":
    comm = Teuchos.DefaultComm.getComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    main(parallelEnv)
