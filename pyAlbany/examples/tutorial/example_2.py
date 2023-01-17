import numpy as np
from PyAlbany import Utils

def main(parallelEnv):
    comm = parallelEnv.getComm()
    filename = 'input_conductivity_dist_paramT.yaml'
    problem = Utils.createAlbanyProblem(filename, parallelEnv)
    problem.performSolve()

    # We can solve the problem and extract the sensitivity w.r.t. a parameter:
    sensitivity = problem.getSensitivity(0, 0)

    # In this example, we illustrate how to return values as output without
    # relying on Kokkos-related object; the local data of the vectors are deeply
    # copied to a new numpy array:
    sensitivity_out = np.copy(sensitivity.getLocalView()[:,0])
    return sensitivity_out

if __name__ == "__main__":
    comm = Utils.getDefaultComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    sensitivity = main(parallelEnv)

    # The returned sensitivity is a vector distributed over the different MPI processes. 
    # Each process can access its local entries (using local index):
    print("MPI process "+str(comm.getRank())+" has "+str(len(sensitivity[:]))+
          " local entries and the first one is "+str(sensitivity[0]))
