from PyAlbany import Utils

def main(parallelEnv):
    filename = 'input_conductivity_dist_paramT.yaml'
    problem = Utils.createAlbanyProblem(filename, parallelEnv)
    
    # Now that the Albany problem is constructed, we can solve
    # it and evaluate the response:
    problem.performSolve()
    response = problem.getResponse(0)
    print(response.getLocalView())

if __name__ == "__main__":
    comm = Utils.getDefaultComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    main(parallelEnv)
