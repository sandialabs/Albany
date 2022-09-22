from PyAlbany import Utils

def main(parallelEnv):
    filename = 'input_conductivity_dist_paramT.yaml'
    problem = Utils.createAlbanyProblem(filename, parallelEnv)

    # We can get from the Albany problem the map of a distributed parameter:
    parameter_map = problem.getParameterMap(0)

    # This map can then be used to construct an RCP to a Tpetra::Multivector:
    m_directions = 4
    directions = Utils.createMultiVector(parameter_map, m_directions)

    directions_view = directions.getLocalView()

    # Numpy operations, such as assignments, can then be performed on the local entries:
    directions_view[:,0] = 1.        # Set all entries of v_0 to   1
    directions_view[:,1] = -1.       # Set all entries of v_1 to  -1
    directions_view[:,2] = 3.        # Set all entries of v_2 to   3
    directions_view[:,3] = -3.       # Set all entries of v_3 to  -3

    directions.setLocalView(directions_view)

    # Now that we have an RCP to the directions, we provide it to the Albany problem:
    problem.setDirections(0, directions)

    # Finally, we can solve the problem (which includes applying the Hessian to the directions) 
    # and get the Hessian-vector products:
    problem.performSolve()
    hessian = problem.getReducedHessian(0, 0)

if __name__ == "__main__":
    comm = Utils.getDefaultComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    main(parallelEnv)
