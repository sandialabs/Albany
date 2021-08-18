from PyTrilinos import Tpetra
from PyTrilinos import Teuchos
import numpy as np
from PyAlbany import Utils
import os

class PyAlbanyInstance:
    def __init__(self, filename, parallelEnv):
        self._parallelEnv = parallelEnv
        self._filename = filename
    def solve(self, tuningParameters):

        # Get the problem's parameter list from PyAlbany
        parameters = Utils.createParameterList(filename, parallelEnv)

        # Get the sublist for mySmoother3
        mySmoother3 = parameters.sublist("Piro").sublist("NOX").sublist("Direction").sublist("Newton").sublist("Stratimikos Linear Solver").sublist("Stratimikos").sublist("Preconditioner Types").sublist("MueLu").sublist("Factories").sublist("mySmoother3")

        # Set the smoother parameters according to tuningParameters
        if tuningParameters[0] == 'MT Gauss-Seidel':
            mySmoother3.set('type', 'RELAXATION')
            pList = Teuchos.ParameterList()
            pList.set('relaxation: type',                     'Two-stage Gauss-Seidel')
            pList.set('relaxation: sweeps',                   tuningParameters[1])
            pList.set('relaxation: damping factor',           tuningParameters[2])
            mySmoother3.set('ParameterList', pList)
        elif tuningParameters[0] == 'Two-stage Gauss-Seidel':
            mySmoother3.set('type', 'RELAXATION')
            pList = Teuchos.ParameterList()
            pList.set('relaxation: type',                     'Two-stage Gauss-Seidel')
            pList.set('relaxation: sweeps',                   tuningParameters[1])
            pList.set('relaxation: inner damping factor',     tuningParameters[3])
            mySmoother3.set('ParameterList', pList)
        elif tuningParameters[0] == 'Chebyshev':
            mySmoother3.set('type', 'CHEBYSHEV')
            pList = Teuchos.ParameterList()
            pList.set('chebyshev: degree',                    tuningParameters[4])
            pList.set('chebyshev: ratio eigenvalue',          tuningParameters[5])
            pList.set('chebyshev: eigenvalue max iterations', tuningParameters[6])
            mySmoother3.set('ParameterList', pList)
        else:
            print("Invalid tuning paremeters, using default mySmoother3.")

        # DEBUG: Print mySmoother3 parameterList
        if parallelEnv.comm.getRank() == 0:
            print(mySmoother3)
        parallelEnv.comm.barrier()

        # Create and solve problem
        problem = Utils.createAlbanyProblem(parameters, parallelEnv)
        problem.performSolve()

        # Get timing information for performSolve
        stackedTimer = problem.getStackedTimer()
        solve_time = stackedTimer.accumulatedTime("PyAlbany: performSolve")

        return solve_time

if __name__ == "__main__":
    # initialize
    comm = Teuchos.DefaultComm.getComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)

    # populate mesh
    filename = 'inputs/input_albany_PopulateMesh_Wedge.yaml'
    problem = Utils.createAlbanyProblem(filename, parallelEnv)
    problem.performSolve()
    problem = None

    # set up pyalbany instance
    filename = 'inputs/input_albany_Velocity_MueLu_Wedge_Tune.yaml'
    solver = PyAlbanyInstance(filename, parallelEnv)

    # input format: ['smoother name', 'relaxation: sweeps', 'relaxation: damping factor',
    #                'relaxation: inner damping factor', 'chebyshev: degree',
    #                'chebyshev: ratio eigenvalue', 'chebyshev: eigenvalue max iterations']
    # example inputs
    tuningParameters = [['MT Gauss-Seidel', 5, 0.75, 1.25, 1, 3.14, 20],
                        ['Two-stage Gauss-Seidel', 5, 0.75, 1.25, 1, 3.14, 20],
                        ['Chebyshev', 5, 0.75, 1.25, 1, 3.14, 20],
                        ['other']]

    # run experiments
    nexps = len(tuningParameters)
    solve_times = np.zeros((nexps,))
    for i in range(nexps):
        solve_times[i] = solver.solve(tuningParameters[i])

    # report solve times
    if comm.getRank() == 0:
        print(solve_times)
    comm.barrier()

    # cleanup
    solver = None
    problem = None
    parallelEnv = None
    comm = None