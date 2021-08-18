from PyTrilinos import Tpetra
from PyTrilinos import Teuchos
import numpy as np
from PyAlbany import Utils
import os

class PyAlbanyInstance:
    def __init__(self, filename, parallelEnv):
        self._parallelEnv = parallelEnv
        self._filename = filename
    def solve(self, damping1, damping2):
        parameters = Utils.createParameterList(filename, parallelEnv)
        parameters.sublist("Piro").sublist("NOX").sublist("Direction").sublist("Newton").sublist("Stratimikos Linear Solver").sublist("Stratimikos").sublist("Preconditioner Types").sublist("MueLu").sublist("Factories").sublist("mySmoother3").sublist("ParameterList").set("relaxation: damping factor", damping1)
        parameters.sublist("Piro").sublist("NOX").sublist("Direction").sublist("Newton").sublist("Stratimikos Linear Solver").sublist("Stratimikos").sublist("Preconditioner Types").sublist("MueLu").sublist("Factories").sublist("mySmoother4").sublist("ParameterList").set("relaxation: damping factor", damping2)
        problem = Utils.createAlbanyProblem(parameters, parallelEnv)
        problem.performSolve()
        stackedTimer = problem.getStackedTimer()
        solve_time = stackedTimer.accumulatedTime("Albany: performSolve")
        return solve_time

if __name__ == "__main__":
    # initialize
    comm = Teuchos.DefaultComm.getComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)

    # populate mesh
    filename = 'input_albany_PopulateMesh_Wedge.yaml'
    problem = Utils.createAlbanyProblem(filename, parallelEnv)
    problem.performSolve()
    problem = None

    # run experiment
    filename = 'input_albany_Velocity_MueLu_Wedge_Tune.yaml'
    solver = PyAlbanyInstance(filename, parallelEnv)
    solve_time = solver.solve(1.0, 0.1)

    if comm.getRank() == 0:
        print(solve_time)
    comm.barrier()

    solver = None
    problem = None
    parallelEnv = None
    comm = None