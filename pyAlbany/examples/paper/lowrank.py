from mpi4py import MPI
import numpy as np
from PyAlbany import Utils
from PyAlbany.RandomizedCompression import doublePass
import os
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class Hessian:
   def __init__(self, problem, parameterIndex, responseIndex):
       self.problem = problem
       self.parameterIndex = parameterIndex
       self.responseIndex  = responseIndex
       self.Map            = self.problem.getParameterMap(self.parameterIndex)
   def dot(me, x):
       self.problem.setDirections(self.parameterIndex, x)
       self.problem.performSolve()
       return self.problem.getReducedHessian(self.responseIndex, self.parameterIndex) 


def main(parallelEnv):
    myGlobalRank = MPI.COMM_WORLD.rank

    # Create an Albany problems:
    filename = "input_distributed.yaml"
    paramList = Utils.createParameterList(filename, parallelEnv)
    paramListDataMisfit = Utils.createParameterList(filename, parallelEnv)
    paramListDataMisfit.sublist("Problem").sublist("Response Functions").sublist("Response 0").sublist("Response 1").set("Scaling", 0.0)
    paramListDataMisfit.sublist("Discretization").set("Exodus Output File Name", "steady2d_DataMisfit.exo")

    problem = Utils.createAlbanyProblem(paramList, parallelEnv)
    problem.performAnalysis()
    problemDataMisfit = Utils.createAlbanyProblem(paramListDataMisfit, parallelEnv)
    problemDataMisfit.setParameter(0, problem.getParameter(0)) 
    problemDataMisfit.performSolve()
    
    parameterIndex = 0
    responseIndex  = 0
    Hess = Hessian(problemDataMisfit, parameterIndex, responseIndex)
    
    k = 100
    eigVals, eigVecs = doublePass(Hess, k, symmetric=True)
    eigVals = np.abs(eigVals)
    eigVals = eigVals[np.argsort(eigVals)[::-1]]
    if myGlobalRank == 0:
        fig = plt.figure(figsize=(6,4))
        plt.plot(eigVals)
        plt.ylabel('Eigenvalues of the Hessian')
        plt.xlabel('Eigenvalue index')
        plt.gca().set_xlim([0, k])
        #plt.gca().set_ylim([0, 2.7e-3])
        plt.grid(True, which="both")
        fig.tight_layout()
        plt.savefig('hessian_eigenvalues.jpeg', dpi=800)
        plt.close()
    

if __name__ == "__main__":
    parallelEnv = Utils.createDefaultParallelEnv()
    main(parallelEnv)
