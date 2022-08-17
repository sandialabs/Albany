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
   def __init__(me, problem, parameterIndex, responseIndex):
       me.problem = problem
       me.parameterIndex = parameterIndex
       me.responseIndex  = responseIndex
       me.Map            = me.problem.getParameterMap(me.parameterIndex)
   def dot(me, x):
       me.problem.setDirections(me.parameterIndex, x)
       me.problem.performSolve()
       return me.problem.getReducedHessian(me.responseIndex, me.parameterIndex) 


def main(parallelEnv):
    myGlobalRank = MPI.COMM_WORLD.rank

    # Create an Albany problem:
    filename = "input_distributed.yaml"
    parameter = Utils.createParameterList(filename, parallelEnv)

    problem = Utils.createAlbanyProblem(parameter, parallelEnv)
    problem.performAnalysis()
    parameterDataMisfit = Utils.createParameterList(filename, parallelEnv)
    parameterDataMisfit.sublist("Problem").sublist("Response Functions").sublist("Response 0").sublist("Response 1").set("Scaling", 0.0)
    
    parameterDataMisfit.sublist("Discretization").set("Exodus Output File Name", "steady2d_DataMisfit.exo")
    problemDataMisfit = Utils.createAlbanyProblem(parameterDataMisfit, parallelEnv)
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
        plt.semilogy(eigVals)
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
