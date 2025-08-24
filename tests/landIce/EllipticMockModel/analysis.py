from PyAlbany import Utils
from mpi4py import MPI

import numpy as np
import os

def main(parallelEnv):
    # This example illustrates how PyAlbany can be used to compute
    # reduced Hessian-vector products w.r.t to the basal friction.

    comm = parallelEnv.getComm()
    rank = comm.getRank()
    nprocs = comm.getSize()

    file_dir = os.path.dirname(__file__)

    filename = 'input_slab_mockmodel_analysis.yaml'

    parameter_index = 0
    response_index = 0

    timers = Utils.createTimers(["PyAlbany: Create Albany Problem", 
                            "PyAlbany: Perform Analysis",
                            "PyAlbany: Perform Solve",
                            "PyAlbany: Total"])

    timers[3].start()
    solve_inverse_problem=True
    n_directions=5
    generateDirections=True

    if(solve_inverse_problem):

      timers[0].start()
      problem = Utils.createAlbanyProblem(filename, parallelEnv)
      timers[0].stop()

      timers[1].start()
      problem.performAnalysis()
      timers[1].stop()
    
      param = problem.getParameter(parameter_index)
      Utils.writeMVector('param', param, distributedFile=False, useBinary=True)
    

      #Note, we need to run "perform solve" before getting the state. 
      problem.performSolve()
      state = problem.getState()
      Utils.writeMVector('state', state, distributedFile=False, useBinary=True)



    #We now compute the Gradient and Hessian vec product using the linearized model


    #Here we write the nominal parameter and solution in Albany ASCII format
    if(MPI.COMM_WORLD.rank == 0):
      state = np.load('state.npy')
      with open('nominal_solution.ascii', 'w') as f:
        f.write(str(state.shape[0]) + '\n')
        np.savetxt(f, state)
      param = np.load('param.npy')
      with open('nominal_parameter.ascii', 'w') as f:
        f.write(str(param.shape[0]) + '\n')
        np.savetxt(f, param)


    filename = 'input_slab_mockmodel_linearized.yaml'
    problem = Utils.createAlbanyProblem(filename, parallelEnv)
    parameter_map = problem.getParameterMap(0)
    param = Utils.loadMVector("./param", 1, parameter_map, distributedFile=False, useBinary=True).getVector(0)
   
    

    timers[2].start()

    #it's annoying but we need to set the directions before we solve the problem
    if (MPI.COMM_WORLD.rank == 0) and generateDirections:
       rng = np.random.default_rng(20250820) 
       N = parameter_map.getGlobalNumElements() 
       omegaRoot = rng.normal(size=(n_directions, N)) 
       np.save("./directions-"+str(n_directions), omegaRoot) 


    directions = Utils.loadMVector("./directions-"+str(n_directions), n_directions, parameter_map, distributedFile=False, useBinary=True)


    #perturb parameter (this will be then done by drawing samples from the Laplace approximation)
    #using the first direction component as perturbation, for simplicity

    #param = 0.1*diretictions(0) + 1.0*param
    param.update(0.1, directions.getVector(0), 1.0)
    problem.setParameter(0,param)


    problem.setDirections(parameter_index, directions)
    problem.performSolve()
    gradient = problem.getSensitivity(response_index, parameter_index)

    #product of reduced hessian with directions
    hessian_vec_prod = problem.getReducedHessian(response_index, parameter_index)
    timers[2].stop()

    hessian_view = hessian_vec_prod.getLocalView()
    gradient_view = gradient.getLocalView()

    norm2 = 0.0
    for i in range(n_directions): 
      norm2 += hessian_vec_prod.getVector(i).dot(hessian_vec_prod.getVector(i))
    hess_vp_norm = np.sqrt(norm2)
    
    gradient_norm = np.sqrt(gradient.getVector(0).dot(gradient.getVector(0)))

    if (MPI.COMM_WORLD.rank == 0):
      print("H_vp(0,:): ", hessian_view[0,:], ", ||H_vp||: ", hess_vp_norm)
      print("G(0): ", gradient_view[0,:], ", ||G||: ", gradient_norm)
   
    timers[3].stop()

    Utils.printTimers(timers, "timers_nprocs_"+str(nprocs)+".txt")

if __name__ == "__main__":
    parallelEnv = Utils.createDefaultParallelEnv()
    main(parallelEnv)
