from PyAlbany import Utils
from mpi4py import MPI

import numpy as np
import os

def main(parallelEnv):
    # This example illustrates how PyAlbany can be used to perform HDS

    comm = parallelEnv.getComm()
    rank = comm.getRank()
    nprocs = comm.getSize()

    file_dir = os.path.dirname(__file__)

    filename_fwd_opt = 'input_hdsa_advDiff3D_lf_fwd_and_opt_vec.yaml'
    filename_hf_fwd_opt = 'input_hdsa_advDiff3D_hf_fwd_and_opt_vec.yaml'
    filename_hdsa = 'input_hdsa_advDiff3D_vec.yaml'

    timers = Utils.createTimers(["PyAlbany: Create Albany Problem", 
                            "PyAlbany: Perform Analysis",
                            "PyAlbany: Perform Solve",
                            "PyAlbany: Total"])

    timers[3].start()
    solve_inverse_problem=True

    pList = Utils.createParameterList(filename_fwd_opt, parallelEnv)
    timers[0].start()
    
    #problem = Utils.createAlbanyProblem(filename_fwd_opt, parallelEnv)
    problem = Utils.createAlbanyProblem(pList, parallelEnv)
    timers[0].stop()

    if(solve_inverse_problem):
      timers[1].start()
      problem.performAnalysis()
      timers[1].stop()
    
      param = problem.getParameter(0)
      Utils.writeMVector('p_opt_3d', param, distributedFile=False, useBinary=True)
    else:
      p_opt_3d = np.loadtxt('../AsciiMeshes/HDSA_AdvDiff/p_opt_3d.ascii')
      p_opt_3d = p_opt_3d[1:]
      np.save('p_opt_3d',p_opt_3d)
      
    
    parameter_map = problem.getParameterMap(0)
    problem.performSolve()
    state = problem.getState()
    Utils.writeMVector('x_opt_3d', state, distributedFile=False, useBinary=True)


    #Here we write the nominal parameter and solution in Albany ASCII format
    if(MPI.COMM_WORLD.rank == 0):
      state = np.load('x_opt_3d.npy')
      with open('x_opt_3d.ascii', 'w') as f:
        f.write(str(round(state.shape[0]/2)) + ' 2\n')
        np.savetxt(f, state[0::2])
        np.savetxt(f, state[1::2])
      param = np.load('p_opt_3d.npy')
      with open('p_opt_3d.ascii', 'w') as f:
        f.write(str(param.shape[0]) + '\n')
        np.savetxt(f, param)
      p_1_3d = np.loadtxt('../AsciiMeshes/HDSA_AdvDiff/p_1_3d.ascii')
      p_1_3d = p_1_3d[1:]
      np.save('p_1_3d',p_1_3d)


    param = Utils.loadMVector("./p_1_3d", 1, parameter_map, distributedFile=False, useBinary=True).getVector(0)
    problem.setParameter(0,param)
    problem.performSolve()
    state = problem.getState()
    Utils.writeMVector('x_p_1_3d', state, distributedFile=False, useBinary=True)


    #Here we write the nominal parameter and solution in Albany ASCII format
    if(MPI.COMM_WORLD.rank == 0):
      state = np.load('x_p_1_3d.npy')
      with open('x_p_1_3d.ascii', 'w') as f:
        f.write(str(round(state.shape[0]/2)) + ' 2\n')
        np.savetxt(f, state[0::2])
        np.savetxt(f, state[1::2])
      param = np.load('p_1_3d.npy')
      with open('p_1_3d.ascii', 'w') as f:
        f.write(str(param.shape[0]) + '\n')
        np.savetxt(f, param)

    problem_hf = Utils.createAlbanyProblem(filename_hf_fwd_opt, parallelEnv)
    
    parameter_map = problem_hf.getParameterMap(0)
    param = Utils.loadMVector("./p_opt_3d", 1, parameter_map, distributedFile=False, useBinary=True).getVector(0)
    problem_hf.setParameter(0,param)
    timers[2].start()
    problem_hf.performSolve()
    timers[2].stop()
    state = problem_hf.getState()
    Utils.writeMVector('x_hf_p_opt_3d', state, distributedFile=False, useBinary=True)
    state_opt = Utils.loadMVector("x_opt_3d", 1, state.getMap(), distributedFile=False, useBinary=True).getVector(0)
    #state = -1.0*state_opt + 1.0*state
    state.update(-1.0, state_opt, 1.0)
    Utils.writeMVector('x_diff_p_opt_3d', state, distributedFile=False, useBinary=True)
    
    if(MPI.COMM_WORLD.rank == 0):
      state_diff = np.load('x_diff_p_opt_3d.npy')
      with open('x_diff_p_opt_3d.ascii', 'w') as f:
        f.write(str(round(state_diff.shape[0]/2)) + ' 2\n')
        np.savetxt(f, state_diff[0::2])
        np.savetxt(f, state_diff[1::2])

    param = Utils.loadMVector("p_1_3d", 1, parameter_map, distributedFile=False, useBinary=True).getVector(0)
    problem_hf.setParameter(0,param)
    timers[2].start()
    problem_hf.performSolve()
    timers[2].stop()
    state = problem_hf.getState()
    Utils.writeMVector('x_hf_p_1_3d', state, distributedFile=False, useBinary=True)
    state_p_1 = Utils.loadMVector("x_p_1_3d", 1, state.getMap(), distributedFile=False, useBinary=True).getVector(0)
    state.update(-1.0, state_p_1, 1.0)
    Utils.writeMVector('x_diff_p_1_3d', state, distributedFile=False, useBinary=True)
    
    if(MPI.COMM_WORLD.rank == 0):
      state_diff = np.load('x_diff_p_1_3d.npy')
      with open('x_diff_p_1_3d.ascii', 'w') as f:
        f.write(str(round(state_diff.shape[0]/2)) + ' 2\n')
        np.savetxt(f, state_diff[0::2])
        np.savetxt(f, state_diff[1::2])


    problem_hdsa = Utils.createAlbanyProblem(filename_hdsa, parallelEnv)
    parameter_map = problem_hdsa.getParameterMap(0)
    param = Utils.loadMVector("./p_opt_3d", 1, parameter_map, distributedFile=False, useBinary=True).getVector(0)
    problem_hdsa.setParameter(0,param)
    
    state_map = problem_hdsa.getParameterMap(1)
    param = Utils.loadMVector("./x_opt_3d", 1, state_map, distributedFile=False, useBinary=True).getVector(0)
    problem_hdsa.setParameter(1,param)

    parameter_map = problem_hdsa.getParameterMap(2)
    param = Utils.loadMVector("./p_opt_3d", 1, parameter_map, distributedFile=False, useBinary=True).getVector(0)
    problem_hdsa.setParameter(2,param)
    
    state_map = problem_hdsa.getParameterMap(3)
    param = Utils.loadMVector("./x_diff_p_opt_3d", 1, state_map, distributedFile=False, useBinary=True).getVector(0)
    problem_hdsa.setParameter(3,param)
    
    parameter_map = problem_hdsa.getParameterMap(4)
    param = Utils.loadMVector("./p_1_3d", 1, parameter_map, distributedFile=False, useBinary=True).getVector(0)
    problem_hdsa.setParameter(4,param)
    
    state_map = problem_hdsa.getParameterMap(5)
    param = Utils.loadMVector("./x_diff_p_1_3d", 1, state_map, distributedFile=False, useBinary=True).getVector(0)
    problem_hdsa.setParameter(5,param)
    
    problem_hdsa.performAnalysis()

    #product of reduced hessian with directions
    Utils.printTimers(timers, "timers_nprocs_"+str(nprocs)+".txt")

if __name__ == "__main__":
    parallelEnv = Utils.createDefaultParallelEnv()
    main(parallelEnv)
