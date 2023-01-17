from PyAlbany import Utils

import numpy as np
import os

def main(parallelEnv):
    # This example illustrates how PyAlbany can be used to compute
    # reduced Hessian-vector products w.r.t to the basal friction.

    comm = parallelEnv.getComm()
    rank = comm.getRank()
    nprocs = comm.getSize()

    file_dir = os.path.dirname(__file__)

    filename = 'input_fo_gis_analysis_beta_smbT.yaml'

    parameter_index = 0
    response_index = 0

    timers = Utils.createTimers(["PyAlbany: Create Albany Problem", 
                            "PyAlbany: Read multivector directions",
                            "PyAlbany: Set directions",
                            "PyAlbany: Perform Solve",
                            "PyAlbany: Get Reduced Hessian",
                            "PyAlbany: Write Reduced Hessian",
                            "PyAlbany: Total"])

    timers[6].start()
    timers[0].start()
    problem = Utils.createAlbanyProblem(filename, parallelEnv)
    timers[0].stop()

    timers[1].start()
    n_directions=4
    parameter_map = problem.getParameterMap(0)
    directions = Utils.loadMVector('random_directions', n_directions, parameter_map, distributedFile = False, useBinary = True)
    timers[1].stop()

    timers[2].start()
    problem.setDirections(parameter_index, directions)
    timers[2].stop()

    timers[3].start()
    problem.performSolve()
    timers[3].stop()

    timers[4].start()
    hessian = problem.getReducedHessian(response_index, parameter_index)
    timers[4].stop()

    timers[5].start()
    Utils.writeMVector("hessian_nprocs_"+str(nprocs), hessian, distributedFile = True, useBinary = False)
    Utils.writeMVector("hessian_all_nprocs_"+str(nprocs), hessian, distributedFile = False, useBinary = False)
    timers[5].stop()

    hessian_view = hessian.getLocalView()
    expected_hessian = np.array([-0.000195717646, -0.000272421749, -4.99117153e-05, -0.000453498014])
    print(hessian_view[0,:])
    print(expected_hessian)

    timers[6].stop()

    Utils.printTimers(timers, "timers_nprocs_"+str(nprocs)+".txt")

if __name__ == "__main__":
    parallelEnv = Utils.createDefaultParallelEnv()
    main(parallelEnv)
