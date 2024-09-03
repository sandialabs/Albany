from mpi4py import MPI
from PyAlbany import Utils

def performGradientChecks(filename, parallelEnv):

    # Create an Albany problem:
    paramList = Utils.createParameterList(
        filename, parallelEnv
    )

    paramList.sublist("Piro").sublist("Analysis").sublist("ROL").set("Check Derivatives", True)
    paramList.sublist("Piro").sublist("Analysis").sublist("ROL").set("Perform Optimization", False)

    problem = Utils.createAlbanyProblem(paramList, parallelEnv)
    print('Perform gradient checks for file ' + filename + ' before optimization')
    problem.performAnalysis()

    paramList = Utils.createParameterList(
        filename, parallelEnv
    )

    paramList.sublist("Piro").sublist("Analysis").sublist("ROL").set("Check Derivatives", True)
    paramList.sublist("Piro").sublist("Analysis").sublist("ROL").set("Perform Optimization", True)

    problem = Utils.createAlbanyProblem(paramList, parallelEnv)
    print('Perform gradient checks for file ' + filename + ' after optimization')
    problem.performAnalysis()


filenames = ["tempus_be_nox_analysis_integrated_response_distributed_param.yaml", 
             "tempus_be_nox_analysis_integrated_response_scalar_params.yaml", 
             "tempus_be_nox_analysis_rol_driven_scalar_params_2.yaml", 
             "tempus_be_nox_analysis_rol_driven_scalar_params.yaml", 
             "tempus_be_nox_analysis_tempus_driven_scalar_params_2.yaml", 
             "tempus_be_nox_analysis_tempus_driven_scalar_params.yaml"]

parallelEnv = Utils.createDefaultParallelEnv()
myGlobalRank = MPI.COMM_WORLD.rank

for filename in filenames:
    performGradientChecks(filename, parallelEnv)
