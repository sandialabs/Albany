from PyAlbany import Utils

# As discussed in example_0.py parallelEnv should be destructed
# after the destruction of all the other variables which rely on
# Kokkos.
# A better way to enforce this than calling the destructor explicitly
# for all the variables is to rely on the garbage collector capability
# of python using the scope of a function.

# At the end of the function all the internal objects (all objects which
# are not inputs or outputs) are destructed.
# Therefore, we can create a function which takes parallelEnv as input in which
# all Kokkos-related objects will be created and will not be passed as output; 
# they will necessarily be destructed before parallelEnv.

# This example revisits example_0.py with the above-mentioned approach.
def main(parallelEnv):
    filename = 'input_conductivity_dist_paramT.yaml'
    problem = Utils.createAlbanyProblem(filename, parallelEnv)

if __name__ == "__main__":
    comm = Utils.getDefaultComm()
    parallelEnv = Utils.createDefaultParallelEnv(comm)
    main(parallelEnv)
