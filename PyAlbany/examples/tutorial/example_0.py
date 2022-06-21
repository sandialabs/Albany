# First, import Utils from PyAlbany:
from PyAlbany import Utils

# Then, the parallel environment is initialized (including Kokkos):
comm = Utils.getDefaultComm()
parallelEnv = Utils.createDefaultParallelEnv(comm,
                                             n_threads=-1,
                                             n_numa=-1,
                                             device_id=-1)

# (Kokkos finalize will be called during the destruction of parallelEnv;
# we will have to enforce that this destructor is called after the destruction
# of every object which relies on Kokkos.)

# Finally, given a filename and the parallel environment, an Albany problem is constructed:
filename = 'input_conductivity_dist_paramT.yaml'
problem = Utils.createAlbanyProblem(filename, parallelEnv)

# Now, we call the problem destructor first (by setting the RCP to null):
problem = None
# And we call the parallelEnv destructor:
parallelEnv = None
