conda create -n my_pytrilinos_env python=3.8
source activate my_pytrilinos_env
MPICC="cc -shared" pip install --no-binary=mpi4py mpi4py==3.0.3
MPICC="cc -shared" pip install --no-binary=numpy numpy==1.20.2