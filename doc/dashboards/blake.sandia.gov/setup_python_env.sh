#!/bin/bash

# gcc-env
python -m venv /home/projects/albany/tpls/python/gcc/12.2.0/openmpi/4.1.5/gcc-env
source /home/projects/albany/tpls/python/gcc/12.2.0/openmpi/4.1.5/gcc-env/bin/activate
pip install -r requirements.txt --no-cache-dir
deactivate

# oneapi-env
python -m venv /home/projects/albany/tpls/python/oneapi/2023.2.0/mpi/2021.10.0/oneapi-env
source /home/projects/albany/tpls/python/oneapi/2023.2.0/mpi/2021.10.0/oneapi-env/bin/activate
export MPICC=/projects/x86-64-icelake-rocky8/tpls/intel-oneapi-mpi/2021.10.0/oneapi/2023.2.0/base/7jy7xel/mpi/2021.10.0/bin/mpiicc
export MPICXX=/projects/x86-64-icelake-rocky8/tpls/intel-oneapi-mpi/2021.10.0/oneapi/2023.2.0/base/7jy7xel/mpi/2021.10.0/bin/mpiicpc
export MPIF90=/projects/x86-64-icelake-rocky8/tpls/intel-oneapi-mpi/2021.10.0/oneapi/2023.2.0/base/7jy7xel/mpi/2021.10.0/bin/mpiifx
pip install -r requirements.txt --no-cache-dir

