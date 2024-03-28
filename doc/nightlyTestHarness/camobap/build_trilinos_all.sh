#!/bin/bash

export PATH=/nightlyCDash/albany-tpls-gcc-11.1.1-openmpi-4.1.0/bin:$PATH:/tpls/install/ninja/build-cmake
which cmake
cmake --version
rm -rf Results
./run_trilinos.sh set_irinas_env.in MPI >& out #Trilinos with Serial KokkosNode
./run_trilinos_openmp.sh set_irinas_env.in MPI >& out_openMP  #Trilinos with OpenMP KokkosNode 
./run_trilinos_debug.sh set_irinas_env.in MPI >& out_Debug  #Trilinos with Serial KokkosNode + dbg build 
