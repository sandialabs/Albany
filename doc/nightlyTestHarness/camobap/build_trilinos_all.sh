#!/bin/bash

export PATH=/home/ikalash/Install/cmake-3.24.3/bin:$PATH:/tpls/install/ninja/build-cmake
rm -rf Results
./run_trilinos.sh set_irinas_env.in MPI >& out #Trilinos with Serial KokkosNode
./run_trilinos_openmp.sh set_irinas_env.in MPI >& out_openMP  #Trilinos with OpenMP KokkosNode 
./run_trilinos_debug.sh set_irinas_env.in MPI >& out_Debug  #Trilinos with Serial KokkosNode + dbg build 
