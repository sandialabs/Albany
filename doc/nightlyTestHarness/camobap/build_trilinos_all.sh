#!/bin/bash

./run_trilinos.sh set_irinas_env.in MPI >& out #Trilinos with Serial KokkosNode
./run_trilinos_openmp.sh set_irinas_env.in MPI >& out_openMP  #Trilinos with OpenMP KokkosNode 
./run_trilinos_debug.sh set_irinas_env.in MPI >& out_openDebug  #Trilinos with OpenMP KokkosNode 
