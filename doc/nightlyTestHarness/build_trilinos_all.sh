#!/bin/bash

./run_tpetra.sh set_irinas_env.in MPI >& out #Trilinos with Serial KokkosNode
./run_tpetra_openmp.sh set_irinas_env.in MPI >& out_openMP  #Trilinos with OpenMP KokkosNode 
./run_tpetra_debug.sh set_irinas_env.in MPI >& out_openDebug  #Trilinos with OpenMP KokkosNode 
