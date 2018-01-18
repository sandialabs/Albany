#!/bin/bash

module purge
module load sems-env
module load sems-gcc/6.1.0
module load sems-openmpi/1.8.7
module load sems-boost/1.63.0/base
module load sems-netcdf/4.4.1/exo_parallel
module load sems-cmake 
module list 
#set                     boost_inc       $env(SEMS_BOOST_INCLUDE_PATH)
#set                     boost_lib       $env(SEMS_BOOST_LIBRARY_PATH)
#set                     mpi_root        $env(SEMS_MPI_ROOT)
#set                     netcdf_root     $env(SEMS_NETCDF_ROOT)
#set                     netcdf_inc      $env(SEMS_NETCDF_INCLUDE_PATH)
#set                     netcdf_lib      $env(SEMS_NETCDF_LIBRARY_PATH)
#setenv                  BOOST_INC       $boost_inc
#setenv                  BOOST_LIB       $boost_lib
#setenv                  BOOSTLIB_INC    $boost_inc
#setenv                  BOOSTLIB_LIB    $boost_lib
#setenv                  MPI_INC         $mpi_root/include
#setenv                  MPI_LIB         $mpi_root/lib
#setenv                  MPI_BIN         $mpi_root/bin
#setenv                  LCM_NETCDF_PARALLEL ON
#setenv                  LCM_LINK_FLAGS  -L$boost_lib
#setenv                  NETCDF          $netcdf_root
#setenv                  NETCDF_INC      $netcdf_inc
#setenv                  NETCDF_LIB      $netcdf_lib
#prepend-path            LD_LIBRARY_PATH $mpi_root/lib
#prepend-path            LD_LIBRARY_PATH $netcdf_lib
#prepend-path PATH $mpi_root/bin
