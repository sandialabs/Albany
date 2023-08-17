cmake_minimum_required (VERSION 3.16)
set (Albany_CXX_COMPILER ${CMAKE_CXX_COMPILER})
project (install_omegah)
include (ExternalProject)

ExternalProject_Add(Omega_h
  GIT_REPOSITORY https://github.com/SCOREC/omega_h
  GIT_TAG origin/master
  CMAKE_ARGS
    -DCMAKE_CXX_COMPILER=${Albany_CXX_COMPILER}
    -DCMAKE_INSTALL_PREFIX=${Albany_BINARY_DIR}/tpls/omegah/install
    -DKokkos_PREFIX=${Kokkos_INSTALL_DIR}
    -DOmega_h_USE_Kokkos=ON
    -DOmega_h_USE_MPI=ON
)
