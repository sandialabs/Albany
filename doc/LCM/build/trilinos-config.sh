#!/bin/bash

# Remove the CMake cache. For an extra clean start in an already-used build
# directory, rm -rf CMake* to get rid of all CMake-generated files.
if [ -f ./CMakeCache.txt ]; then
    rm CMakeCache.txt
fi

export OMPI_CC=ompi_cc
export OMPI_CXX=ompi_cxx
export OMPI_FC=ompi_fc

#
# The CMake command.
#
cmake \
 -D BUILD_SHARED_LIBS:BOOL=ON \
 -D CMAKE_BUILD_TYPE:STRING="build_type" \
 -D CMAKE_CXX_COMPILER:FILEPATH="/usr/lib64/openmpi/bin/mpicxx" \
 -D CMAKE_C_COMPILER:FILEPATH="/usr/lib64/openmpi/bin/mpicc" \
 -D CMAKE_Fortran_COMPILER:FILEPATH="/usr/lib64/openmpi/bin/mpif90" \
 -D CMAKE_INSTALL_PREFIX:PATH=install_dir \
 -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
\
 -D TPL_ENABLE_MPI:BOOL=ON \
 -D TPL_MPI_INCLUDE_DIRS:FILEPATH="/usr/include/openmpi-x86_64" \
 -D TPL_MPI_LIBRARY_DIRS:FILEPATH="/usr/lib64/openmpi/lib" \
 -D MPI_BIN_DIR:FILEPATH="/usr/lib64/openmpi/bin" \
\
 -D TPL_ENABLE_Boost:BOOL=ON \
 -D TPL_ENABLE_BoostLib:BOOL=ON \
 -D BoostLib_INCLUDE_DIRS:FILEPATH="/usr/include/boost" \
 -D BoostLib_LIBRARY_DIRS:FILEPATH="/usr/lib64" \
 -D Boost_INCLUDE_DIRS:FILEPATH="/usr/include/boost" \
\
 -D Trilinos_ASSERT_MISSING_PACKAGES:BOOL=OFF \
 -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
 -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
 -D Trilinos_ENABLE_CXX11:BOOL=ON \
 -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
 -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
 -D Trilinos_VERBOSE_CONFIGURE:BOOL=OFF \
 -D Trilinos_WARNINGS_AS_ERRORS_FLAGS:STRING="" \
\
 -D CUDA_PROPAGATE_HOST_FLAGS:BOOL=OFF \
 -D CUDA_VERBOSE_BUILD:BOOL=OFF \
 -D ENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=OFF \
 -D HAVE_INTREPID_KOKKOSCORE:BOOL=ON \
 -D Kokkos_ENABLE_CUDA:BOOL=OFF \
 -D Kokkos_ENABLE_CXX11:BOOL=ON \
 -D Kokkos_ENABLE_Cuda_UVM=OFF \
 -D Kokkos_ENABLE_EXAMPLES:BOOL=OFF \
 -D Kokkos_ENABLE_OpenMP:BOOL=OFF \
 -D Kokkos_ENABLE_Pthread:BOOL=OFF \
 -D Kokkos_ENABLE_Serial:BOOL=ON \
 -D Kokkos_ENABLE_TESTS:BOOL=OFF \
 -D TPL_ENABLE_CUDA:STRING=OFF \
\
 -D Amesos2_ENABLE_KLU2:BOOL=ON \
 -D EpetraExt_USING_HDF5:BOOL=OFF \
 -D Intrepid_ENABLE_TESTS:BOOL=OFF \
 -D Phalanx_INDEX_SIZE_TYPE:STRING="INT" \
 -D Phalanx_KOKKOS_DEVICE_TYPE:STRING="SERIAL" \
 -D Phalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF \
 -D Sacado_ENABLE_COMPLEX:BOOL=ON \
 -D Teuchos_ENABLE_COMPLEX:BOOL=ON \
 -D Tpetra_ENABLE_CLASSIC_VBR:BOOL=ON \
 -D Tpetra_ENABLE_Kokkos_Refactor:BOOL=ON \
\
 -D TPL_ENABLE_HDF5:BOOL=OFF \
 -D TPL_ENABLE_HWLOC:STRING=OFF \
 -D TPL_ENABLE_Matio:BOOL=OFF \
 -D TPL_ENABLE_Netcdf:BOOL=ON \
 -D TPL_ENABLE_OpenMP:BOOL=OFF \
 -D TPL_ENABLE_X11:BOOL=OFF \
 -D TPL_Netcdf_INCLUDE_DIRS:PATH=/usr/include \
\
 -D Trilinos_ENABLE_Amesos2:BOOL=ON \
 -D Trilinos_ENABLE_Amesos:BOOL=ON \
 -D Trilinos_ENABLE_Anasazi:BOOL=ON \
 -D Trilinos_ENABLE_AztecOO:BOOL=ON \
 -D Trilinos_ENABLE_Belos:BOOL=ON \
 -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
 -D Trilinos_ENABLE_Epetra:BOOL=ON \
 -D Trilinos_ENABLE_EpetraExt:BOOL=ON \
 -D Trilinos_ENABLE_Ifpack2:BOOL=ON \
 -D Trilinos_ENABLE_Ifpack:BOOL=ON \
 -D Trilinos_ENABLE_Intrepid:BOOL=ON \
 -D Trilinos_ENABLE_Kokkos:BOOL=ON \
 -D Trilinos_ENABLE_KokkosAlgorithms:BOOL=ON \
 -D Trilinos_ENABLE_KokkosContainers:BOOL=ON \
 -D Trilinos_ENABLE_KokkosCore:BOOL=ON \
 -D Trilinos_ENABLE_KokkosExample:BOOL=OFF \
 -D Trilinos_ENABLE_ML:BOOL=ON \
 -D Trilinos_ENABLE_MOOCHO:BOOL=OFF \
 -D Trilinos_ENABLE_OpenMP:BOOL=OFF \
 -D Trilinos_ENABLE_MueLu:BOOL=ON \
 -D Trilinos_ENABLE_NOX:BOOL=ON \
 -D Trilinos_ENABLE_Pamgen:BOOL=ON \
 -D Trilinos_ENABLE_Phalanx:BOOL=ON \
 -D Trilinos_ENABLE_Piro:BOOL=ON \
 -D Trilinos_ENABLE_Rythmos:BOOL=ON \
 -D Trilinos_ENABLE_SEACAS:BOOL=ON \
 -D Trilinos_ENABLE_STK:BOOL=ON \
 -D Trilinos_ENABLE_STKClassic:BOOL=OFF \
 -D Trilinos_ENABLE_STKDoc_tests:BOOL=OFF \
 -D Trilinos_ENABLE_STKIO:BOOL=ON \
 -D Trilinos_ENABLE_STKMesh:BOOL=ON \
 -D Trilinos_ENABLE_STKSearch:BOOL=OFF \
 -D Trilinos_ENABLE_STKSearchUtil:BOOL=OFF \
 -D Trilinos_ENABLE_STKTopology:BOOL=ON \
 -D Trilinos_ENABLE_STKTransfer:BOOL=ON \
 -D Trilinos_ENABLE_STKUnit_tests:BOOL=OFF \
 -D Trilinos_ENABLE_STKUtil:BOOL=ON \
 -D Trilinos_ENABLE_Sacado:BOOL=ON \
 -D Trilinos_ENABLE_Shards:BOOL=ON \
 -D Trilinos_ENABLE_Stokhos:BOOL=ON \
 -D Trilinos_ENABLE_Stratimikos:BOOL=ON \
 -D Trilinos_ENABLE_TESTS:BOOL=OFF \
 -D Trilinos_ENABLE_Teko:BOOL=ON \
 -D Trilinos_ENABLE_Teuchos:BOOL=ON \
 -D Trilinos_ENABLE_ThreadPool:BOOL=ON \
 -D Trilinos_ENABLE_Thyra:BOOL=ON \
 -D Trilinos_ENABLE_Tpetra:BOOL=ON \
 -D Trilinos_ENABLE_Zoltan2:BOOL=ON \
 -D Trilinos_ENABLE_Zoltan:BOOL=ON \
 package_dir
