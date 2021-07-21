#!/bin/bash

# WARNING: This file is generated automatically. Any changes made here
# will be lost when the package is configured again.  Any permament
# changes should go into the corresponding template at the top level
# LCM directory.

if [ -f ./CMakeCache.txt ]; then
    rm ./CMakeCache.txt
fi

if [ -d ./CMakeFiles ]; then
    rm ./CMakeFiles -rf
fi

export OMPI_CC=lcm_ompi_cc
export OMPI_CXX=lcm_ompi_cxx
export OMPI_FC=lcm_ompi_fc

#
# The CMake command.
#
cmake \
 -D BUILD_SHARED_LIBS:BOOL=ON \
 -D CMAKE_BUILD_TYPE:STRING="ali_build_type" \
 -D CMAKE_CXX_COMPILER:FILEPATH="$MPI_BIN/mpicxx" \
 -D CMAKE_C_COMPILER:FILEPATH="$MPI_BIN/mpicc" \
 -D CMAKE_Fortran_COMPILER:FILEPATH="$MPI_BIN/mpif90" \
 -D CMAKE_INSTALL_PREFIX:PATH=lcm_install_dir \
 -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
\
 -D TPL_ENABLE_MPI:BOOL=ON \
 -D TPL_ENABLE_BinUtils:BOOL=OFF \
 -D TPL_MPI_INCLUDE_DIRS:STRING="$MPI_INC" \
 -D TPL_MPI_LIBRARY_DIRS:STRING="$MPI_LIB" \
 -D MPI_BIN_DIR:PATH="$MPI_BIN" \
\
 -D TPL_ENABLE_Boost:BOOL=ON \
 -D TPL_ENABLE_BoostLib:BOOL=ON \
 -D Boost_INCLUDE_DIRS:STRING="$BOOST_INC" \
 -D Boost_LIBRARY_DIRS:STRING="$BOOST_LIB" \
 -D BoostLib_INCLUDE_DIRS:STRING="$BOOSTLIB_INC" \
 -D BoostLib_LIBRARY_DIRS:STRING="$BOOSTLIB_LIB" \
\
 -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
 -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
 -D Trilinos_ENABLE_CXX11:BOOL=ON \
 -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
 -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
 -D Trilinos_VERBOSE_CONFIGURE:BOOL=OFF \
 -D Trilinos_WARNINGS_AS_ERRORS_FLAGS:STRING="" \
 \
 -D Teuchos_ENABLE_STACKTRACE:BOOL=OFF \
 -D Teuchos_ENABLE_DEFAULT_STACKTRACE:BOOL=OFF \
 -D Kokkos_ENABLE_CUDA_UVM:BOOL=lcm_enable_uvm \
 -D Kokkos_ENABLE_EXAMPLES:BOOL=lcm_enable_kokkos_examples \
 -D Kokkos_ENABLE_OPENMP:BOOL=lcm_enable_openmp \
 -D Kokkos_ENABLE_PTHREAD:BOOL=lcm_enable_pthreads \
 -D Kokkos_ENABLE_SERIAL:BOOL=ON \
 -D Kokkos_ENABLE_TESTS:BOOL=OFF \
 -D TPL_ENABLE_CUDA:STRING=lcm_enable_cuda \
 -D TPL_ENABLE_CUSPARSE:BOOL=lcm_enable_cusparse \
\
 -D Amesos2_ENABLE_KLU2:BOOL=ON \
 -D EpetraExt_USING_HDF5:BOOL=OFF \
 -D MiniTensor_ENABLE_TESTS:BOOL=ON \
 -D ROL_ENABLE_TESTS:BOOL=OFF \
 -D Phalanx_INDEX_SIZE_TYPE:STRING="lcm_phalanx_index_type" \
 -D Phalanx_KOKKOS_DEVICE_TYPE:STRING="lcm_kokkos_device" \
 -D Phalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF \
 -D Tpetra_ENABLE_Kokkos_Refactor:BOOL=ON \
 -D Tpetra_INST_PTHREAD:BOOL=lcm_tpetra_inst_pthread \
 -D Tpetra_ENABLE_DEPRECATED_CODE:BOOL=OFF \
 -D Xpetra_ENABLE_DEPRECATED_CODE:BOOL=OFF \
\
 -D TPL_ENABLE_HDF5:BOOL=OFF \
 -D TPL_ENABLE_HWLOC:STRING=lcm_enable_hwloc \
 -D TPL_ENABLE_Matio:BOOL=OFF \
 -D TPL_ENABLE_Netcdf:BOOL=ON \
 -D TPL_ENABLE_X11:BOOL=OFF \
 -D TPL_Netcdf_INCLUDE_DIRS:STRING="lcm_netcdf_inc" \
 -D TPL_Netcdf_LIBRARY_DIRS:STRING="lcm_netcdf_lib" \
 -D TPL_Netcdf_LIBRARIES:STRING="lcm_netcdf_lib/libnetcdf.so" \
 -D TPL_Netcdf_PARALLEL:BOOL=ON \
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
 -D Trilinos_ENABLE_Intrepid2:BOOL=ON \
 -D Trilinos_ENABLE_Kokkos:BOOL=ON \
 -D Trilinos_ENABLE_KokkosAlgorithms:BOOL=ON \
 -D Trilinos_ENABLE_KokkosContainers:BOOL=ON \
 -D Trilinos_ENABLE_KokkosCore:BOOL=ON \
 -D Trilinos_ENABLE_KokkosExample:BOOL=OFF \
 -D Trilinos_ENABLE_MiniTensor:BOOL=ON \
 -D Trilinos_ENABLE_ML:BOOL=ON \
 -D Trilinos_ENABLE_MueLu:BOOL=ON \
 -D Trilinos_ENABLE_NOX:BOOL=ON \
 -D Trilinos_ENABLE_OpenMP:BOOL=lcm_enable_openmp \
 -D Trilinos_ENABLE_Pamgen:BOOL=ON \
 -D Trilinos_ENABLE_PanzerExprEval:BOOL=ON \
 -D Trilinos_ENABLE_Phalanx:BOOL=ON \
 -D Trilinos_ENABLE_Piro:BOOL=ON \
 -D Trilinos_ENABLE_ROL:BOOL=ON \
 -D Trilinos_ENABLE_Rythmos:BOOL=ON \
 -D Trilinos_ENABLE_SEACAS:BOOL=ON \
 -D Trilinos_ENABLE_STKClassic:BOOL=OFF \
 -D Trilinos_ENABLE_STKIO:BOOL=ON \
 -D Trilinos_ENABLE_STKMesh:BOOL=ON \
 -D Trilinos_ENABLE_STKExprEval:BOOL=ON \
 -D Trilinos_ENABLE_Sacado:BOOL=ON \
 -D Trilinos_ENABLE_Shards:BOOL=ON \
 -D Trilinos_ENABLE_Stokhos:BOOL=OFF \
 -D Trilinos_ENABLE_Stratimikos:BOOL=ON \
 -D Trilinos_ENABLE_TESTS:BOOL=OFF \
 -D Trilinos_ENABLE_Teko:BOOL=ON \
 -D Trilinos_ENABLE_Tempus:BOOL=ON \
 -D Trilinos_ENABLE_Teuchos:BOOL=ON \
 -D Trilinos_ENABLE_ThreadPool:BOOL=ON \
 -D Trilinos_ENABLE_Thyra:BOOL=ON \
 -D Trilinos_ENABLE_Tpetra:BOOL=ON \
 -D Trilinos_ENABLE_Zoltan2:BOOL=ON \
 -D Trilinos_ENABLE_Zoltan:BOOL=ON \
 -D KOKKOS_ENABLE_LIBDL:BOOL=ON \
 -D Trilinos_ENABLE_PanzerDofMgr:BOOL=ON \
 -D Tpetra_ENABLE_DEPRECATED_CODE=ON \
  -D Xpetra_ENABLE_DEPRECATED_CODE=ON \
 lcm_package_dir
