INSTALL_DIR=../trilinos-install

HDF5_LIB_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
HDF5_INC_DIR=/usr/include/hdf5/openmpi
NETCDF_ROOT=${HOME}/netcdf-c/netcdf-c-install

cmake -Wno-dev                                                    \
  -D CMAKE_BUILD_TYPE:STRING=RELEASE                              \
  -D CMAKE_INSTALL_PREFIX:STRING=${INSTALL_DIR}                   \
  -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF                              \
  -D CMAKE_C_COMPILER:STRING="mpicc"                              \
  -D CMAKE_CXX_COMPILER:STRING="mpicxx"                           \
  -D CMAKE_Fortran_COMPILER:STRING="mpif90"                       \
  -D CMAKE_CXX_FLAGS:STRING="-Wno-terminate -Wno-literal-suffix"  \
  -D BUILD_SHARED_LIBS:BOOL=ON                                    \
  \
  -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF                        \
  -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF               \
  -D Trilinos_ENABLE_Amesos:BOOL=ON                               \
  -D Trilinos_ENABLE_Amesos2:BOOL=ON                              \
  -D Trilinos_ENABLE_Anasazi:BOOL=ON                              \
  -D Trilinos_ENABLE_AztecOO:BOOL=ON                              \
  -D Trilinos_ENABLE_Belos:BOOL=ON                                \
  -D Trilinos_ENABLE_Epetra:BOOL=ON                               \
  -D Trilinos_ENABLE_EpetraExt:BOOL=ON                            \
  -D Trilinos_ENABLE_Ifpack:BOOL=ON                               \
  -D Trilinos_ENABLE_Ifpack2:BOOL=ON                              \
  -D Trilinos_ENABLE_Intrepid:BOOL=ON                             \
  -D Trilinos_ENABLE_Intrepid2:BOOL=ON                            \
  -D Trilinos_ENABLE_Isorropia:BOOL=ON                            \
  -D Trilinos_ENABLE_Kokkos:BOOL=ON                               \
  -D Trilinos_ENABLE_KokkosKernels:BOOL=ON                        \
  -D Trilinos_ENABLE_MiniTensor:BOOL=ON                           \
  -D Trilinos_ENABLE_ML:BOOL=ON                                   \
  -D Trilinos_ENABLE_MueLu:BOOL=ON                                \
  -D Trilinos_ENABLE_NOX:BOOL=ON                                  \
  -D Trilinos_ENABLE_Pamgen:BOOL=ON                               \
  -D Trilinos_ENABLE_PanzerExprEval:BOOL=ON                       \
  -D Trilinos_ENABLE_Phalanx:BOOL=ON                              \
  -D Trilinos_ENABLE_Piro:BOOL=ON                                 \
  -D Trilinos_ENABLE_Rythmos:BOOL=ON                              \
  -D Trilinos_ENABLE_ROL:BOOL=ON                                  \
  -D Trilinos_ENABLE_Sacado:BOOL=ON                               \
  -D Trilinos_ENABLE_SCOREC:BOOL=ON                               \
  -D Trilinos_ENABLE_SEACAS:BOOL=ON                               \
  -D Trilinos_ENABLE_SEACASExodus:BOOL=ON                         \
  -D Trilinos_ENABLE_SEACASIoss:BOOL=ON                           \
  -D Trilinos_ENABLE_Shards:BOOL=ON                               \
  -D Trilinos_ENABLE_Stokhos:BOOL=OFF                             \
  -D Trilinos_ENABLE_STK:BOOL=ON                                  \
  -D Trilinos_ENABLE_STKDoc_tests:BOOL=OFF                        \
  -D Trilinos_ENABLE_STKExp:BOOL=OFF                              \
  -D Trilinos_ENABLE_STKIO:BOOL=ON                                \
  -D Trilinos_ENABLE_STKMesh:BOOL=ON                              \
  -D Trilinos_ENABLE_STKSearch:BOOK=ON                            \
  -D Trilinos_ENABLE_STKSearchUtil:BOOL=OFF                       \
  -D Trilinos_ENABLE_STKTopology:BOOL=ON                          \
  -D Trilinos_ENABLE_STKTransfer:BOOL=ON                          \
  -D Trilinos_ENABLE_STKUnit_tests:BOOL=OFF                       \
  -D Trilinos_ENABLE_STKUtil:BOOL=ON                              \
  -D Trilinos_ENABLE_Stratimikos:BOOL=ON                          \
  -D Trilinos_ENABLE_Teko:BOOL=ON                                 \
  -D Trilinos_ENABLE_Tempus:BOOL=ON                               \
  -D Trilinos_ENABLE_Teuchos:BOOL=ON                              \
  -D Trilinos_ENABLE_TeuchosParser:BOOL=ON                        \
  -D Trilinos_ENABLE_Thyra:BOOL=ON                                \
  -D Trilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON                  \
  -D Trilinos_ENABLE_ThyraEpetraAdapters:BOOL=ON                  \
  -D Trilinos_ENABLE_Tpetra:BOOL=ON                               \
  -D Trilinos_ENABLE_Zoltan:BOOL=ON                               \
  -D Trilinos_ENABLE_Zoltan2:BOOL=ON                              \
  \
  -D Trilinos_VERBOSE_CONFIGURE:BOOL=OFF                          \
  -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF                            \
  -D Trilinos_ENABLE_TESTS:BOOL=OFF                               \
  -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON               \
  \
  -D Amesos2_ENABLE_KLU2:BOOL=ON                                  \
  -D KokkosKernels_ENABLE_Experimental:BOOL=ON                    \
  -D Phalanx_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON                  \
  -D Phalanx_ALLOW_MULTIPLE_EVALUATORS_FOR_SAME_FIELD:BOOL=OFF    \
  -D Piro_ENABLE_TESTS:BOOL=ON                                    \
  -D Stratimikos_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON              \
  -D Tpetra_INST_SERIAL:BOOL=ON                                   \
  -D TeuchosCore_ENABLE_yaml-cpp:BOOL=ON                          \
  -D Zoltan_ENABLE_ULLONG_IDS:BOOL=ON                             \
  \
  -D TPL_ENABLE_MPI:BOOL=ON                                       \
  -D TPL_ENABLE_HWLOC:BOOL=OFF                                    \
  -D TPL_ENABLE_Matio:BOOL=OFF                                    \
  -D TPL_ENABLE_HDF5:BOOL=ON                                      \
  -D TPL_ENABLE_Netcdf:BOOL=ON                                    \
  -D TPL_Netcdf_PARALLEL:BOOL=ON                                  \
  \
  -D HDF5_INCLUDE_DIRS:PATH=${HDF5_INC_DIR}                       \
  -D HDF5_LIBRARY_DIRS:PATH=${HDF5_LIB_DIR}                       \
  -D Netcdf_INCLUDE_DIRS:PATH=${NETCDF_ROOT}/include              \
  -D Netcdf_LIBRARY_DIRS:PATH=${NETCDF_ROOT}/lib                  \
  \
  ../trilinos-src
