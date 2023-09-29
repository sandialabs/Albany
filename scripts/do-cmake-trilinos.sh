# This file contains Trilinos config setting that are common across all machines
# and all kokkos backends. Specific mach/backend settings are loaded from
# the cache file in the first line.
#
# NOTE: in the cache file, the user can also overwrite the settings that
# are explicitly set in this script. In other words, the cache file takes
# precedence over any setting passed via -DVAR=value
#
# This script MUST be invoked with the following env vars defined:
#
# SOURCE_DIR: the path to trilinos source dir
# INSTALL_DIR: the path where to install trilinos
# CACHE_FILE: the path to this machine/build cmake var settings
#
# The cache file can include multiple files inside, but all that these
# file can contain is a series of
#   set (VARNAME VARVALUE CACHE VARTYPE "docstring")
# calls. No other cmake command should appear in those files.
#
# These cache entries MUST be set in that file:
#  - CMAKE_<LANG>_COMPILER (STRING): path to <LANG> compiler (LANG=C,CXX,Fortran)
#
# These cache entries are optional
#  - Kokkos_ARCH_<ARCH> (BOOL): which kokkos arch to enable
#  - Kokkos_ENABLE_<DEVICE> (BOOL): kokkos backend to enable
#  - Other Kokkos_<OPTION> settings, depending on needs
#  - CMAKE_<LANG>_FLAGS (STRING): flags for compiler/linker
#  - Tpetra_ASSUME_CUDA_AWARE_MPI (BOOL)
#
# TPLs are the most annoying part of configuyring trilinos. Some advice:
#  - If your environment setup provides <PKG>_ROOT, then PKG should
#    be correctly found by Trilinos, without any need for CMake cache entries.
#  - BLAS/LAPACK are a good exception for this. I find that I need to
#    set the following cache entries
#      <LIB>_LIBRARY_DIRS
#      <LIB>_LIBRARY_NAMES
#    where LIB=BLAS,LAPACK. You may be lucky, and blas/lapack get found
#    automatically, but setting the above vars should be safe. Also,
#    lots of machines already have a system blas/lapack library, but
#    it may have been compiled with the system compilers, which may
#    or may not be compatible with the ones you use to build Trilinos
# - HDF5 may require to set TPL_HDF5_LIBRARY_DIRS
# - In general, if a tpl is not found (or not found from the location
#   you expected), you may need to set
#     <LIB>_LIBRARY_DIRS
#     <LIB>_INCLUDE_DIRS
#   or the same but with TPL_ prepended. I still don't understand
#   when/why TPL_ is needed with TriBITs; sometimes it is, sometimes
#   it's not. I will find out one day.
#
# NOTE: the following lines are organized in blocks:
# - General CMake options
# - General Trilinos options
# - Trilinos Packages enable's
# - Packages-specific options
# - TPL enable's
# - TPL include/library dirs

for var in "CACHE_FILE" "SOURCE_DIR" "INSTALL_DIR"; do
  if [[ "${!var}" == "" ]]; then
    echo "Error! ${var} env var is not set."
    exit 1
  else
    echo "${var}: ${!var}"
  fi
done

rm -rf CMakeFiles.txt
rm -f  CMakeCache.txt

cmake \
  -C ${CACHE_FILE}                                  \
  -D CMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR}       \
  -D CMAKE_BUILD_TYPE:STRING=RELEASE                \
  -D CMAKE_CXX_STANDARD=17                          \
  -D CMAKE_SKIP_RULE_DEPENDENCY=ON                  \
  \
  -D Trilinos_ASSERT_DEFINED_DEPENDENCIES:BOOL=OFF  \
  -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
  -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF          \
  -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF              \
  -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
  -D Trilinos_VERBOSE_CONFIGURE:BOOL=OFF            \
  -D Trilinos_WARNINGS_AS_ERRORS_FLAGS:STRING=''    \
  -D Trilinos_ENABLE_TESTS:BOOL=OFF                 \
  \
  -D Trilinos_ENABLE_Amesos2:BOOL=ON                \
  -D Trilinos_ENABLE_Amesos:BOOL=ON                 \
  -D Trilinos_ENABLE_Anasazi:BOOL=ON                \
  -D Trilinos_ENABLE_AztecOO:BOOL=ON                \
  -D Trilinos_ENABLE_Belos:BOOL=ON                  \
  -D Trilinos_ENABLE_Epetra:BOOL=ON                 \
  -D Trilinos_ENABLE_EpetraExt:BOOL=ON              \
  -D Trilinos_ENABLE_Ifpack2:BOOL=ON                \
  -D Trilinos_ENABLE_Ifpack:BOOL=ON                 \
  -D Trilinos_ENABLE_Intrepid2:BOOL=ON              \
  -D Trilinos_ENABLE_Kokkos:BOOL=ON                 \
  -D Trilinos_ENABLE_ML:BOOL=ON                     \
  -D Trilinos_ENABLE_MueLu:BOOL=ON                  \
  -D Trilinos_ENABLE_NOX:BOOL=ON                    \
  -D Trilinos_ENABLE_Pamgen:BOOL=ON                 \
  -D Trilinos_ENABLE_PanzerDofMgr:BOOL=ON           \
  -D Trilinos_ENABLE_PanzerExprEval:BOOL=ON         \
  -D Trilinos_ENABLE_Phalanx:BOOL=ON                \
  -D Trilinos_ENABLE_Piro:BOOL=ON                   \
  -D Trilinos_ENABLE_ROL:BOOL=OFF                   \
  -D Trilinos_ENABLE_SEACAS:BOOL=ON                 \
  -D Trilinos_ENABLE_SEACASAprepro_lib:BOOL=OFF     \
  -D Trilinos_ENABLE_STKDoc_tests:BOOL=OFF          \
  -D Trilinos_ENABLE_STKExprEval:BOOL=ON            \
  -D Trilinos_ENABLE_STKIO:BOOL=ON                  \
  -D Trilinos_ENABLE_STKMesh:BOOL=ON                \
  -D Trilinos_ENABLE_Sacado:BOOL=ON                 \
  -D Trilinos_ENABLE_Shards:BOOL=ON                 \
  -D Trilinos_ENABLE_ShyLU_DDFROSch:BOOL=ON         \
  -D Trilinos_ENABLE_Stokhos:BOOL=OFF               \
  -D Trilinos_ENABLE_Stratimikos:BOOL=ON            \
  -D Trilinos_ENABLE_Teko:BOOL=ON                   \
  -D Trilinos_ENABLE_Tempus:BOOL=ON                 \
  -D Trilinos_ENABLE_Teuchos:BOOL=ON                \
  -D Trilinos_ENABLE_Thyra:BOOL=ON                  \
  -D Trilinos_ENABLE_ThyraEpetraAdapters:BOOL=ON    \
  -D Trilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON    \
  -D Trilinos_ENABLE_Tpetra:BOOL=ON                 \
  -D Trilinos_ENABLE_Zoltan2:BOOL=ON                \
  -D Trilinos_ENABLE_Zoltan:BOOL=ON                 \
  \
  -D Amesos2_ENABLE_KLU2:BOOL=ON                    \
  -D Belos_ENABLE_Epetra=OFF                        \
  -D Kokkos_ENABLE_EXAMPLES:BOOL=OFF                \
  -D Kokkos_ENABLE_TESTS:BOOL=OFF                   \
  -D MueLu_ENABLE_Epetra=OFF                        \
  -D MueLu_ENABLE_Tutorial:BOOL=OFF                 \
  -D Phalanx_INDEX_SIZE_TYPE:STRING=UINT            \
  -D Sacado_ENABLE_COMPLEX:BOOL=OFF                 \
  -D Tempus_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON   \
  -D Teuchos_ENABLE_COMPLEX:BOOL=OFF                \
  -D Tpetra_ENABLE_DEPRECATED_CODE:BOOL=OFF         \
  -D Tpetra_INST_INT_INT:BOOL=OFF                   \
  -D Tpetra_INST_INT_LONG_LONG:BOOL=ON              \
  -D Xpetra_ENABLE_DEPRECATED_CODE:BOOL=OFF         \
  -D Xpetra_ENABLE_Epetra=OFF                       \
  -D Xpetra_ENABLE_EpetraExt=OFF                    \
  \
  -D TPL_ENABLE_BLAS:BOOL=ON                        \
  -D TPL_ENABLE_Boost:BOOL=ON                       \
  -D TPL_ENABLE_BoostLib:BOOL=ON                    \
  -D TPL_ENABLE_HDF5:STRING=ON                      \
  -D TPL_ENABLE_LAPACK:BOOL=ON                      \
  -D TPL_ENABLE_MPI:BOOL=ON                         \
  -D TPL_ENABLE_Matio:BOOL=OFF                      \
  -D TPL_ENABLE_Netcdf:BOOL=ON                      \
  -D TPL_ENABLE_Pnetcdf:STRING=ON                   \
  -D TPL_ENABLE_X11:BOOL=OFF                        \
  \
  -S ${SOURCE_DIR}
