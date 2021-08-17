module swap  cmake/3.18.2

TRILINSTALLDIR="/global/homes/k/knliege/shared/trilinos"
PYTRI_ANACONDA_HOME="/global/homes/k/knliege/.conda/envs/my_pytrilinos_env"
BOOST_DIR="/global/cfs/cdirs/e3sm/software/albany-trilinos/tpls/boost_1_73_0"

rm -fr CMake*

cmake \
      -D CMAKE_INSTALL_PREFIX:PATH=$TRILINSTALLDIR \
      -D Boost_INCLUDE_DIRS:FILEPATH="${BOOST_DIR}" \
      -D Netcdf_LIBRARY_DIRS:FILEPATH="${NETCDF_DIR}/lib" \
      -D TPL_Netcdf_INCLUDE_DIRS:PATH="${NETCDF_DIR}/include" \
      -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
      -D CMAKE_BUILD_TYPE:STRING=RELEASE \
      -D Trilinos_WARNINGS_AS_ERRORS_FLAGS:STRING="" \
      -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
      -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
\
      -D Trilinos_ENABLE_Fortran:BOOL=ON \
\
      -D TPL_ENABLE_SuperLU:BOOL=OFF \
      -D Amesos2_ENABLE_KLU2:BOOL=ON \
\
      -D Trilinos_ASSERT_MISSING_PACKAGES=OFF \
      -D Trilinos_ENABLE_Teuchos:BOOL=ON \
      -D HAVE_TEUCHOS_COMM_TIMERS=ON \
      -D Trilinos_ENABLE_Kokkos:BOOL=ON \
      -D Trilinos_ENABLE_KokkosCore:BOOL=ON \
      -D Trilinos_ENABLE_Zoltan:BOOL=ON \
      -D Trilinos_ENABLE_Zoltan2:BOOL=ON\
      -D Trilinos_ENABLE_Sacado:BOOL=ON \
      -D Trilinos_ENABLE_Intrepid2:BOOL=ON \
      -D Trilinos_ENABLE_Epetra:BOOL=ON \
      -D Trilinos_ENABLE_Tpetra:BOOL=ON \
      -D Trilinos_ENABLE_EpetraExt:BOOL=ON \
      -D Trilinos_ENABLE_Ifpack:BOOL=ON \
      -D Trilinos_ENABLE_Ifpack2:BOOL=ON \
      -D Trilinos_ENABLE_AztecOO:BOOL=ON \
      -D Trilinos_ENABLE_Amesos:BOOL=ON \
      -D Trilinos_ENABLE_Amesos2:BOOL=ON \
      -D Trilinos_ENABLE_Belos:BOOL=ON \
      -D Trilinos_ENABLE_Phalanx:BOOL=ON \
      -D Trilinos_ENABLE_ROL:BOOL=ON \
      -D Trilinos_ENABLE_Tempus:BOOL=ON \
      -D Trilinos_ENABLE_ML:BOOL=ON \
      -D Trilinos_ENABLE_MueLu:BOOL=ON \
      -D Trilinos_ENABLE_NOX:BOOL=ON \
      -D Trilinos_ENABLE_Stratimikos:BOOL=ON \
      -D Trilinos_ENABLE_Thyra:BOOL=ON \
      -D Trilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON \
      -D Trilinos_ENABLE_Piro:BOOL=ON \
      -D Trilinos_ENABLE_STKIO:BOOL=ON \
      -D Trilinos_ENABLE_STKExprEval:BOOL=ON \
      -D Trilinos_ENABLE_STKMesh:BOOL=ON \
      -D Trilinos_ENABLE_SEACASExodus:BOOL=ON \
      -D Trilinos_ENABLE_SEACASIoss:BOOL=ON \
      -D Trilinos_ENABLE_SEACASEpu:BOOL=ON \
      -D Trilinos_ENABLE_SEACASNemspread:BOOL=ON \
      -D Trilinos_ENABLE_SEACASNemslice:BOOL=ON \
      -D Trilinos_ENABLE_Pamgen:BOOL=ON \
      -D Trilinos_ENABLE_Teko:BOOL=ON \
      -D Trilinos_ENABLE_PanzerDofMgr:BOOL=ON \
\
      -D Trilinos_ENABLE_TESTS:BOOL=OFF \
      -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
\
      -D TPL_FIND_SHARED_LIBS:BOOL=ON \
      -D BUILD_SHARED_LIBS:BOOL=ON \
\
      -D Kokkos_ENABLE_SERIAL:BOOL=ON \
      -D Kokkos_ENABLE_OPENMP:BOOL=OFF \
      -D Kokkos_ENABLE_PTHREAD:BOOL=OFF \
      -D Zoltan_ENABLE_ULONG_IDS:BOOL=ON \
      -D ZOLTAN_BUILD_ZFDRIVE:BOOL=OFF \
      -D Phalanx_KOKKOS_DEVICE_TYPE:STRING="SERIAL" \
      -D Phalanx_INDEX_SIZE_TYPE:STRING="INT" \
      -D Phalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF \
\
      -D Boost_LIBRARY_DIRS:FILEPATH="${BOOST_DIR}/libs" \
\
      -D TPL_ENABLE_Netcdf:BOOL=ON \
\
      -D TPL_ENABLE_BLAS:BOOL=ON \
      -D TPL_BLAS_LIBRARIES:STRING="${MKLROOT}/lib/intel64/libmkl_core.so" \
      -D TPL_ENABLE_LAPACK:BOOL=ON \
      -D TPL_LAPACK_LIBRARIES:STRING="${MKLROOT}/lib/intel64/libmkl_core.so" \
      -D TPL_ENABLE_GLM:BOOL=OFF \
      -D TPL_ENABLE_Matio:BOOL=OFF \
\
      -D TPL_ENABLE_MPI:BOOL=ON \
      -D TPL_ENABLE_Boost:BOOL=ON \
\
      -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
      -D Trilinos_VERBOSE_CONFIGURE:BOOL=OFF \
\
      -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
      -D Tpetra_INST_INT_LONG_LONG:BOOL=ON \
      -D Tpetra_INST_INT_LONG:BOOL=OFF \
      -D Tpetra_INST_INT_INT:BOOL=OFF \
      -D Tpetra_INST_DOUBLE:BOOL=ON \
      -D Tpetra_INST_FLOAT:BOOL=OFF \
      -D Tpetra_INST_COMPLEX_FLOAT:BOOL=OFF \
      -D Tpetra_INST_COMPLEX_DOUBLE:BOOL=OFF \
      -D Tpetra_INST_INT_UNSIGNED:BOOL=OFF \
      -D Tpetra_INST_INT_UNSIGNED_LONG:BOOL=OFF \
\
      -D MPI_USE_COMPILER_WRAPPERS:BOOL=OFF \
      -D CMAKE_CXX_COMPILER:FILEPATH="CC" \
      -D CMAKE_C_COMPILER:FILEPATH="cc" \
      -D CMAKE_Fortran_COMPILER:FILEPATH="ftn" \
      -D Trilinos_ENABLE_Fortran=ON \
      -D CMAKE_C_FLAGS:STRING="-target-cpu=haswell -mkl -O3 -DREDUCE_SCATTER_BUG" \
      -D CMAKE_CXX_FLAGS:STRING="-target-cpu=haswell -mkl -O3 -DREDUCE_SCATTER_BUG -DBOOST_NO_HASH" \
      -D CMAKE_Fortran_FLAGS:STRING="-mkl" \
      -D CMAKE_EXE_LINKER_FLAGS="-mkl -ldl" \
      -D Trilinos_ENABLE_SHADOW_WARNINGS=OFF \
      -DTPL_ENABLE_Pthread:BOOL=OFF \
      -DTPL_ENABLE_BinUtils:BOOL=OFF \
\
      -D MPI_EXEC:FILEPATH=srun \
      -D MPI_EXEC_MAX_NUMPROCS:STRING=4 \
      -D MPI_EXEC_NUMPROCS_FLAG:STRING=-n \
      -D CMAKE_SKIP_INSTALL_RPATH=TRUE \
\
      -D Trilinos_ENABLE_PyTrilinos:BOOL=ON \
      -D PyTrilinos_ENABLE_TESTS:BOOL=ON \
      -D Teuchos_ENABLE_TESTS:BOOL=ON \
      -D Ifpack2_ENABLE_TESTS:BOOL=ON \
      -D PyTrilinos_DOCSTRINGS:BOOL=OFF \
      -D SWIG_EXECUTABLE:FILEPATH="/usr/bin/swig" \
      -D PYTHON_LIBRARY=$PYTRI_ANACONDA_HOME/lib/libpython3.8.so \
      -D PYTHON_INCLUDE_DIR=$PYTRI_ANACONDA_HOME/include/python3.8/ \
      -D PYTHON_EXECUTABLE=$PYTRI_ANACONDA_HOME/bin/python \
      -D MPI_BASE_DIR="${MPICH_DIR}" \
      -D Trilinos_EXTRA_LINK_FLAGS:STRING="-L/opt/cray/pe/atp/2.1.3/libApp/ -lAtpSigHandler -lAtpSigHCommData" \
\
      -D Trilinos_ENABLE_ShyLU_DDFROSch:BOOL=ON \
\
../Trilinos
