SRC_DIR=${SOFTWARE_ROOT}/albany/source
INSTALL_DIR=${SOFTWARE_ROOT}/albany/install

rm -rf CMakeFiles
rm -f  CMakeCache.txt
rm -rf tpls

cmake \
  -D CMAKE_BUILD_TYPE:STRING=RELEASE                          \
  -D CMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR}                 \
  -D CMAKE_C_COMPILER:STRING="mpicc"                          \
  -D CMAKE_CXX_COMPILER:STRING="mpicxx"                       \
  \
  -D ALBANY_TRILINOS_DIR:PATH=${TRILINOS_ROOT}                \
  -D ALBANY_MPI_EXEC_MAX_NUMPROCS:STRING=4                    \
  -D ALBANY_MPI_EXEC_TRAILING_OPTIONS:STRING="--bind-to core" \
  \
  -D ENABLE_OMEGAH:BOOL=OFF                                   \
  -D ENABLE_DEMO_PDES:BOOL=OFF                                \
  -D ENABLE_LANDICE:BOOL=ON                                   \
  -D ENABLE_MPAS_INTERFACE:BOOL=ON                            \
  -D ENABLE_MESH_DEPENDS_ON_PARAMETERS:BOOL=OFF               \
  \
  ${SRC_DIR}
