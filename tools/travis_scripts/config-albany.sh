INSTALL_DIR=../albany-install
TRILINOS_ROOT=${HOME}/trilinos/trilinos-install

rm -rf CMakeFiles
rm -f  CMakeCache.txt

cmake \
  -D CMAKE_BUILD_TYPE:STRING=RELEASE                          \
  -D CMAKE_CXX_FLAGS:STRING="-w"                              \
  -D CMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR}                 \
  -D INSTALL_ALBANY:BOOL=ON                                   \
  -D ALBANY_TRILINOS_DIR:PATH=${TRILINOS_ROOT}                \
  -D ENABLE_AERAS:BOOL=ON                                     \
  -D ENABLE_AMP:BOOL=ON                                       \
  -D ENABLE_ATO:BOOL=ON                                       \
  -D ENABLE_LANDICE:BOOL=ON                                   \
  -D ENABLE_LCM:BOOL=ON                                       \
  -D ENABLE_DEMO_PDES:BOOL=ON                                 \
  -D ENABLE_SCOREC:BOOL=ON                                    \
  -D ENABLE_TSUNAMI:BOOL=ON                                   \
  -D ENABLE_MESH_DEPENDS_ON_PARAMETERS:BOOL=OFF               \
  -D ENABLE_MESH_DEPENDS_ON_SOLUTION:BOOL=OFF                 \
  -D ENABLE_PARAMETERS_DEPEND_ON_SOLUTION:BOOL=OFF            \
  -D USE_NEW_POLICY_CMP0060:BOOL=ON                           \
  -D ALBANY_MPI_EXEC_MAX_NUMPROCS:STRING=4                    \
  -D ALBANY_MPI_EXEC_LEADING_OPTIONS:STRING='--bind-to core'  \
  ../albany-src
