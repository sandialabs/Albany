
module swap  cmake/3.14.4

rm -fr CMake*

ALBANYINSTALLDIR="/global/homes/k/knliege/shared/albany"
TRILINSTALLDIR="/global/homes/k/knliege/shared/trilinos"
TRILBUILDDIR="/global/homes/k/knliege/dev/TrilinosB"
TRILSOURCEDIR="/global/homes/k/knliege/dev/Trilinos"

PYTRI_ANACONDA_HOME="/global/homes/k/knliege/.conda/envs/my_pytrilinos_env"

BUILD_DIR=${PWD}

cmake \
      -D ALBANY_TRILINOS_DIR:FILEPATH="$TRILINSTALLDIR" \
      -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
      -D ENABLE_FAD_TYPE:STRING=SFad \
      -D ALBANY_SFAD_SIZE=12 \
      -D ENABLE_LANDICE:BOOL=ON \
      -D INSTALL_ALBANY:BOOL=ON \
      -D CMAKE_INSTALL_PREFIX:PATH=$ALBANYINSTALLDIR \
      -D ENABLE_DEMO_PDES:BOOL=OFF \
      -D ENABLE_UNIT_TESTS:BOOL=OFF \
      -D ENABLE_MPAS_INTERFACE:BOOL=ON \
      -D ENABLE_MESH_DEPENDS_ON_PARAMETERS:BOOL=OFF \
      -D Albany_BUILD_STATIC_EXE:BOOL=OFF \
      -D BUILD_SHARED_LIBS:BOOL=ON \
      -D CMAKE_EXE_LINKER_FLAGS:STRING="-Wl,-zmuldefs" \
\
      -D ENABLE_ALBANY_PYTHON:BOOL=ON \
      -D SWIG_EXECUTABLE:FILEPATH="/usr/bin/swig" \
      -D PYTHON_LIBRARY=$PYTRI_ANACONDA_HOME/lib/libpython3.8.so \
      -D PYTHON_INCLUDE_DIR=$PYTRI_ANACONDA_HOME/include/python3.8/ \
      -D PYTHON_EXECUTABLE=$PYTRI_ANACONDA_HOME/bin/python \
      -D TRILINOS_SOURCE_DIR=$TRILSOURCEDIR \
      -D TRILINOS_BUILD_DIR=$TRILBUILDDIR \
       ../Albany
