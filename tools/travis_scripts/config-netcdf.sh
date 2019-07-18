INSTALL_DIR=${HOME}/netcdf-c/netcdf-c-install
HDF5_LIB_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
HDF5_INC_DIR=/usr/include/hdf5/openmpi

rm -rf CMakeCache.txt
rm -f  CMakeFiles

cmake -Wno-dev                                              \
  -D CMAKE_BUILD_TYPE:STRING=RELEASE                        \
  -D CMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR}               \
  -D CMAKE_C_COMPILER:STIRNG=mpicc                          \
  -D HDF5_C_LIBRARY:FILEPATH=${HDF5_LIB_DIR}/libhdf5.so     \
  -D HDF5_HL_LIBRARY:FILEPATH=${HDF5_LIB_DIR}/libhdf5_hl.so \
  -D HDF5_INCLUDE_DIR:PATH=${HDF5_INC_DIR}                  \
  -D ENABLE_NETCDF_4:BOOL=ON                                \
  -D ENABLE_TESTS:BOOL=OFF                                  \
  ../netcdf-c-src
