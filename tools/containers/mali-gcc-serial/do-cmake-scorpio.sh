SRC_DIR=${SOFTWARE_ROOT}/scorpio/source
INSTALL_DIR=${SOFTWARE_ROOT}/scorpio/install

rm -rf CMake*

cmake -Wno-dev \
 -D CMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR}   \
 -D CMAKE_C_COMPILER:STRING=mpicc              \
 -D CMAKE_CXX_COMPILER:STRING=mpicxx           \
 -D CMAKE_Fortran_COMPILER:STRING=mpif90       \
 \
 -D NetCDF_C_PATHS:PATH=${NETCDF_C_ROOT}       \
 -D NetCDF_Fortran_PATHS:PATH=${NETCDF_F_ROOT} \
 -D PnetCDF_C_PATHS:PATH=${PNETCDF_ROOT}       \
 -D WITH_HDF5:BOOL=ON                          \
 -D PIO_ENABLE_TIMING:BOOL=OFF                 \
 -D PIO_ENABLE_TESTS:BOOL=OFF                  \
 \
 -S ${SRC_DIR}
