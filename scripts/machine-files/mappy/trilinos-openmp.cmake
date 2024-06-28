# The user may have a symlink to this file somewhere, so make sure
# we resolve links first, so that relative paths are in terms of
# the *real* file
get_filename_component(THIS_FILE ${CMAKE_CURRENT_LIST_FILE} REALPATH)
get_filename_component(THIS_PATH ${THIS_FILE} DIRECTORY)

# Get common settings
include(${THIS_PATH}/trilinos_common.cmake)

# Set Kokkos device
include(${THIS_PATH}/../kokkos/device/openmp.cmake)

# Enable OpenMP in trilinos
set (Trilinos_ENABLE_OpenMP ON CACHE BOOL "")
