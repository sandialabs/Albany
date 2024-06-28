# Notice that this script does not assume anything on the
# trilinos installation, so can be used for CPU and GPU builds.

# Set compilers
set (CMAKE_CXX_COMPILER mpicxx CACHE STRING "")
set (CMAKE_C_COMPILER mpicc CACHE STRING "")
set (CMAKE_FORTRAN_COMPILER mpifort CACHE STRING "")

# Physics packages
option (ENABLE_LANDICE "Flag to turn on LandIce Source code" ON)
option (ENABLE_DEMO_PDES "Flag to turn on demonstration PDEs problems" ON)
option (ENABLE_MPAS_INTERFACE "Flag to turn on LandIce Source code" ON)
option (ENABLE_CISM_INTERFACE "Flag to turn on LandIce Interface to the CISM code" ON)

# Fad options
set (ENABLE_FAD_TYPE "SFad" CACHE STRING "Sacado forward mode automatic differentiation data type")
set (ALBANY_SFAD_SIZE 32 CACHE STRING "Number of derivative components chosen at compile-time for AD")
set (ENABLE_TAN_FAD_TYPE "SFad" CACHE STRING "Sacado forward mode automatic differentiation data type for Tangent")
set (ALBANY_TAN_SFAD_SIZE 32 CACHE STRING "Number of derivative components chosen at compile-time for Tangent AD")
set (ENABLE_HES_VEC_FAD_TYPE "SFad" CACHE STRING "Sacado forward mode automatic differentiation data type for Hessian-vector product")
set (ALBANY_HES_VEC_SLFAD_SIZE 32 CACHE STRING "Maximum number of derivative components chosen at compile-time for Hessian-vector AD")

# Misc options
option (ENABLE_MESH_DEPENDS_ON_PARAMETERS "Flag to turn on dependency of mesh on parameters, e.g for shape optimization" OFF)
option (ENABLE_OMEGAH "Flag to turn on the Omega_h mesh interface" ON)
