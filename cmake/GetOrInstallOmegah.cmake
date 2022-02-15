set (tmpStr "Looking for valid Omega_h installation ...")
message (STATUS ${tmpStr})

set (CMAKE_FIND_DEBUG_MODE ON)
find_package(Omega_h 10.8 CONFIG QUIET
  # Avoid all defaults. Only check env/CMake var Omega_h_ROOT
  NO_CMAKE_PATH
  NO_CMAKE_ENVIRONMENT_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  NO_CMAKE_PACKAGE_REGISTRY
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_INSTALL_PREFIX
  NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)
set (CMAKE_FIND_DEBUG_MODE OFF)

# If we're building shared libs in Albany, we need Omega_h to be built with shared libs
# TODO: if CMake adds a "DYNAMYC=<value>" to find_package (to specify what variant
# of libs we want), then we can use it, and remove (some of) this snippet
if (Omega_h_FOUND)
  message (STATUS "${tmpStr} Found.")
  message ("  -- Omega_h_DIR: ${Omega_h_DIR}")
  message ("  -- Omega_h_VERSION: ${Omega_h_VERSION}")
  get_target_property(Omega_h_LIBTYPE Omega_h::omega_h TYPE)
  if (BUILD_SHARED_LIBS AND NOT Omega_h_LIBTYPE STREQUAL SHARED_LIBRARY)
    message ("     -> Wrong lib variant: ${Omega_h_LIBTYPE} instead of SHARED_LIBRARY.")
    message ("        Please, point to a compatible Omega_h installation (or none at all to force a new install)")
    message (FATAL_ERROR "Aborting...")
  else()
    foreach (opt IN ITEMS Omega_h_USE_Kokkos Omega_h_USE_MPI)
      if (NOT ${opt})
        message ("     -> Wrong configuration: ${opt} is OFF (should be ON)")
        message ("        Please, point to a compatible Omega_h installation (or none at all to force local install)")
        message (FATAL_ERROR "Aborting...")
      endif()
    endforeach()
  endif()
else ()
  message (STATUS "${tmpStr} NOT Found.")
  message (STATUS "  -> Downloading and building locally in ${CMAKE_BINARY_DIR}/tpls/omega_h")

  include (FetchContent)

  # Fetch and populate the external project
  set (FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/tpls/omega_h)

  FetchContent_Declare (
    Omega_h
    GIT_REPOSITORY git@github.com:SCOREC/omega_h
    GIT_TAG        origin/master
    OVERRIDE_FIND_PACKAGE
  )

  # Set options for Omega_h before adding the subdirectory
  get_target_property(Kokkos_INCLUDE_DIR Kokkos::kokkos INTERFACE_INCLUDE_DIRECTORIES)
  string(REPLACE "include/kokkos" "" Kokkos_INSTALL_DIR ${Kokkos_INCLUDE_DIR})
  set(Kokkos_PREFIX ${Kokkos_INSTALL_DIR} PATH "Path to Kokkos install")

  option (Omega_h_USE_Kokkos "Use Kokkos as a backend" ON)
  option (Omega_h_USE_MPI "Use MPI for parallelism" ON)
  set (MPIEXEC_EXECUTABLE ${Albany_CXX_COMPILER})
  if (Kokkos_ENABLE_CUDA_UVM)
    option (Omega_h_MEM_SPACE_SHARED "enabled shared memory space" ON)
  endif()

  message (STATUS " *** Begin of Omega_h configuration ***")
  FetchContent_MakeAvailable (Omega_h)
  message (STATUS " ***  End of Omega_h configuration  ***")
endif()
