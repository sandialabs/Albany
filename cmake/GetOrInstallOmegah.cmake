set (tmpStr "Looking for valid Omega_h installation ...")
message (STATUS ${tmpStr})
find_package(Omega_h 10.8 CONFIG
             HINTS ${Albany_BINARY_DIR}/tpls/omegah/install)

# If we're building shared libs in Albany, we need Omega_h to be built with shared libs
# TODO: if CMake adds a "DYNAMYC=<value>" to find_package (to specify what variant
# of libs we want), then we can use it, and remove this snippet
if (Omega_h_FOUND)
  get_target_property(Omega_h_LIBTYPE Omega_h::omega_h TYPE)
  if (BUILD_SHARED_LIBS AND NOT Omega_h_LIBTYPE STREQUAL SHARED_LIBRARY)
    set (Omega_h_FOUND FALSE)

    message (STATUS "${tmpStr} NOT Found (wrong lib variant found).")
    message (STATUS "  Wrong lib variant: ${Omega_h_LIBTYPE} instead of SHARED_LIBRARY.")
  else()
    foreach (opt IN ITEMS Omega_h_USE_Kokkos Omega_h_USE_MPI)
      if (NOT ${opt})
        set (Omega_h_FOUND FALSE)

        message (STATUS "${tmpStr} NOT Found (wrong config lib found).")
        message (STATUS "  A lib was found, but with ${opt} OFF (should be ON)")
      endif()
    endforeach()
  endif()
else ()
  message (STATUS "${tmpStr} NOT Found.")
endif()

if (NOT Omega_h_FOUND)
  # Need to prune the build dir, otherwise we may be keeping an existing
  # installation which had the wrong configuration (e.g., keep a static
  # installation when a shared one is needed)
  file (REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/tpls/omegah)

  message (STATUS "Installing Omega_h locally ...")
  message (STATUS "  prefix: ${CMAKE_CURRENT_BINARY_DIR}/tpls/omegah/install")
  # The first external project will be built at *configure stage*
  # Configure
  execute_process(
    COMMAND ${CMAKE_COMMAND}
            -B ${Albany_BINARY_DIR}/tpls/omegah/tmp -Wno-dev
            -S ${CMAKE_SOURCE_DIR}/cmake/omegah
            -D Albany_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -D Albany_MPI_EXEC=${MPIEX}
            -D Albany_BINARY_DIR=${Albany_BINARY_DIR}
            -D Kokkos_INSTALL_DIR=${ALBANY_TRILINOS_DIR}
            -D IS_SHARED=${BUILD_SHARED_LIBS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_VARIABLE CONFIG_OMEGAH_OUT
    ERROR_VARIABLE CONFIG_OMEGAH_OUT
    RESULT_VARIABLE CONFIG_OMEGAH_RES
  )
  if (NOT CONFIG_OMEGAH_RES EQUAL 0)
    message ("Could not configure omegah")
    message (" output:")
    message ("${CONFIG_OMEGAH_OUT}")
    message (FATAL_ERROR "Die")
  endif()
  # Build/install
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build
            ${Albany_BINARY_DIR}/tpls/omegah/tmp
            --parallel 8
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_VARIABLE BUILD_OMEGAH_OUT
    ERROR_VARIABLE BUILD_OMEGAH_OUT
    RESULT_VARIABLE BUILD_OMEGAH_RES
  )
  if (NOT BUILD_OMEGAH_RES EQUAL 0)
    message ("Could not build omegah")
    message (" output:")
    message ("${BUILD_OMEGAH_OUT}")
    message (FATAL_ERROR "Die")
  endif()
  message (STATUS "Installing Omega_h locally ... DONE!")
  set (Omega_h_DIR ${Albany_BINARY_DIR}/tpls/omegah/install)
  find_package (Omega_h REQUIRED
                HINTS ${Albany_BINARY_DIR}/tpls/omegah/install)
else()
  message (STATUS "${tmpStr} Found.")
  message ("  -- Omega_h_DIR: ${Omega_h_DIR}")
  message ("  -- Omega_h_VERSION: ${Omega_h_VERSION}")
endif()
