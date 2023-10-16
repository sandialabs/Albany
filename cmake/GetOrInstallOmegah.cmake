set (tmpStr "Looking for valid Omega_h installation ...")
message (STATUS ${tmpStr})
# NOTE: if Omega_h_ROOT is declared as env/cmake var, it will take precedence
find_package(Omega_h 10.8 CONFIG QUIET
             HINTS ${CMAKE_INSTALL_PREFIX})

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
  # Check if the install prefix is writable. If not, it may be the user is
  # not interested in installing albany, so they didn't expliclty set
  # CMAKE_INSTALL_PREFIX, which then defaulted to system folders. If so,
  # we simply install inside the build tree, to avoid installation errors
  if (EXISTS ${CMAKE_INSTALL_PREFIX})
    execute_process (COMMAND test -w ${CMAKE_INSTALL_PREFIX}
                     RESULT_VARIABLE denied)
    if (denied)
      set (LOCAL_INSTALL TRUE)
      set (Omega_h_DIR ${Albany_BINARY_DIR}/tpls/omegah/install)
    else()
      set (Omega_h_DIR ${CMAKE_INSTALL_PREFIX})
    endif()
  else ()
    # Folder does not exists. We can try to create it. If we succeed, we can use it,
    # otherwise it is probably the default system installation, so we install
    # omegah locally, just like the 'denied' branch above.
    execute_process (COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}
                     OUTPUT_QUIET ERROR_QUIET
                     RESULT_VARIABLE denied)
    if (denied)
      set (LOCAL_INSTALL TRUE)
      set (Omega_h_DIR ${Albany_BINARY_DIR}/tpls/omegah/install)
    else()
      set (Omega_h_DIR ${CMAKE_INSTALL_PREFIX})
    endif()
  endif()

  # We may have found the wrong config (e.g., static libs instead of shared).
  # To avoid keeping the wrong lib around (which may get linked), we remove
  # any lib found, as well as all the CMake config files.
  file (GLOB OMEGAH_LIBS ${Omega_h_DIR}/${CMAKE_INSTALL_LIBDIR}/libomega_h.*)
  if (OMEGAH_LIBS)
    file (REMOVE ${OMEGAH_LIBS})
  endif()
  file (GLOB OMEGAH_CMAKE_DIR RELATIVE ${Omega_h_DIR} LIST_DIRECTORIES TRUE Omega_h)
  if (OMEGAH_CMAKE_DIR)
    file (REMOVE_RECURSE ${OMEGAH_CMAKE_DIR})
  endif()

  message (STATUS "Installing Omega_h ...")
  message (STATUS "  prefix: ${Omega_h_DIR}")
  if (LOCAL_INSTALL)
    message (STATUS "  Installing locally in build tree since CMAKE_INSTALL_PREFIX is not writable")
    message (STATUS "     CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
  endif ()
  # The first external project will be built at *configure stage*
  # Configure
  execute_process(
    COMMAND ${CMAKE_COMMAND}
            -B ${Albany_BINARY_DIR}/tpls/omegah -Wno-dev
            -S ${CMAKE_SOURCE_DIR}/cmake/omegah
            -D Albany_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -D Albany_MPI_EXEC=${MPIEX}
            -D Albany_BINARY_DIR=${Albany_BINARY_DIR}
            -D Kokkos_INSTALL_DIR=${ALBANY_TRILINOS_DIR}
            -D IS_SHARED=${BUILD_SHARED_LIBS}
            -D Omega_h_INSTALL_DIR=${Omega_h_DIR}
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
            ${Albany_BINARY_DIR}/tpls/omegah
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
  message (STATUS "Installing Omega_h ... DONE!")
  find_package (Omega_h REQUIRED QUIET
                HINTS ${Omega_h_DIR})
else()
  message (STATUS "${tmpStr} Found.")
  message ("  -- Omega_h_DIR: ${Omega_h_DIR}")
  message ("  -- Omega_h_VERSION: ${Omega_h_VERSION}")
endif()
