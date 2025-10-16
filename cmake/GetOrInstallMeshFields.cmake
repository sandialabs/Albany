set (tmpStr "Looking for valid MeshFields installation ...")
message (STATUS ${tmpStr})

# Get all propreties that cmake supports
if(NOT CMAKE_PROPERTY_LIST)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
    
    # Convert command output into a CMake list
    string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    list(REMOVE_DUPLICATES CMAKE_PROPERTY_LIST)
endif()
    
function(print_properties)
    message("CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST}")
endfunction()
    
function(print_target_properties target)
    if(NOT TARGET ${target})
      message(STATUS "There is no target named '${target}'")
      return()
    endif()

    foreach(property ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
            continue()
        endif()

        get_property(was_set TARGET ${target} PROPERTY ${property} SET)
        if(was_set)
            get_target_property(value ${target} ${property})
            message("${target} ${property} = ${value}")
        endif()
    endforeach()
endfunction()

find_package(MeshFields CONFIG QUIET
  # Avoid all defaults. Only check env/CMake var MeshFields_ROOT
  NO_CMAKE_PATH
  NO_CMAKE_ENVIRONMENT_PATH
  NO_SYSTEM_ENVIRONMENT_PATH
  NO_CMAKE_PACKAGE_REGISTRY
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_INSTALL_PREFIX
  NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

# If we're building shared libs in Albany, we need MeshFields to be built with shared libs
# TODO: if CMake adds a "DYNAMYC=<value>" to find_package (to specify what variant
# of libs we want), then we can use it, and remove (some of) this snippet
if (MeshFields_FOUND)
  message (STATUS "${tmpStr} Found.")
  message ("  -- MeshFields_DIR: ${MeshFields_DIR}")
  message ("  -- MeshFields_VERSION: ${MeshFields_VERSION}")
  get_target_property(MeshFields_LIBTYPE MeshFields::meshfields TYPE)
  if (BUILD_SHARED_LIBS AND NOT MeshFields_LIBTYPE STREQUAL SHARED_LIBRARY)
    message ("     -> Wrong lib variant: ${MeshFields_LIBTYPE} instead of SHARED_LIBRARY.")
    message ("        Please, point to a compatible MeshFields installation (or none at all to force a new install)")
    message (FATAL_ERROR "Aborting...")
  else()
    foreach (opt IN ITEMS MeshFields_USE_Kokkos MeshFields_USE_MPI)
      if (NOT ${opt})
        message ("     -> Wrong configuration: ${opt} is OFF (should be ON)")
        message ("        Please, point to a compatible MeshFields installation (or none at all to force local install)")
        message (FATAL_ERROR "Aborting...")
      endif()
    endforeach()
  endif()
else ()
  message (STATUS "${tmpStr} NOT Found.")
  message (STATUS "  -> Downloading and building locally in ${CMAKE_BINARY_DIR}/tpls/meshfields")

  include (FetchContent)

  # Fetch and populate the external project
  set (FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/tpls/meshfields)

  FetchContent_Declare (
    MeshFields
    GIT_REPOSITORY git@github.com:SCOREC/meshFields
    GIT_TAG        origin/cws/supportFetchContent
    OVERRIDE_FIND_PACKAGE
  )

  # Set options for MeshFields before adding the subdirectory
  get_target_property(Kokkos_INCLUDE_DIR Kokkos::kokkos INTERFACE_INCLUDE_DIRECTORIES)
  string(REPLACE "include/kokkos" "" Kokkos_INSTALL_DIR ${Kokkos_INCLUDE_DIR})
  set(Kokkos_PREFIX ${Kokkos_INSTALL_DIR} PATH "Path to Kokkos install")

  message (STATUS " *** Begin of MeshFields configuration ***")
  FetchContent_MakeAvailable (MeshFields)
  message (STATUS " ***  End of MeshFields configuration  ***")
endif()
