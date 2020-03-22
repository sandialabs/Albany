if(LCM_DO_PACKAGE_CMAKE)
  return()
endif()
set(LCM_DO_PACKAGE_CMAKE true)

include(${CMAKE_CURRENT_LIST_DIR}/lcm_do_trilinos.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/lcm_do_albany.cmake)

function(lcm_do_package)
  set(BOOL_OPTS
      "CLEAN_BUILD"
      "CLEAN_INSTALL"
      "DO_UPDATE"
      "DO_CONFIG"
      "DO_BUILD"
      "DO_TEST"
     )
  set(UNARY_OPTS
      "PACKAGE"
      "BUILD_THREADS"
      "RESULT_VARIABLE"
      "BUILD_ID_STRING"
    )
  message("lcm_do_package(${ARGN})")
  cmake_parse_arguments(ARG "${BOOL_OPTS}" "${UNARY_OPTS}" "" ${ARGN}) 
  if (ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
        "lcm_do_trilinos called with unrecognized arguments ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  set(ARG_BOOL_OPTS)
  foreach (BOOL_OPT IN LISTS BOOL_OPTS)
    if (ARG_${BOOL_OPT})
      set(ARG_BOOL_OPTS ${ARG_BOOL_OPTS} ${BOOL_OPT})
    endif()
  endforeach()
  set(PASS_ARGS ${ARG_BOOL_OPTS}
      BUILD_THREADS "${ARG_BUILD_THREADS}"
      RESULT_VARIABLE "PACKAGE_ERR"
      BUILD_ID_STRING "${ARG_BUILD_ID_STRING}"
     )
  # all other arguments passed to do_trilinos or do_albany
  if ("${ARG_PACKAGE}" STREQUAL "trilinos")
    lcm_do_trilinos(${PASS_ARGS})
  elseif ("${ARG_PACKAGE}" STREQUAL "albany")
    lcm_do_albany(${PASS_ARGS})
  else()
    message(FATAL_ERROR 
      "PACKAGE was \"${ARG_PACKAGE}\", should be \"trilinos\" or \"albany\"")
  endif()
  if (ARG_RESULT_VARIABLE)
    set(${ARG_RESULT_VARIABLE} ${PACKAGE_ERR} PARENT_SCOPE)
  endif()
endfunction(lcm_do_package)
