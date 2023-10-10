# When debugging the CMake logic, it can be helpful to
# print all vars with a certain regex, to see what's going on.
# This function does precisely that. If REGEX is not provided,
# it prints ALL cmake vars in scope
function(dump_cmake_variables)
  set (opts)
  set (args1v REGEX)
  set (argsMv)
  cmake_parse_arguments (dcv "${opts}" "${args1v}" "${argsMv}" ${ARGN})

  get_cmake_property(dcv_var_names VARIABLES)

  foreach (var_name IN LISTS dcv_var_names)
    if (dcv_REGEX)
      string(REGEX MATCH ${dcv_REGEX} MATCHES ${var_name})
      if (MATCHES)
        message(STATUS "${var_name}=${${var_name}}")
      endif()
    else ()
      message(STATUS "${var_name}=${${var_name}}")
    endif ()
  endforeach()
endfunction()
