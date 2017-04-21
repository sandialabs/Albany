function(do_config SOURCE_IN BUILD_IN OPTS_IN)
  if (EXISTS "${BUILD_IN}/CMakeCache.txt")
    file(REMOVE "${BUILD_IN}/CMakeCache.txt")
  endif()

  if (EXISTS "${BUILD_IN}/CMakeFiles")
    file(REMOVE_RECURSE "${BUILD_IN}/CMakeFiles")
  endif()

  ctest_configure(
    BUILD "${BUILD_IN}"
    SOURCE "${SOURCE_IN}"
    OPTIONS "${OPTS_IN}"
    RETURN_VALUE CONFIG_ERR
    APPEND
  )

  if (CTEST_DO_SUBMIT)
    ctest_submit(PARTS Configure RETURN_VALUE SUBMIT_ERR)
    if(SUBMIT_ERR)
      message("Cannot submit Albany configure results!")
    endif()
  endif()

  if (CONFIG_ERR)
    message("Cannot configure Albany!")
  endif()
endfunction(do_config)
