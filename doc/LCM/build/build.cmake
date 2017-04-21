set(LINE "------------------------------------------------------------")

function(do_config PKG_IN SOURCE_IN BUILD_IN OPTS_IN RETVAR)
  set(${RETVAR} FALSE PARENT_SCOPE)
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
      message("Cannot submit ${PKG_IN} configure results!")
    endif()
  endif()
  if (CONFIG_ERR)
    message("Cannot configure ${PKG_IN}!")
    return()
  endif()
  set(${RETVAR} TRUE PARENT_SCOPE)
endfunction(do_config)

function(do_build PKG_IN BUILD_IN NPROCS_IN TARGET_IN RETVAR)
  set(${RETVAR} FALSE PARENT_SCOPE)
  message("BUILDING ${PKG_IN} ...")
  message("${LINE}")
  ctest_build(
    BUILD "${BUILD_IN}"
    APPEND
    FLAGS "-j ${NPROCS_IN}"
    TARGET "${TARGET_IN}"
    NUMBER_ERRORS NERRS
    NUMBER_WARNINGS NWARNS
    RETURN_VALUE STATUS
  )
  if (CTEST_DO_SUBMIT)
    ctest_submit(PARTS Build RETURN_VALUE SUBMIT_ERR)
    if(SUBMIT_ERR)
      message("Cannot submit ${PKG_IN} build results!")
    endif()
  endif()
  if (STATUS)
    string(TOUPPER "${TARGET_IN}" TARGET_ALLCAPS)
    message("*** MAKE ${TARGET_ALLCAPS} COMMAND FAILED ***")
    return()
  endif()
  set(${RETVAR} TRUE PARENT_SCOPE)
endfunction(do_build)

function(do_test BUILD_IN RETVAR)
  set(${RETVAR} FALSE PARENT_SCOPE)
  if (NOT EXISTS "${BUILD_IN}")
    message("Build directory does not exist. Run:")
    message("  [clean-]config-build.sh ${PKG_IN} ...")
    message("to create.")
    return()
  endif()
  message("TESTING ${PKG_IN} ...")
  message("${LINE}")
  ctest_test(
    BUILD "${BUILD_IN}"
    APPEND
    RETURN_VALUE ERR
  )
  if (CTEST_DO_SUBMIT)
    ctest_submit(PARTS Test RETURN_VALUE SUBMIT_ERR)
    if(SUBMIT_ERR)
      message("Cannot submit ${PKG_IN} test results!")
    endif()
  endif()
endfunction(do_test)
