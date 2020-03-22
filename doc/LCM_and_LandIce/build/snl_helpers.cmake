if(SNL_HELPERS_CMAKE)
  return()
endif()
set(SNL_HELPERS_CMAKE true)

include(CMakeParseArguments)

#IKT, 3/21/20: I don't think the following is needed if no subprojects...
#function(snl_move_xml PART)
#  message("expecting files to be in \"${CTEST_DROP_LOCATION}\"")
#  file(GLOB MATCHING_FILES "${CTEST_DROP_LOCATION}/*___XML___${PART}.xml")
#  get_property(SUBPROJECT GLOBAL PROPERTY SubProject)
#  foreach(MATCHING_FILE IN LISTS MATCHING_FILES)
#    message("mv ${MATCHING_FILE} -> ${CTEST_DROP_LOCATION}/${SUBPROJECT}_${PART}.xml")
#    file(RENAME "${MATCHING_FILE}" "${CTEST_DROP_LOCATION}/${SUBPROJECT}_${PART}.xml")
#  endforeach()
#endfunction(snl_move_xml)

function(snl_submit PART)
  if (CTEST_DO_SUBMIT)
    ctest_submit(PARTS ${PART}
      RETRY_COUNT 3
      RETRY_DELAY 10
      RETURN_VALUE SUBMIT_ERR)
    if(SUBMIT_ERR)
      message(WARNING "Cannot submit ${PART} results!")
    endif()
  endif()
  message("CTEST_DROP_METHOD \"${CTEST_DROP_METHOD}\"")
  if (CTEST_DROP_METHOD STREQUAL "cp")
    snl_move_xml("${PART}")
  endif()
endfunction(snl_submit)

function(snl_rmdir DIR)
  message("snl_rmdir ${DIR}")
  if (EXISTS "${DIR}")
    message("rm -rf ${DIR}")
    file(REMOVE_RECURSE "${DIR}")
  endif()
endfunction(snl_rmdir)

function(snl_mkdir DIR)
  if (NOT EXISTS "${DIR}")
    file(MAKE_DIRECTORY "${DIR}")
  endif()
endfunction(snl_mkdir)

function(snl_update REPO_URL BRANCH SOURCE_DIR ERR)
  if (NOT BRANCH)
    set(BRANCH "master")
  endif()
  if (NOT EXISTS "${SOURCE_DIR}")
    execute_process(COMMAND "${CTEST_GIT_COMMAND}"
      clone -b "${BRANCH}" "${REPO_URL}" "${SOURCE_DIR}"
      RESULT_VARIABLE CLONE_ERR)
    if (CLONE_ERR)
      message(WARNING "Cannot clone ${REPO_URL} branch ${BRANCH}")
      set(${ERR} ${CLONE_ERR} PARENT_SCOPE)
      return()
    endif()
  endif()
  ctest_update(SOURCE "${SOURCE_DIR}" RETURN_VALUE FILES_CHANGED)
  if (FILES_CHANGED LESS 0)
    # we can in theory proceed with the out-of-date code
    message(WARNING "Cannot update ${REPO_URL} branch ${BRANCH}") 
  else()
    snl_submit("Update")
  endif()
  set(${ERR} 0 PARENT_SCOPE)
endfunction(snl_update)

function(snl_config SOURCE_DIR BUILD_DIR CONFIG_OPTS ERR)
  snl_mkdir("${BUILD_DIR}")
  ctest_configure(
    BUILD "${BUILD_DIR}"
    SOURCE "${SOURCE_DIR}"
    APPEND
    OPTIONS "${CONFIG_OPTS}"
    RETURN_VALUE CONFIG_ERR
  )
  snl_submit("Configure")
  if (CONFIG_ERR)
    message(WARNING "Cannot configure!")
    set(${ERR} ${CONFIG_ERR} PARENT_SCOPE)
    return()
  endif()
  set(${ERR} 0 PARENT_SCOPE)
endfunction(snl_config)

function(snl_build BUILD_DIR NUM_THREADS TARGET ERR)
  ctest_build(
    BUILD "${BUILD_DIR}"
    APPEND
    FLAGS "-j ${NUM_THREADS}"
    TARGET "${TARGET}"
    NUMBER_ERRORS NERRS
    NUMBER_WARNINGS NWARNS
    RETURN_VALUE BUILD_ERR
  )
  snl_submit("Build")
  if ((NOT BUILD_ERR) AND (NERRS GREATER "0"))
    message("BUILD_ERR was ${BUILD_ERR} despite NERRS being ${NERRS}")
    set(BUILD_ERR "-${NERRS}")
  endif()
  if (BUILD_ERR)
    message(WARNING "Cannot make ${TARGET}!")
    set(${ERR} ${BUILD_ERR} PARENT_SCOPE)
    return()
  endif()
  set(${ERR} 0 PARENT_SCOPE)
endfunction(snl_build)

function(snl_test BUILD_DIR)
  #TODO: the following call may not be appropriate when
  #this script is used by others besides LCM
  message("ctest_read_custom_files(${CTEST_BINARY_DIRECTORY})")
  ctest_read_custom_files("${CTEST_BINARY_DIRECTORY}")
  ctest_test(
    BUILD "${BUILD_DIR}"
    APPEND
  )
  snl_submit("Test")
endfunction(snl_test)

function(snl_submit_subprojects)
  if (NOT CTEST_DO_SUBMIT)
    return()
  endif()
  set(ONE_VALUE_OPTS
      "PROJECT"
      "FILE_DIR"
    )
  cmake_parse_arguments(SNL "" "${ONE_VALUE_OPTS}" "${MULTI_VALUE_OPTS}" ${ARGN})
  if (SNL_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "snl_create_subprojects called with unrecognized arguments \"${SNL_UNPARSED_ARGUMENTS}\", all arguments were \"${ARGN}\"")
  endif()
  set(CONTENT "<Project name=\"${SNL_PROJECT}\">")
  message("CONTENT
${CONTENT}")
  set(CONTENT "${CONTENT}
</Project>")
  message("CONTENT
${CONTENT}")
  message("file(WRITE ${SNL_FILE_DIR}/Project.xml CONTENT
${CONTENT}")
  snl_mkdir("${SNL_FILE_DIR}")
  file(WRITE "${SNL_FILE_DIR}/Project.xml" "${CONTENT}") 
  ctest_submit(FILES "${SNL_FILE_DIR}/Project.xml"
               RETURN_VALUE SUBMIT_ERR)
  if(SUBMIT_ERR)
    message(WARNING "Cannot submit ${SNL_FILE_DIR}/Project.xml!")
  endif()
endfunction(snl_submit_subprojects)

function(snl_do_subproject)
  set(BOOL_OPTS
      "CLEAN_SOURCE"
      "CLEAN_BUILD"
      "CLEAN_INSTALL"
      "DO_PROJECT"
      "DO_UPDATE"
      "DO_CONFIG"
      "DO_BUILD"
      "DO_INSTALL"
      "DO_TEST")
  set(ONE_VALUE_OPTS
      "PROJECT"
      "REPO_URL"
      "BRANCH"
      "SOURCE_DIR"
      "BUILD_DIR"
      "INSTALL_DIR"
      "BUILD_THREADS"
      "RESULT_VARIABLE"
    )
  set(MULTI_VALUE_OPTS
      "CONFIG_OPTS"
    )
  cmake_parse_arguments(SNL "${BOOL_OPTS}" "${ONE_VALUE_OPTS}" "${MULTI_VALUE_OPTS}" ${ARGN})
  if (SNL_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "snl_do_subproject called with unrecognized arguments \"${SNL_UNPARSED_ARGUMENTS}\", all arguments were \"${ARGN}\"")
  endif()
  if (SNL_CLEAN_INSTALL)
    snl_rmdir("${SNL_INSTALL_DIR}")
  endif()
  if (SNL_CLEAN_BUILD)
    snl_rmdir("${SNL_BUILD_DIR}")
  endif()
  if (SNL_CLEAN_SOURCE)
    snl_rmdir("${SNL_SOURCE_DIR}")
  endif()
  if (SNL_DO_PROJECT)
    snl_submit_subprojects(
        "PROJECT" "${SNL_PROJECT}" 
        FILE_DIR "${SNL_BUILD_DIR}"
        )
  endif()
  if (SNL_DO_UPDATE)
    snl_update("${SNL_REPO_URL}" "${SNL_BRANCH}"
        "${SNL_SOURCE_DIR}" UPDATE_ERR)
    if (UPDATE_ERR)
      if (SNL_RESULT_VARIABLE)
        set(${SNL_RESULT_VARIABLE} ${UPDATE_ERR} PARENT_SCOPE)
      endif()
      return()
    endif()
  endif()
  if (SNL_DO_CONFIG)
    snl_config("${SNL_SOURCE_DIR}" "${SNL_BUILD_DIR}" "${SNL_CONFIG_OPTS}" CONFIG_ERR)
    if (CONFIG_ERR)
      if (SNL_RESULT_VARIABLE)
        set(${SNL_RESULT_VARIABLE} ${CONFIG_ERR} PARENT_SCOPE)
      endif()
      return()
    endif()
  endif()
  if (SNL_DO_BUILD)
    snl_build("${SNL_BUILD_DIR}" "${SNL_BUILD_THREADS}" "all" BUILD_ERR)
    message("snl_build()  returned ${BUILD_ERR}")
    if (BUILD_ERR)
      if (SNL_RESULT_VARIABLE)
        set(${SNL_RESULT_VARIABLE} ${BUILD_ERR} PARENT_SCOPE)
      endif()
      return()
    endif()
  endif()
  if (SNL_DO_TEST)
    snl_test("${SNL_BUILD_DIR}")
  endif()
  if (SNL_DO_INSTALL)
    snl_mkdir("${SNL_INSTALL_DIR}")
    snl_build("${SNL_BUILD_DIR}" "${SNL_BUILD_THREADS}" "install" INSTALL_ERR)
    if (INSTALL_ERR)
      if (SNL_RESULT_VARIABLE)
        set(${SNL_RESULT_VARIABLE} ${INSTALL_ERR} PARENT_SCOPE)
      endif()
      return()
    endif()
  endif()
  if (SNL_RESULT_VARIABLE)
    set(${SNL_RESULT_VARIABLE} 0 PARENT_SCOPE)
  endif()
endfunction(snl_do_subproject)
