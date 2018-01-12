macro(do_wiki_update TEST_RUN_FAILED)

# Only update the "Good Commits" page on the wiki if we are submitting results to the dashboard
# This will allow for testing the scripts without messing with the wiki page
# Also, only update if the status of the tests is 0

if (NOT TEST_RUN_FAILED AND CTEST_DO_SUBMIT)

  message(STATUS "")
  message(STATUS "Updating ** Good Commits ** wiki page")

# Get the Trininos HEAD sha 
  execute_process (COMMAND "${CTEST_GIT_COMMAND}" rev-parse HEAD
    WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/Trilinos
    OUTPUT_VARIABLE TRILINOS_HEAD_SHA
    ERROR_VARIABLE _err
    RESULT_VARIABLE HAD_ERROR)
  message(STATUS "Trilinos HEAD sha: ${TRILINOS_HEAD_SHA}")
  message(STATUS "error text: ${_err}")
  message(STATUS "command result status: ${HAD_ERROR}")
  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot determine Trilinos HEAD sha!")
  endif ()

# Get the Albany HEAD sha 
  execute_process (COMMAND "${CTEST_GIT_COMMAND}" rev-parse HEAD
    WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/Albany
    OUTPUT_VARIABLE ALBANY_HEAD_SHA
    ERROR_VARIABLE _err
    RESULT_VARIABLE HAD_ERROR)
  message(STATUS "Albany HEAD sha: ${ALBANY_HEAD_SHA}")
  message(STATUS "error text: ${_err}")
  message(STATUS "command result status: ${HAD_ERROR}")
  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot determine Albany HEAD sha!")
  endif ()

# Pull the current wiki state
  execute_process (COMMAND "${CTEST_GIT_COMMAND}" pull
    WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/Albany.wiki
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE HAD_ERROR)
  message(STATUS "Output of wiki pull: ${_out}")
  message(STATUS "error text: ${_err}")
  message(STATUS "command result status: ${HAD_ERROR}")
  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot pull Albany.wiki!")
  endif ()

# Update the wiki page

  CONFIGURE_FILE (${CTEST_SCRIPT_DIRECTORY}/commits_template.md
     ${CTEST_SOURCE_DIRECTORY}/Albany.wiki/Current-Working-Sha.md)

# Formulate the commit
  SET(WIKI_COMMIT_MSG "Update latest known good commits")
  execute_process (COMMAND "${CTEST_GIT_COMMAND}" commit -a -m "${WIKI_COMMIT_MSG}"
    WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/Albany.wiki
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE HAD_ERROR)
  message(STATUS "Output of wiki commit: ${_out}")
  message(STATUS "error text: ${_err}")
  message(STATUS "command result status: ${HAD_ERROR}")
  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot commit changes to Albany.wiki!")
  endif ()

# Do the push
  execute_process (COMMAND "${CTEST_GIT_COMMAND}" push
    WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/Albany.wiki
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE HAD_ERROR)
  message(STATUS "Output of wiki push: ${_out}")
  message(STATUS "error text: ${_err}")
  message(STATUS "command result status: ${HAD_ERROR}")
  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot push changes to Albany.wiki!")
 endif ()

  message(STATUS "Finished updating wiki ** Good Commits ** page.")

endif (NOT TEST_RUN_FAILED AND CTEST_DO_SUBMIT)

endmacro(do_wiki_update TEST_RUN_FAILED)
