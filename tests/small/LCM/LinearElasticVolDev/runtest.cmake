


# 1. Run the program and generate the exodus output

message("Running the command:")
message("${TEST_PROG} " " ${TEST_ARGS}")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} ${TEST_ARGS}
                OUTPUT_FILE ${LOGFILE} 
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
  EXECUTE_PROCESS(COMMAND cat
          INPUT_FILE ${LOGFILE}
          RESULT_VARIABLE CAT_ERROR)
  message(FATAL_ERROR "Albany didn't run: test failed")
endif()


EXECUTE_PROCESS(COMMAND python
                INPUT_FILE ${PY_FILE}
                RESULT_VARIABLE PY_ERROR)
if(PY_ERROR)
        message(FATAL_ERROR "Python step failed")
endif()


