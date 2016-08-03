# 1. Run the program and generate the exodus output

message("Running the test prep code:")
message("${AUX_ENV} " " ${AUX_PROG} ")

SET(TMP_COMMAND_STR "${AUX_ENV} ${AUX_PROG}")
STRING(REPLACE " " ";" TMP_LIST ${TMP_COMMAND_STR})

# Get the command
LIST(GET TMP_LIST 0 TMP_COMMAND)
LIST(REMOVE_AT TMP_LIST 0)

EXECUTE_PROCESS(COMMAND ${TMP_COMMAND} ${TMP_LIST}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Test prep didn't run: test failed")
endif()

# 2. Run Albany

message("Running the command:")
message("${TEST_PROG} " " ${TEST_ARGS}")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} ${TEST_ARGS}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany didn't run: test failed")
endif()

