# 1. Run the program and generate the exodus output

message("Running the test prep code:")
message("${AUX_ENV} " " ${AUX_PROG} ")

# Get rid of any spaces
STRING(STRIP "${AUX_ENV}" AUX_PROG_ENV)
STRING(STRIP "${AUX_PROG}" AUX_PROG_PROG)

EXECUTE_PROCESS(COMMAND ${AUX_PROG_ENV} ${AUX_PROG_PROG}
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

