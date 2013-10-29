# 1. Run the program and generate the exodus output

message("Running the test prep code:")
message("${AUX_PROG} ")

EXECUTE_PROCESS(COMMAND ${AUX_PROG}
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

