# 1. Run the program and generate the exodus output

EXECUTE_PROCESS(COMMAND ${TEST_PROG} nodal.yaml RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany: test failed")
endif()
