

EXECUTE_PROCESS(COMMAND bash 
    COMMAND bash -c "sed -i 's/Use Serial Mesh: true/Use Serial Mesh: false/g' ${TEST_ARGS}"
    RESULT_VARIABLE SED_ERROR)
if(SED_ERROR)
	message(FATAL_ERROR "sed step failed")
endif()

#Run the program and generate the exodus output

message("Running the command:")
message("${TEST_PROG} " " ${TEST_ARGS}")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} ${TEST_ARGS}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany didn't run: test failed")
endif()


