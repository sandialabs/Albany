# 1. Run the program and generate the exodus output

message("Running the command:")
message("${TEST_PROG} " " ${TEST_ARGS}")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} ${TEST_ARGS}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany didn't run: test failed")
endif()

# 2. Find and run sed

FIND_PROGRAM(SED_EXECUTABLE sed PATHS /usr/bin)

SET(SED_COMMAND ${SED_EXECUTABLE} -n $p ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME})

EXECUTE_PROCESS(COMMAND ${SED_COMMAND}
   OUTPUT_VARIABLE SED_LINE)

IF("${SED_LINE}" MATCHES "</VTKFile>")
   RETURN()
ELSE()
   message(FATAL_ERROR "Test failed: PVD file truncated!")
ENDIF()
