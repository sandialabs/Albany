# 1. Run the program and generate the exodus output

message("running: " ${ALBANY} ${TESTNAME}T.yaml)

EXECUTE_PROCESS(COMMAND ${ALBANY} ${TESTNAME}T.yaml
    OUTPUT_FILE "Albany.out"
    ERROR_FILE "Albany.err"
    OUTPUT_VARIABLE ALBANY_OUTPUT
    ERROR_VARIABLE ALBANY_ERROR
    RESULT_VARIABLE RET)

if(${ALBANY_OUTPUT})
  message("stdout:")
  message(${ALBANY_OUTPUT})
endif()
if(${ALBANY_ERROR})
  message("stderr:")
  message(${ALBANY_ERROR})
endif()

if(RET)
	message(FATAL_ERROR "Albany failed")
endif()

EXECUTE_PROCESS(
    COMMAND ${EXODIFF} -stat -f ${TESTNAME}.exodiff
    ${TESTNAME}.gold.e ${TESTNAME}.e
    OUTPUT_FILE "exodiff.out"
    ERROR_FILE "exodiff.err"
    RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "exodiff failed")
endif()

