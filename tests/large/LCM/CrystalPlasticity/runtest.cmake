


# 1. Run the program and generate the exodus output

message("Running the command:")
message("${TEST_PROG} " " ${TEST_ARGS}")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} ${TEST_ARGS}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany didn't run: test failed")
endif()

# 2. Run epu 

if (DEFINED MPIMNP AND ${MPIMNP} GREATER 1)
  if (NOT SEACAS_EPU)
    message(FATAL_ERROR "Cannot find epu")
  endif()

  SET(EPU_COMMAND ${SEACAS_EPU} -auto ${OUTPUT_FILENAME}.${MPIMNP}.0)

  message("Running the command:")
  message("${EPU_COMMAND}")

  EXECUTE_PROCESS(COMMAND ${EPU_COMMAND}
                RESULT_VARIABLE HAD_ERROR)

  if(HAD_ERROR)
    message(FATAL_ERROR "epu failed")
  endif()

endif()


# 3. Find and run exodiff

if (NOT SEACAS_EXODIFF)
  message(FATAL_ERROR "Cannot find exodiff")
endif()

if(DEFINED MPIMNP AND ${MPIMNP} GREATER 1)
  SET(EXODIFF_TEST ${SEACAS_EXODIFF} -i -m -f ${DATA_DIR}/${TEST_NAME}.exodiff ${OUTPUT_FILENAME} ${DATA_DIR}/${REF_FILENAME})
ELSE()
  SET(EXODIFF_TEST ${SEACAS_EXODIFF} -i -f ${DATA_DIR}/${TEST_NAME}.exodiff ${OUTPUT_FILENAME} ${DATA_DIR}/${REF_FILENAME})
ENDIF()

message("Running the command:")
message("${EXODIFF_TEST}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST}
    #OUTPUT_FILE ${TEST_NAME}.exodiff.out
    RESULT_VARIABLE HAD_ERROR)


if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()
