# 1. Run the program and generate the exodus output

message("Running the command:")
message("${TEST_PROG} " " ${TEST_ARGS}")

EXECUTE_PROCESS(COMMAND ${TEST_PROG} ${TEST_ARGS}
                RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany didn't run: test failed")
endif()

# 2. Find and run epu if parallel

if(DEFINED MPIMNP AND ${MPIMNP} GREATER 1)

#	find_program(SEACAS_EPU NAMES epu PATHS ENV PATH) 

	if (NOT SEACAS_EPU)
		message(FATAL_ERROR "Cannot find epu")
	endif()

	SET(EPU_COMMAND ${SEACAS_EPU} -auto ${FILE_NAME}.exo.${MPIMNP}.0)

  message("Running the command:")
  message("${EPU_COMMAND}")

	EXECUTE_PROCESS(COMMAND ${EPU_COMMAND}
		RESULT_VARIABLE HAD_ERROR)

	if(HAD_ERROR)
		message(FATAL_ERROR "epu failed")
	endif()

endif()


# 2. Find and run exodiff

# find_program(EXODIFF NAMES exodiff PATHS ENV PATH)

if (NOT SEACAS_EXODIFF)
  message(FATAL_ERROR "Cannot find exodiff")
endif()

if(DEFINED MPIMNP AND ${MPIMNP} GREATER 1)
  SET(EXODIFF_TEST ${SEACAS_EXODIFF} -i -m -f ${DATA_DIR}/${FILE_NAME}.exodiff_commands ${FILE_NAME}.exo ${DATA_DIR}/${FILE_NAME}.ref.exo)
ELSE()
  SET(EXODIFF_TEST ${SEACAS_EXODIFF} -i -f ${DATA_DIR}/${FILE_NAME}.exodiff_commands ${FILE_NAME}.exo ${DATA_DIR}/${FILE_NAME}.ref.exo)
ENDIF()

message("Running the command:")
message("${EXODIFF_TEST}")

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST}
    OUTPUT_FILE exodiff.out
    RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()
