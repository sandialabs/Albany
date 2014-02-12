# 1. Run the program and generate the exodus output

EXECUTE_PROCESS(COMMAND ${TEST_PROG} RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Albany didn't run: test failed")
endif()

# 2. Find and run epu if parallel

SET (OUTFILE_NAME hole_out${OUTFILE_SUFFIX}.exo)

if(DEFINED MPIMNP AND ${MPIMNP} GREATER 1)

#	find_program(SEACAS_EPU NAMES epu PATHS ENV PATH) 

	if (NOT SEACAS_EPU)
		message(FATAL_ERROR "Cannot find epu")
	endif()

	SET(EPU_COMMAND ${SEACAS_EPU} -auto ${OUTFILE_NAME}.${MPIMNP}.0)

	EXECUTE_PROCESS(COMMAND ${EPU_COMMAND}
		RESULT_VARIABLE HAD_ERROR)

	if(HAD_ERROR)
		message(FATAL_ERROR "epu failed")
	endif()

endif()

# Old approach - this uses an external program to check the result at one node

## 3. Find and run ncdump
#
#find_program(NCDUMP NAMES ncdump)
#
#if (NOT NCDUMP)
#  message(FATAL_ERROR "Cannot find ncdump")
#endif()
#
#SET(NCDUMP_COMMAND ${NCDUMP} -v vals_elem_var4eb1 hole_out.exo)
#
#EXECUTE_PROCESS(COMMAND ${NCDUMP_COMMAND} OUTPUT_FILE hole_out.ncdump
#	RESULT_VARIABLE HAD_ERROR)
#
#if(HAD_ERROR)
#	message(FATAL_ERROR "ncdump failed")
#endif()
#
## 4. Run the comparison program to test the validity of info in the ncdump file
#
#EXECUTE_PROCESS(COMMAND "./TestNcdumpValues"
#	RESULT_VARIABLE HAD_ERROR)
#
#message(${HAD_ERROR})
#
#if(HAD_ERROR)
#	message(FATAL_ERROR "Test failed")
#endif()

# 3. Find and run exodiff

# find_program(EXODIFF NAMES exodiff PATHS ENV PATH)

if (NOT SEACAS_EXODIFF)
  message(FATAL_ERROR "Cannot find exodiff")
endif()

SET(EXODIFF_TEST ${SEACAS_EXODIFF} -file exodiff_commands ${OUTFILE_NAME} reference_hole.exo)

#message(${EXODIFF_TEST})

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST}
    OUTPUT_FILE exodiff.out${OUTFILE_SUFFIX}
    RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()
