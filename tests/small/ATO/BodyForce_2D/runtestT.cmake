# 1. Run the program and generate the exodus output

set (PARALLEL,  DEFINED MPIMNP AND ${MPIMNP} GREATER 1 )

if( NOT ${PARALLEL} )

  EXECUTE_PROCESS(COMMAND ${TEST_PROG} mitchellT.xml RESULT_VARIABLE HAD_ERROR)

  IF( HAD_ERROR )
    message(FATAL_ERROR, "Test had errors.")
  ENDIF()

endif()
