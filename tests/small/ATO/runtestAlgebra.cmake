# This script is responsible of running a test and then compare against a blessed baseline.

# The old behavior of CMP0012 does not guarantee that, say, "if(TRUE)"
# will execute the branch. It always dereferenced the input as if it
# was a variable name (except for numbers '0' and '1').
# The new behavior recognizes boolean variables as well as yes, no,
# on, off, y, n, and numbers (case insensitive).
# Since we configure this script into the bin dir, variables like
# ALBANY_MPI will no longer be visible to cmake during the execution,
# so we need to expand them now, leading to code like 'if(TRUE)'.
# Hence, we need the new policy

cmake_policy(SET CMP0012 NEW)

# bare test name (without Epetra/Tpetra specifier)
STRING (REGEX MATCH _[TE]petra petraName @TEST_NAME@)
STRING(REPLACE "${petraName}" "" TEST_NAME_BARE  @TEST_NAME@)

# 1. Run the program and generate the exodus output
message("Running the command: @TEST_PROG@ @TEST_ARGS@")

EXECUTE_PROCESS(COMMAND @TEST_PROG@ @TEST_ARGS@
                OUTPUT_VARIABLE OUT_VAR
                ERROR_VARIABLE  OUT_AR
                RESULT_VARIABLE HAD_ERROR)

message ("Output:\n${OUT_VAR}\n---")

if(HAD_ERROR)
	message(FATAL_ERROR "Error running test @TEST_NAME@:\n${HAD_ERROR}")
endif()

# 2. Find and run epu if parallel

if(@ALBANY_MPI@ AND @MPIMNP@ GREATER 1)

	if ("@SEACAS_EPU@" STREQUAL "")
		message(FATAL_ERROR "Cannot find epu")
	endif()

	SET(EPU_COMMAND @SEACAS_EPU@ -auto @EXO_FILE_NAME@.exo.@MPIMNP@.0)

  message("Running the command:")
  message("${EPU_COMMAND}")

	EXECUTE_PROCESS(COMMAND ${EPU_COMMAND}
              		RESULT_VARIABLE HAD_ERROR)

	if(HAD_ERROR)
		message(FATAL_ERROR "epu failed")
	endif()

endif()


# 2. Find and run exodiff

if ("@SEACAS_EXODIFF@" STREQUAL "")
  message(FATAL_ERROR "Cannot find exodiff")
endif()

if ("@SEACAS_ALGEBRA@" STREQUAL "")
  message(FATAL_ERROR "Cannot find algebra")
endif()

if(@ALBANY_MPI@ AND @MPIMNP@ GREATER 1)
  SET(EXODIFF_TEST @SEACAS_EXODIFF@ -i -m -f @DATA_DIR@/${TEST_NAME_BARE}.exodiff_commands @EXO_FILE_NAME@.b1.exo @DATA_DIR@/${TEST_NAME_BARE}.ref.exo)
ELSE()
  SET(EXODIFF_TEST @SEACAS_EXODIFF@ -i -f @DATA_DIR@/${TEST_NAME_BARE}.exodiff_commands @EXO_FILE_NAME@.b1.exo @DATA_DIR@/${TEST_NAME_BARE}.ref.exo)
ENDIF()

message("Running the command:")
message("${EXODIFF_TEST}")

EXECUTE_PROCESS(
    COMMAND @SEACAS_ALGEBRA@ @EXO_FILE_NAME@.exo @EXO_FILE_NAME@.b1.exo
    INPUT_FILE @DATA_DIR@/alg.in
    OUTPUT_FILE algebra${petraName}.out
    RESULT_VARIABLE ALG_ERROR)

if(ALG_ERROR)
	message(FATAL_ERROR "Algebra step failed")
endif()

EXECUTE_PROCESS(
    COMMAND ${EXODIFF_TEST}
    OUTPUT_FILE @EXO_DIFF_FILE_NAME@${petraName}
    RESULT_VARIABLE HAD_ERROR)

if(HAD_ERROR)
	message(FATAL_ERROR "Test failed")
endif()

