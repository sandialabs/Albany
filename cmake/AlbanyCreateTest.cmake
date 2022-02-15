# This function creates a unit test that uses one of the Albany executables,
# and feeds the provided input file to it.
# The following optional argument are available:
#
#   - SERIAL: runs the Albany executable with 1 rank
#   - ANALYSIS: runs the AlbanyAnalysis executable, instead of Albany
#   - NP8: limit number of ranks to 8(unless paralle call is already less than 8)
#   - WILL_FAIL: add "WILL_FAIL TRUE" to tests property
#   - RUN_SERIAL: add "RUN_SERIAL TRUE" to tests property
#   - DEFAULT_TIMEOUT: if ${ALBANY_CTEST_TIMEOUT} is set, adds it as timeout
#   - LABELS: add specified labels to the test
#   - FIXTURES_SETUP: same as the native ctest property
#   - FIXTURES_REQUIRED: same as the native ctest property
#   - FIXTURES_REQUIRED_IF: interprets the first element in the list as the name
#     of a cmake var. If that var is TRUE, adds the remaining entries as if they
#     had been provided via FIXTURES_REQUIRED, otherwise do nothing
#   - DEPENDS: same as the native ctest property
#
# This function also takes care of setting the PROCESSORS property to
# match the number of MPI ranks used, and also sets PROCESSOR_AFFINITY to TRUE.
# This allows to run ctest -jN, without mpi jobs clashing on the same cores.

function(AlbanyCreateTest testName inputFile)
  # Parse optional arguments
  set(options SERIAL ANALYSIS NP8 WILL_FAIL RUN_SERIAL DEFAULT_TIMEOUT)
  set(args1v NP)
  set(argsMv LABELS FIXTURES_SETUP FIXTURES_REQUIRED FIXTURES_REQUIRED_IF DEPENDS)

  cmake_parse_arguments(ACT "${options}" "${args1v}" "${argsMv}" ${ARGN})

  # Sanity check
  if(ACT_SERIAL)
    if(ACT_NP8 OR ACT_NP)
      message(FATAL_ERROR "[AlbanyCreateTest] Option 'SERIAL' cannot be used together with NP8 or NP.")
    endif()
  elseif(ACT_NP8 AND ACT_NP)
    message(FATAL_ERROR "[AlbanyCreateTest] Option 'NP8' cannot be used together with NP.")
  endif()

  if(ACT_NP8 AND ACT_ANALYSIS)
    message(FATAL_ERROR "[AlbanyCreateTest] Option 'NP8' and 'ANALYSIS' cannot be used together.")
  endif()

  # Figure out the command to run, as well as the proc count
  # Note: we use two-level indirection here. We are figuring out
  #       the name of the cmake var containing the right command.
  set(numProcs ${MPIMNP})

  set(execCmakeVarName Albany)
  if(ACT_SERIAL)
    set(numProcs 1)
    set(execCmakeVarName Serial${execCmakeVarName})
  elseif(ACT_NP8)
    set(numProcs 8)
    set(execCmakeVarName ${execCmakeVarName}8)
  endif()

  if(ACT_ANALYSIS)
    set(execCmakeVarName ${execCmakeVarName}Analysis)
  endif()

  if(ACT_NP8)
    if(MPIMNP LESS 8)
      set(numProcs 8)
    endif()
    set(execCmakeVarName ${execCmakeVarName}8)
  elseif(ACT_NP)
    if(ALBANY_MPI_EXEC_MAX_NUMPROCS)

  endif()
  set(execCmakeVarName ${execCmakeVarName}.exe)

  set(execCall ${${execCmakeVarName}})

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${inputFile}
                 ${CMAKE_CURRENT_BINARY_DIR}/${inputFile})

  # Create the test
  add_test(NAME ${testName}
            COMMAND ${execCall} ${inputFile})

  # Set all properties. PROCESSORS and PROCESSOR_AFFINITY are always set.
  # All others, are only set if the optional args were passed in
  set_tests_properties(${testName} PROPERTIES
    PROCESSORS ${numProcs}
    PROCESSOR_AFFINITY TRUE
  )

  if(ACT_LABELS) 
    set_tests_properties(${testName} PROPERTIES
      LABELS "${ACT_LABELS}"
    )
  endif()

  if(ACT_FIXTURES_SETUP) 
    set_tests_properties(${testName} PROPERTIES
      FIXTURES_SETUP "${ACT_FIXTURES_SETUP}"
    )
  endif()

  if(ACT_FIXTURES_REQUIRED) 
    set_tests_properties(${testName} PROPERTIES
      FIXTURES_REQUIRED "${ACT_FIXTURES_REQUIRED}"
    )
  endif()

  if(ACT_FIXTURES_REQUIRED_IF) 
    list(GET ACT_FIXTURES_REQUIRED_IF 0 HAS_FIXTURES_VAR)
    # Note: you need ${} since HAS_FIXTURES contains the name of the
    #       cmake var specifying if you have fixtures or not
    if(${HAS_FIXTURES_VAR})
      list(REMOVE_AT ACT_FIXTURES_REQUIRED_IF 0)
      set_tests_properties(${testName} PROPERTIES
        FIXTURES_REQUIRED ${ACT_FIXTURES_REQUIRED_IF}
      )
    endif()
  endif()

  if(ACT_WILL_FAIL) 
    set_tests_properties(${testName} PROPERTIES
      WILL_FAIL TRUE
    )
  endif()

  if(ACT_DEPENDS)
    set_tests_properties(${testName} PROPERTIES
      DEPENDS ${ACT_DEPENDS}
    )
  endif()

  if(ACT_RUN_SERIAL)
    set_tests_properties(${testName} PROPERTIES
      RUN_SERIAL TRUE
    )
  endif()

  if(ACT_DEFAULT_TIMEOUT)
    if(ALBANY_CTEST_TIMEOUT)
      set_tests_properties(${testName} PROPERTIES
        TIMEOUT ${ALBANY_CTEST_TIMEOUT}
      ) 
    endif()
  endif()
endfunction()
