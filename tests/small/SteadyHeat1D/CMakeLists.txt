

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input.yaml)
endif()

if (ALBANY_IFPACK2)
# 1'. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
# 2'. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3'. Create the test with this name and standard executable
add_test(${testName}_Tpetra ${AlbanyT.exe} inputT.yaml)
endif ()

# 1". Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputThyra.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputThyra.yaml COPYONLY)
# 2". Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Thyra NAME)
# 3". Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} inputThyra.yaml)
endif()

# 5. Repeat process for Dakota problems if "dakota.in" exists
if (ALBANY_DAKOTA)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                 ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_nls.data COPYONLY)
  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota NAME)
  add_test(${testName} ${AlbanyDakota.exe})

# Additional files and tests for Dakota restart testing
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dak.in
                 ${CMAKE_CURRENT_BINARY_DIR}/dak.in COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dak.restart
                 ${CMAKE_CURRENT_BINARY_DIR}/dak.restart COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_restart.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_restart.yaml COPYONLY)
  if (ALBANY_DAKOTA_RESTART_EXAMPLES)
    add_test(${testName}_restart ${AlbanyDakota.exe} input_restart.yaml)
    if(ALBANY_CTEST_TIMEOUT)
      set_tests_properties (${testName}_restart PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
    endif()

  endif (ALBANY_DAKOTA_RESTART_EXAMPLES)
#

endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)

# 6. Repeat process for Analysis problems if "inputAnalysis.yaml" exists
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis.yaml COPYONLY)

  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)

  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Analysis NAME)
  add_test(${testName} ${AlbanyAnalysis.exe})

# Additional files and tests for Dakota restart testing
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dak.in
                 ${CMAKE_CURRENT_BINARY_DIR}/dak.in COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dak.restart
                 ${CMAKE_CURRENT_BINARY_DIR}/dak.restart COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis_restart.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis_restart.yaml COPYONLY)

  if (ALBANY_DAKOTA_RESTART_EXAMPLES)
    add_test(${testName}_restart ${AlbanyAnalysis.exe} inputAnalysis_restart.yaml)
    if(ALBANY_CTEST_TIMEOUT)
      set_tests_properties (${testName}_restart PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
    endif()
  endif (ALBANY_DAKOTA_RESTART_EXAMPLES)
#

endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml)
# This Analysis requres Dakota
endif()
