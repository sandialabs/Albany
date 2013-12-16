

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(${testName} ${Albany.exe})

# 1'. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputThyra.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputThyra.xml COPYONLY)
# 2'. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Thyra NAME)
# 3'. Create the test with this name and standard executable
add_test(${testName} ${Albany.exe} inputThyra.xml)

# 4. Repeat process for SG problems if "inputSG.xml" exists
if (ALBANY_SG_MP)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputSG.xml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSG.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputSG.xml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_SG NAME)
  add_test(${testName} ${AlbanySG.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputSG.xml)
endif ()

# 5. Repeat process for Dakota problems if "dakota.in" exists
if (ALBANY_DAKOTA)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                 ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
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
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_restart.xml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_restart.xml COPYONLY)
  if (ALBANY_DAKOTA_RESTART_EXAMPLES)
    add_test(${testName}_restart ${AlbanyDakota.exe} input_restart.xml)
    if(ALBANY_CTEST_TIMEOUT)
      set_tests_properties (${testName}_restart PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
    endif()

  endif (ALBANY_DAKOTA_RESTART_EXAMPLES)
#

endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)

# 6. Repeat process for Analysis problems if "inputAnalysis.xml" exists
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis.xml COPYONLY)

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
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis_restart.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis_restart.xml COPYONLY)

  if (ALBANY_DAKOTA_RESTART_EXAMPLES)
    add_test(${testName}_restart ${AlbanyAnalysis.exe} inputAnalysis_restart.xml)
    if(ALBANY_CTEST_TIMEOUT)
      set_tests_properties (${testName}_restart PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
    endif()
  endif (ALBANY_DAKOTA_RESTART_EXAMPLES)
#

endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml)
# This Analysis requres Dakota
endif()
