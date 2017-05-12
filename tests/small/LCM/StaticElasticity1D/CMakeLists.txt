# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
               ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data 
               ${CMAKE_CURRENT_BINARY_DIR}/dakota_nls.data COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Create the test with this name and standard executable
if (ALBANY_IFPACK2)
add_test(${testName} ${AlbanyT.exe} input.yaml)
endif()

# Repeat process for Analysis problems if "inputAnalysis.yaml" exists
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml AND ALBANY_IFPACK2 AND ALBANY_DAKOTA)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis.yaml COPYONLY)

  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Analysis NAME)
  add_test(${testName} ${AlbanyAnalysisT.exe} inputAnalysis.yaml)
endif ()
