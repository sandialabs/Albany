# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_IFPACK2)
add_test(${testName} ${AlbanyT.exe} input.yaml)
endif()

# 4. Repeat process for SG problems if "inputSG.yaml" exists
# Skip, no Epetra for LCM

# 5. Repeat process for Dakota problems if "dakota.in" exists
# Skip, no Epetra for LCM

# 6. Repeat process for Analysis problems if "inputAnalysis.yaml" exists
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml AND ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis.yaml COPYONLY)

  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Analysis NAME)
  add_test(${testName} ${AlbanyAnalysisT.exe} inputAnalysis.yaml)
endif ()
