# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_STOKHOS AND ALBANY_MUELU_EXAMPLES)
  add_test(${testName} ${AlbanyT.exe} input.yaml)
    if (ALBANY_SCOREC)
      configure_file (${CMAKE_CURRENT_SOURCE_DIR}/input_pumi.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/input_pumi.yaml COPYONLY)
      add_test (${testName}_pumi ${SerialAlbanyT.exe} input_pumi.yaml)
    endif ()
endif()
