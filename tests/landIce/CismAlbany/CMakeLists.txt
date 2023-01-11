add_subdirectory(albanyMesh)
add_subdirectory(HandsOn)
add_subdirectory(inputFiles)
add_subdirectory(mFiles)
add_subdirectory(ncGridSamples)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyPath} ${CMAKE_CURRENT_BINARY_DIR}/Albany)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CismAlbanyPath} ${CMAKE_CURRENT_BINARY_DIR}/cism_driver)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh
               ${CMAKE_CURRENT_BINARY_DIR}/run_test.sh COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/process_output_test.sh
               ${CMAKE_CURRENT_BINARY_DIR}/process_output_test.sh COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestT.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtestT.py COPYONLY)

get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testName ${parentDirName}_${dirName})

option(ENABLE_CISM_CHECK_COMPARISONS "Flag to turn on comparisons checking for CISM-Albany" OFF)
if (ENABLE_CISM_CHECK_COMPARISONS)
  message("  -- CISM_CHECK_COMPARISONS              Enabled. Error will be thrown if comparisons fail for CISM-Albany run.")
else()
  message("  -- CISM_CHECK_COMPARISONS  NOT Enabled.  Error will NOT be thrown if comparisons fail for CISM-Albany run.  Please be cautious when running ctest from CISM!")
endif()
if (ENABLE_CISM_CHECK_COMPARISONS)
	add_test(NAME LandIce_${testName} COMMAND ${PYTHON_EXECUTABLE}  "runtestT.py")
endif()
