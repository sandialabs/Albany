# Name the test with the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testName ${parentDirName}_${dirName})

# Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quad_tri.exo
               ${CMAKE_CURRENT_BINARY_DIR}/quad_tri.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quad_tri.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/quad_tri.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quad_tri.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/quad_tri.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quad_tri.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/quad_tri.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quad_tri.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/quad_tri.exo.4.3 COPYONLY)

####################################
###          Main tests          ###
####################################

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)

  add_test(${testName}_Tpetra ${Albany.exe} inputT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")
endif ()
