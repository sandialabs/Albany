# Name the test with the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

# Copy materials and mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mesh.exo
               ${CMAKE_CURRENT_BINARY_DIR}/mesh.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mesh.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/mesh.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mesh.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/mesh.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mesh.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/mesh.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mesh.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/mesh.exo.4.3 COPYONLY)

####################################
###          Main tests          ###
####################################

set(testName ${testNameRoot})

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Basic;Forward")
endif ()
