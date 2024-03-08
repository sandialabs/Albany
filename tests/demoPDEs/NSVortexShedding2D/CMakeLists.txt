# Name the test with the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

# Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vortex.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/vortex.exo.4.3 COPYONLY)

####################################
###         Steady tests         ###
####################################

set (testName ${testNameRoot}_Steady)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)

add_test(${testName} ${Albany.exe} input.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "Demo;Forward")


####################################
###       Transient tests        ###
####################################

set (testName ${testNameRoot}_TransientIRK)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTransientRK.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputTransientRK.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTransientRK_RegressFail.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputTransientRK_RegressFail.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} inputTransientRK.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "Demo;Forward")

  add_test(NAME ${testName}_RegressFail
           COMMAND ${Albany.exe} inputTransientRK_RegressFail.yaml)
  set_tests_properties(${testName}_RegressFail PROPERTIES WILL_FAIL TRUE)
  set_tests_properties(${testName}_RegressFail PROPERTIES LABELS "Demo;Forward;RegressFail")
endif(ALBANY_IFPACK2)
