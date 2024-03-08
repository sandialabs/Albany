# Name the test with the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

####################################
###         TestA  tests         ###
####################################

set(testName ${testNameRoot}_TestA)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testA.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testA.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input_fo_ismip-hom_testA.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")
endif()

####################################
###         TestB  tests         ###
####################################


set(testName ${testNameRoot}_TestB)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testB.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testB.yaml COPYONLY)

add_test(${testName} ${Albany.exe} input_fo_ismip-hom_testB.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")


####################################
###         TestC  tests         ###
####################################

set(testName ${testNameRoot}_TestC)


if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testC.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testC.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input_fo_ismip-hom_testC.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")
endif()

####################################
###         TestD  tests         ###
####################################


set(testName ${testNameRoot}_TestD)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testD.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testD.yaml COPYONLY)

add_test(${testName} ${Albany.exe} input_fo_ismip-hom_testD.yaml)
set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")
