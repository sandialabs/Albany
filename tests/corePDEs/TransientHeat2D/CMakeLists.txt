# The root name for the tests in this directory is the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

#
# Tempus tests
#

# BE test
set(testName ${testNameRoot}_Tempus_BackwardEuler_NOXSolver)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_be_nox_solver.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/tempus_be_nox_solver.yaml COPYONLY)

add_test(${testName} ${Albany.exe} tempus_be_nox_solver.yaml)
set_tests_properties(${testName} PROPERTIES LABELS
                                            "Basic;Tempus;Tpetra;Forward")

# RK 4 test
#set(testName ${testNameRoot}_Tempus_GERK)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_gerk.yaml
#               ${CMAKE_CURRENT_BINARY_DIR}/tempus_gerk.yaml COPYONLY)

#add_test(${testName} ${Albany.exe} tempus_gerk.yaml)
#set_tests_properties(${testName} PROPERTIES LABELS
#                                            "Basic;Tempus;Tpetra;Forward")
