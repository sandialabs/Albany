//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTri.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTri.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputQuad.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputQuad.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputExo.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputExo.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/quadQuad.exo
               ${CMAKE_CURRENT_BINARY_DIR}/quadQuad.exo COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA)
add_test(${testName}_Tri ${Albany.exe} inputTri.yaml)
set_tests_properties(${testName}_Tri PROPERTIES LABELS "LCM;Epetra;Forward")
add_test(${testName}_Quad ${Albany.exe} inputQuad.yaml)
set_tests_properties(${testName}_Quad PROPERTIES LABELS "LCM;Epetra;Forward")
add_test(${testName}_Exo ${Albany.exe} inputExo.yaml)
set_tests_properties(${testName}_Exo PROPERTIES LABELS "LCM;Epetra;Forward")
endif()

