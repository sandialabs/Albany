##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material-clamped-stk.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material-clamped-stk.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material-clamped-expl.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material-clamped-expl.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped-stk-shift.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped-stk-shift.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped-stk-bl-mesh.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped-stk-bl-mesh.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped-stk-expl.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped-stk-expl.yaml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_TEMPUS)
if (ALBANY_IFPACK2)
add_test(Dynamic_${testName}_NewmarkImplicitDForm_Tempus ${Albany.exe} clamped-stk-shift.yaml)
set_tests_properties(Dynamic_${testName}_NewmarkImplicitDForm_Tempus PROPERTIES LABELS "LCM;Tpetra;Forward")
add_test(Dynamic_${testName}_NewmarkExplicitAForm_Tempus ${Albany.exe} clamped-stk-expl.yaml)
set_tests_properties(Dynamic_${testName}_NewmarkExplicitAForm_Tempus PROPERTIES LABELS "LCM;Tpetra;Forward")
add_test(Dynamic_${testName}_NewmarkExplicitAForm_BLMesh_Tempus ${Albany.exe} clamped-stk-bl-mesh.yaml)
set_tests_properties(Dynamic_${testName}_NewmarkExplicitAForm_BLMesh_Tempus PROPERTIES LABELS "LCM;Tpetra;Forward")
endif()
endif()

