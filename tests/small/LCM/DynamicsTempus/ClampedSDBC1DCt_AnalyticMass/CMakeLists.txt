##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material-clamped-ct.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material-clamped-ct.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material-clamped-ct-lumped.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material-clamped-ct-lumped.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped-ct.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped-ct.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped-ct-expl.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped-ct-expl.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_ct.g.4.0 
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_ct.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_ct.g.4.1 
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_ct.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_ct.g.4.2 
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_ct.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped_ct.g.4.3 
               ${CMAKE_CURRENT_BINARY_DIR}/clamped_ct.g.4.3 COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_TEMPUS)
if (ALBANY_IFPACK2)
add_test(${testName}_NewmarkImplicitDForm_CompositeTet10_AnalyticMass ${Albany.exe} clamped-ct.yaml)
set_tests_properties(${testName}_NewmarkImplicitDForm_CompositeTet10_AnalyticMass PROPERTIES LABELS "LCM;Tpetra;Forward")
add_test(${testName}_NewmarkExplicitAForm_CompositeTet10_LumpedMass ${Albany.exe} clamped-ct-expl.yaml)
set_tests_properties(${testName}_NewmarkExplicitAForm_CompositeTet10_LumpedMass PROPERTIES LABELS "LCM;Tpetra;Forward")
endif()
endif()

