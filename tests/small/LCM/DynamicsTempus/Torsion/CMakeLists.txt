##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nobcs.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/nobcs.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nobcs-expl.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/nobcs-expl.yaml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_TEMPUS)
if (ALBANY_IFPACK2)
add_test(Dynamic_${testName}_NewmarkImplicitAForm_Tempus ${Albany.exe} nobcs.yaml)
set_tests_properties(Dynamic_${testName}_NewmarkImplicitAForm_Tempus PROPERTIES LABELS "LCM;Tpetra;Forward")
add_test(Dynamic_${testName}_NewmarkExplicitAForm_Tempus ${Albany.exe} nobcs-expl.yaml)
set_tests_properties(Dynamic_${testName}_NewmarkImplicitAForm_Tempus PROPERTIES LABELS "LCM;Tpetra;Forward")
endif()
endif()

