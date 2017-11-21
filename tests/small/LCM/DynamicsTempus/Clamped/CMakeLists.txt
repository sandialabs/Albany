##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/material-clamped-stk.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/material-clamped-stk.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/clamped-stk.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/clamped-stk.yaml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

if (ALBANY_TEMPUS)
if (ALBANY_IFPACK2) 
add_test(Dynamic_${testName}_NewmarkImplicitAForm_Tempus ${AlbanyT.exe} clamped-stk.yaml)
#add_test(Dynamic_${testName}_NewmarkImplicitAForm_Tempus_Shift ${AlbanyT.exe} clamped-stk-shift.yaml)
endif()
endif()

