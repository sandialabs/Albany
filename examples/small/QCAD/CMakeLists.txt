##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# QCAD ##################
file(COPY input_exodus DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/default_piro_params.xml
               ${CMAKE_CURRENT_BINARY_DIR}/default_piro_params.xml COPYONLY)

add_subdirectory(Poisson)
add_subdirectory(Schrodinger)
add_subdirectory(PoissonSchrodinger)

IF (ALBANY_CI)
add_subdirectory(SchrodingerCI)
add_subdirectory(PoissonSchrodingerCI)
ENDIF()



