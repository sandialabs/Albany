##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# QCAD ##################
file(COPY input_exodus DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(Poisson)
add_subdirectory(Schrodinger)
add_subdirectory(PoissonSchrodinger)

IF (ALBANY_CI)
add_subdirectory(SchrodingerCI)
add_subdirectory(PoissonSchrodingerCI)
ENDIF()



