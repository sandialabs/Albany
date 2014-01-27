##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

#This test is not currently appropriate for the BGQ
IF(NOT ALBANY_PARALLEL_ONLY)

set(runtest.py ${CMAKE_CURRENT_SOURCE_DIR}/runtest.py)

add_subdirectory(AnisotropicHyperelasticDamage)
add_subdirectory(AnisotropicDamage)
add_subdirectory(Gurson)
add_subdirectory(Neohookean)

ENDIF()
