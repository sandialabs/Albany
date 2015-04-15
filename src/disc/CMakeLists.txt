#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# SCOREC PUMI

IF(ALBANY_SCOREC)
  add_subdirectory(pumi)
ENDIF()

if(ALBANY_CATALYST)
  add_subdirectory(catalyst)
endif()

add_subdirectory(stk)

