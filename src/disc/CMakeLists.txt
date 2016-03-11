#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

IF(ALBANY_APF)
  add_subdirectory(pumi)
ENDIF()

if(ALBANY_CATALYST)
  add_subdirectory(catalyst)
endif()

IF(ALBANY_HAVE_STK)
  add_subdirectory(stk)
ENDIF()

