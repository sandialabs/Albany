##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

IF(PYALBANY_EXTREME_EVENTS)
    add_subdirectory(ExtremeEvent)
ENDIF()
add_subdirectory(IO)
add_subdirectory(SteadyHeat)
add_subdirectory(RandCompress)
