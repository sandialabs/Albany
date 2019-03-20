##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

IF(ALBANY_DTK)
  add_subdirectory(DTKInterp)
ENDIF()

IF(ALBANY_STK)
  IF(ALBANY_SEACAS)
    add_subdirectory(CrystalPlasticity)
    # amota: disabled because enters infinite loop
    # add_subdirectory(HMC)
IF (ALBANY_SCHWARZ)
    add_subdirectory(Schwarz)
ENDIF()
    add_subdirectory(MechanicsWithHydrogenBar)
  ENDIF()
ENDIF()
