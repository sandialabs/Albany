##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

IF(ALBANY_DTK)
  add_subdirectory(DTKInterp)
ENDIF()

IF(ALBANY_HAVE_STK)
  IF(ALBANY_SEACAS)
    add_subdirectory(CrystalPlasticity)
    IF(ALBANY_EPETRA)
      add_subdirectory(HMC)
      add_subdirectory(Schwarz)
    ENDIF()
  ENDIF()
  IF(ALBANY_HYDRIDE)
    add_subdirectory(LaplaceBeltrami)
    IF(ALBANY_MPI)
      add_subdirectory(MultiScaleRing)
    ENDIF()
  ENDIF(ALBANY_HYDRIDE)
ENDIF()
