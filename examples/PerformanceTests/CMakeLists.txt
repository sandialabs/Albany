##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Get machineName for comparison with gold-standard timings in data.perf
execute_process(COMMAND hostname -s COMMAND tr -d '\n' OUTPUT_VARIABLE machineName)
message("-- Performance Test Machine Name = " ${machineName})

# Set test command that cmake will run, in subdirectories
set(performanceTestScript  
    python ${CMAKE_CURRENT_SOURCE_DIR}/perfScript.py
     -machine ${machineName}
     -executable "${AlbanyPath}")

# Heat Transfer Problems ###############
  add_subdirectory(SteadyHeat2D)
  IF(ALBANY_SEACAS)
    #add_subdirectory(SteadyHeat2DSS)
  ENDIF()

# DEMO PDES ###############
IF(ALBANY_DEMO_PDES)
  #add_subdirectory(LinComprNS)

  # These require exodus, Ioss, stk_io, to read meshes
  IF(ALBANY_SEACAS)
    add_subdirectory(NSVortexShedding)
  ENDIF()

  if (ALBANY_ASCR)
    #add_subdirectory(CoupledReactor)
  endif()

ENDIF(ALBANY_DEMO_PDES)

# LCM ###############
IF(ALBANY_LCM)
  add_subdirectory(TLPoroElasticity3D)

  IF(ALBANY_SEACAS)
    add_subdirectory(Necking3D)
  ENDIF()

  IF(ALBANY_HYDRIDE)
    #add_subdirectory(LaplaceBeltrami)
  ENDIF()

ENDIF(ALBANY_LCM)

# QCAD ##################

IF(ALBANY_QCAD)
  add_subdirectory(Poisson)
ENDIF()

# FELIX ##################

IF(ALBANY_FELIX )
  add_subdirectory(FELIX_FO_MMS)
ENDIF()

# MOR   ##################

IF(ALBANY_MOR)
  #add_subdirectory(MOR)
ENDIF()

###################
