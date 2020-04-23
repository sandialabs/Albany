##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
     -executable "${Albany_BINARY_DIR}/src")

set(performanceTestScript_2
    python ${CMAKE_CURRENT_SOURCE_DIR}/perfScript.py
     -machine ${machineName}_2
     -executable "${Albany_BINARY_DIR}/src")

# LANDICE ##################
IF(ALBANY_LANDICE )
  add_subdirectory(LANDICE_FO_MMS)
ENDIF()
