##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

if(SEACAS_EXODIFF AND SEACAS_ALGEBRA)
# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_ALGEBRA} ${CMAKE_CURRENT_BINARY_DIR}/algebra)
if (ALBANY_IFPACK2)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
endif()

if (ALBANY_IFPACK2)
# input files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputNoVolJ.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputNoVolJ.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputVolJ.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputVolJ.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAveragePressure.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputAveragePressure.yaml COPYONLY)
endif()

# material files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/1materialNoVolJ.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/1materialNoVolJ.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/1materialVolJ.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/1materialVolJ.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/1materialAveragePressure.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/1materialAveragePressure.yaml COPYONLY)

# mesh file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface1ElemMeshRev.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface1ElemMeshRev.g COPYONLY)

# algebra file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/alg.i
               ${CMAKE_CURRENT_BINARY_DIR}/alg.i COPYONLY)

# exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NoVolJ.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/NoVolJ.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VolJ.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/VolJ.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AveragePressure.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/AveragePressure.exodiff COPYONLY)

# gold files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/outNoVolJ.alg.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/outNoVolJ.alg.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/outVolJ.alg.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/outVolJ.alg.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/outAveragePressure.alg.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/outAveragePressure.alg.gold.e COPYONLY)

if (ALBANY_IFPACK2)
# python runtest file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.py COPYONLY)
endif()

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  if (ALBANY_IFPACK2)
    add_test(NAME ${testName} COMMAND "python" "runtest.py")
  endif()
ENDIF()
endif(SEACAS_EXODIFF AND SEACAS_ALGEBRA)
