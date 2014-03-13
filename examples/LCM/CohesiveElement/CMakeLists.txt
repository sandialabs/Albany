##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_ALGEBRA} ${CMAKE_CURRENT_BINARY_DIR}/algebra)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyPath} ${CMAKE_CURRENT_BINARY_DIR}/Albany)


# input files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputCohesive.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputCohesive.xml COPYONLY)

# material files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/1materialCohesive.xml
               ${CMAKE_CURRENT_BINARY_DIR}/1materialCohesive.xml COPYONLY)

# mesh file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface1ElemMeshRev.g
               ${CMAKE_CURRENT_BINARY_DIR}/surface1ElemMeshRev.g COPYONLY)

# algebra file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/alg.i
               ${CMAKE_CURRENT_BINARY_DIR}/alg.i COPYONLY)

# exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Cohesive.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/Cohesive.exodiff COPYONLY)

# gold files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/outCohesive.alg.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/outCohesive.alg.gold.e COPYONLY)

# python runtest file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.py COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(NAME ${testName} COMMAND "python" "runtest.py")
  set_tests_properties(${testName}  PROPERTIES REQUIRED_FILES "${SEACAS_EXODIFF};${SEACAS_ALGEBRA}")
ENDIF()




