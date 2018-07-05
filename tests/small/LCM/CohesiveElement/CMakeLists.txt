##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Create a symlink to exodiff
if (SEACAS_EXODIFF AND SEACAS_ALGEBRA)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${SEACAS_ALGEBRA} ${CMAKE_CURRENT_BINARY_DIR}/algebra)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_algebra.sh
    ${CMAKE_CURRENT_BINARY_DIR}/run_algebra.sh COPYONLY)
  if (ALBANY_EPETRA)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${AlbanyPath} ${CMAKE_CURRENT_BINARY_DIR}/Albany)
  endif()
  if (ALBANY_IFPACK2)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
  endif()

  if (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputCohesive.yaml
      ${CMAKE_CURRENT_BINARY_DIR}/inputCohesive.yaml COPYONLY)
  endif()

  # material files
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/1materialCohesive.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/1materialCohesive.yaml COPYONLY)

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
  if (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)
  endif()

  # 2. Name the test with the directory name
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # 3. Create the test with this name and standard executable
  IF(NOT ALBANY_PARALLEL_ONLY)
    if (ALBANY_IFPACK2)
      SET(OUTFILE "outCohesive.e")
      SET(REF_FILE "outCohesive.alg.gold.e")
      SET(ALG_OUTFILE "outCohesive.alg.e") 
      add_test(NAME ${testName}
               COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
               -DTEST_NAME=Cohesive -DTEST_ARGS=inputCohesive.yaml -DMPIMNP=1
               -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
               -DSEACAS_ALGEBRA=${SEACAS_ALGEBRA}
               -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${ALG_OUTFILE}
               -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)

    endif()
  ENDIF()
endif(SEACAS_EXODIFF AND SEACAS_ALGEBRA)
