IF (SEACAS_EPU AND SEACAS_EXODIFF)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
    SET(REF_FILE "RegHeaviside_3D.ref_intel.exo")
  ELSE()
    SET(REF_FILE "RegHeaviside_3D.ref.exo")
  ENDIF()


  # Copy mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.3 COPYONLY)

  IF (ALBANY_EPETRA) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.yaml COPYONLY)
    createExoDiffTest(${testName}_Epetra mitchell3D.yaml physics_0_mitchell3D)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF() 

  IF (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3DT.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchell3DT.yaml COPYONLY)
    createExoDiffTest(${testName}_Tpetra mitchell3DT.yaml physics_0_mitchell3DT)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  ENDIF()
ENDIF()
