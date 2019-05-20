# Create the test with this name and standard executable
IF (SEACAS_EPU AND SEACAS_EXODIFF)
  get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # If we don't set it, createExoDiffTestWithOutFile will use ${testName}.ref.exo,
  # while we have a single ref file for all tests
  SET (EXO_REF_FILE_NAME Mitchell_3D.ref.exo)

  # Copy mesh files from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.3 COPYONLY)

  SET (testName ${testNameRoot}_oc)
  IF (ALBANY_EPETRA) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_oc.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_oc.yaml COPYONLY)
    createExoDiffTestWithOutFile(${testName}_Epetra nodal_oc.yaml physics_0_mitchell3D_oc exodiff_oc.txt)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF()

  IF (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_ocT.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_ocT.yaml COPYONLY)
    createExoDiffTestWithOutFile(${testName}_Tpetra nodal_ocT.yaml physics_0_mitchell3D_ocT exodiff_oc_dp.txt)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  ENDIF()

  SET (testName ${testNameRoot}_oc_dp)
  IF (ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_oc_dp.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_oc_dp.yaml COPYONLY)
    createExoDiffTestWithOutFile(${testName}_Epetra nodal_oc_dp.yaml physics_0_mitchell3D_oc_dp exodiff_ocT.txt)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
  ENDIF()
  IF (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_oc_dpT.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_oc_dpT.yaml COPYONLY)
    createExoDiffTestWithOutFile(${testName}_Tpetra nodal_oc_dpT.yaml physics_0_mitchell3D_oc_dpT exodiff_oc_dpT.txt)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
  ENDIF()

  IF (ATO_NLOPT)
    SET (testName ${testNameRoot}_nlopt)
    IF (ALBANY_EPETRA) 
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_nlopt.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_nlopt.yaml COPYONLY)
      add_test(${testName}_Epetra ${Albany.exe} nodal_nlopt.yaml)
      set_tests_properties(${testName}_Epetra PROPERTIES LABELS "ATO;Epetra;Forward")
    ENDIF()

    IF (ALBANY_IFPACK2)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_nloptT.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_nloptT.yaml COPYONLY)
      add_test(${testName}_Tpetra ${Albany.exe} nodal_nlopt.yaml)
      set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "ATO;Tpetra;Forward")
    ENDIF()
  ENDIF()
ENDIF()
