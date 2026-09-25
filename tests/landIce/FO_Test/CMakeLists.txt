# Name the test with the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

####################################
###        FO dome tests         ###
####################################

IF(ALBANY_SEACAS)
  set(testName ${testNameRoot}_Dome)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_dome.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_dome.yaml COPYONLY)

  add_test(${testName}_SDBC ${Albany.exe} input_fo_dome.yaml)
  set_tests_properties(${testName}_SDBC PROPERTIES LABELS "LandIce;Forward")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_domeAnalysis.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_domeAnalysis.yaml COPYONLY)

  add_test(${testName}_Analysis ${Albany.exe} input_fo_dome.yaml)
  set_tests_properties(${testName}_Analysis PROPERTIES LABELS "LandIce;Forward")


  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_dome_interpSurf.yaml
                ${CMAKE_CURRENT_BINARY_DIR}/input_fo_dome_interpSurf.yaml COPYONLY)

  add_test(${testName}_interpSurf ${Albany.exe} input_fo_dome_interpSurf.yaml)
  set_tests_properties(${testName}_interpSurf PROPERTIES LABELS "LandIce;Forward")
ENDIF()

####################################
###     Confined Shelf tets      ###
####################################

set(testName ${testNameRoot}_ConfinedShelf)

if (ALBANY_IFPACK2) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_confinedShelf.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_confinedShelf.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input_fo_confinedShelf.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")
endif()

set(testName ${testNameRoot}_ConfinedShelf_DepthIntBC)

if (ALBANY_IFPACK2) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_confinedShelf_depthIntBC.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_confinedShelf_depthIntBC.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input_fo_confinedShelf_depthIntBC.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")
endif()

####################################
###     Circular Shelf tets      ###
####################################

IF(ALBANY_SEACAS)
  set (testName ${testNameRoot}_CircularShelf)

  if (ALBANY_IFPACK2) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_circularShelf.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_circularShelf.yaml COPYONLY)

    add_test(${testName} ${Albany.exe} input_fo_circularShelf.yaml)
    set_tests_properties(${testName} PROPERTIES LABELS "LandIce;Forward")
  endif()
ENDIF()

####################################
###     MISMIP test case         ###
####################################

if (NOT ALBANY_PARALLEL_EXODUS)
	set (testNameDecompMesh ${testNameRoot}_MISMIP_2d_decompMesh)
  add_test (NAME ${testNameDecompMesh}
        COMMAND ${SerialSeacasDecomp.exe} -processors ${MPIMNP} mismip_2d.exo
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../AsciiMeshes/MISMIP)
  set_tests_properties (${testNameDecompMesh} PROPERTIES FIXTURES_SETUP MismipMeshSetup2d)
endif()

IF(NOT Kokkos_ENABLE_CUDA)
  set (testName ${testNameRoot}_MIMSIP_Transient_Impl)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_mismip_transient.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_mismip_transient.yaml)
  add_test(${testName} ${Albany.exe} input_mismip_transient.yaml)
  set_tests_properties(${testName}  PROPERTIES LABELS "LandIce;Forward;Implcit" FIXTURES_REQUIRED MismipMeshSetup2d)

  set (testName ${testNameRoot}_MIMSIP_Transient_2d_Impl)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_mismip_transient_2d.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_mismip_transient_2d.yaml)
  add_test(${testName} ${Albany.exe} input_mismip_transient_2d.yaml)
  set_tests_properties(${testName}  PROPERTIES LABELS "LandIce;Forward;Implcit" FIXTURES_REQUIRED MismipMeshSetup2d)

  set (testName ${testNameRoot}_MIMSIP_Transient_Expl)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_mismip_transient_expl.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_mismip_transient_expl.yaml)
  add_test(${testName} ${Albany.exe} input_mismip_transient_expl.yaml)
  set_tests_properties(${testName}  PROPERTIES LABELS "LandIce;Forward;Explicit" FIXTURES_REQUIRED MismipMeshSetup2d)
ENDIF()