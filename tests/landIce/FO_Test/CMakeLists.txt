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


########################################
###     Box mesh transient test      ###
########################################

if (ALBANY_OMEGAH)
  # Transient FO+H test on an extruded, small, structured MxN Omega_h basal box mesh, with analytic input fields.
  # Meant as a small test case for debugging/adaptivity (see also FO_GIS Humboldt transient Omega_h test).
  set (testName ${testNameRoot}_Box_Transient_Omega_h)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_box_transient_omegah.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_box_transient_omegah.yaml)
  add_test(${testName} ${Albany.exe} input_fo_box_transient_omegah.yaml)
  set_tests_properties(${testName}  PROPERTIES
    LABELS "LandIce;Forward;Omega_h")
endif()
