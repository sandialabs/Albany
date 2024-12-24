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

######################
###   HDSA tests   ###
######################

if (ALBANY_IFPACK2)
  set (testName ${testNameRoot}_HDSA_Piro)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_hdsa_piro.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_hdsa_piro.yaml)
  add_test(${testName} ${AlbanyAnalysis.exe} input_hdsa_piro.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce")

  set (testName ${testNameRoot}_HDSA_AdvDiff)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_hdsa_advDiff.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_hdsa_advDiff.yaml)
  add_test(${testName} ${AlbanyAnalysis.exe} input_hdsa_advDiff.yaml)
  set_tests_properties(${testName} PROPERTIES LABELS "LandIce")

endif()

