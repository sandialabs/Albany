# Name the test with the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})


####################################
###       SinCosGlen tests       ###
####################################

set(testName ${testNameRoot}_SincCosGlen)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincos2DT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincos2DT.yaml COPYONLY)

  add_test(${testName}_Tpetra ${Albany.exe} input_fo_sincos2DT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")
endif()

####################################
###   SinCos3DPeriodicXY tests   ###
####################################

set(testName ${testNameRoot}_SincCos3DPeriodicXY)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincoszT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincoszT.yaml COPYONLY)
  add_test(${testName}_Tpetra ${Albany.exe} input_fo_sincoszT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")
  if(ALBANY_CTEST_TIMEOUT)
    set_tests_properties (${testName}_Tpetra PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
  endif()
endif()

####################################
###         SinExp tests         ###
####################################

set(testName ${testNameRoot}_SinExp_Neumann)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sinexp_neumannT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sinexp_neumannT.yaml COPYONLY)
  add_test(${testName}_Tpetra ${Albany.exe} input_fo_sinexp_neumannT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")
  if(ALBANY_CTEST_TIMEOUT)
    set_tests_properties (${testName}_Tpetra PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
  endif()
endif()

####################################
###       CosExp Basal tests     ###
####################################

set(testName ${testNameRoot}_CosExp_Basal)

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basalT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basalT.yaml COPYONLY)

  add_test(${testName}_Tpetra ${Albany.exe} input_fo_cosexp_basalT.yaml)
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")
  if(ALBANY_CTEST_TIMEOUT)
    set_tests_properties (${testName}_Tpetra PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
  endif()
endif()

####################################
###    CosExp Basal all tests    ###
####################################

IF(ALBANY_SEACAS)

  set(testName ${testNameRoot}_CosExp_BasalAll)

  if (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal_all_glensLawT.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal_all_glensLawT.yaml COPYONLY)

    add_test(${testName}_Tpetra ${SerialAlbany.exe} input_fo_cosexp_basal_all_glensLawT.yaml)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "LandIce;Tpetra;Forward")
  endif()
ENDIF()

####################################
###     Grounding Line tests     ###
####################################

IF (ALBANY_PANZER_EXPR_EVAL)
  set(testName ${testNameRoot}_FO_GLF)

  # Quads
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_glf_quad.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_glf_quad.yaml COPYONLY)

  add_test(${testName}_QUAD ${SerialAlbany.exe} input_glf_quad.yaml)
  set_tests_properties(${testName}_QUAD PROPERTIES LABELS "LandIce;Tpetra;Forward")

  # Triangles
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_glf_tri.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_glf_tri.yaml COPYONLY)

  add_test(${testName}_TRI ${SerialAlbany.exe} input_glf_tri.yaml)
  set_tests_properties(${testName}_TRI PROPERTIES LABELS "LandIce;Tpetra;Forward")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_glf_tri_edge_on_gl.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_glf_tri_edge_on_gl.yaml COPYONLY)

  # Triangles, corner case
  add_test(${testName}_TRI_CORNER_CASE ${SerialAlbany.exe} input_glf_tri_edge_on_gl.yaml)
  set_tests_properties(${testName}_TRI_CORNER_CASE PROPERTIES LABELS "LandIce;Tpetra;Forward")

  IF (ALBANY_PAMGEN)
    # Triangles, nonconst velocity
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_glf_tri_nonconst_vel.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_glf_tri_nonconst_vel.yaml COPYONLY)

    add_test(${testName}_TRI_NONCONST_VELOCITY ${SerialAlbany.exe} input_glf_tri_nonconst_vel.yaml)
    set_tests_properties(${testName}_TRI_NONCONST_VELOCITY PROPERTIES LABELS "LandIce;Tpetra;Forward")
  ENDIF()
ENDIF()

###################################################
###     Surface Mass Balance Mismatch tests     ###
###################################################

IF (ALBANY_PANZER_EXPR_EVAL)
  set(testName ${testNameRoot}_FO_SMB)

  # Quads
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_smb_quad.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_smb_quad.yaml COPYONLY)

  add_test(${testName}_QUAD ${SerialAlbany.exe} input_smb_quad.yaml)
  set_tests_properties(${testName}_QUAD PROPERTIES LABELS "LandIce;Tpetra;Forward")

  IF (ALBANY_PAMGEN)
    # Triangles, nonconst velocity
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_smb_tri.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_smb_tri.yaml COPYONLY)

    add_test(${testName}_TRI ${SerialAlbany.exe} input_smb_tri.yaml)
    set_tests_properties(${testName}_TRI PROPERTIES LABELS "LandIce;Tpetra;Forward")
  ENDIF()
ENDIF()

