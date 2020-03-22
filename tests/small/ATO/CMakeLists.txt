##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# We often create a test using the 'runtest.cmake' script. So let's create
# a macro to make that process easier to read in a cmake file
SET (ATO_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

MACRO (createExoDiffTestWithOutFile testName testArgs exoFileName exoDiffOutFile)
  set (runtestScriptSrc ${ATO_TEST_DIR}/runtest.cmake)
  set (runtestScriptBin ${CMAKE_CURRENT_BINARY_DIR}/runtest_${testName}.cmake)

  SET(TEST_PROG           ${Albany.exe})
  SET(TEST_NAME           ${testName})
  SET(TEST_ARGS           ${testArgs})
  SET(EXO_FILE_NAME       ${exoFileName})
  SET(EXO_DIFF_FILE_NAME  ${exoDiffOutFile})
  SET(DATA_DIR            ${CMAKE_CURRENT_SOURCE_DIR})

  configure_file(${runtestScriptSrc} ${runtestScriptBin} @ONLY)

  add_test(NAME ${testName} 
           COMMAND ${CMAKE_COMMAND}
           -P ${runtestScriptBin})
ENDMACRO(createExoDiffTestWithOutFile)

MACRO (createExoDiffTest testName testArgs exoFileName)
  createExoDiffTestWithOutFile(${testName} ${testArgs} ${exoFileName} exodiff.out)
ENDMACRO(createExoDiffTest)

# Subdirs
add_subdirectory(2Matl_Homog)
add_subdirectory(BodyForce_2D)
add_subdirectory(Constraint_2D)
add_subdirectory(CurrentOpt)
add_subdirectory(DistParam_2D)
add_subdirectory(Filtered_2D)
add_subdirectory(FixedBlocks)
add_subdirectory(FixedField)
add_subdirectory(Homogenize_2D)
add_subdirectory(InvHomog_2D)
add_subdirectory(Mitchell_2D)
add_subdirectory(Mitchell_3D)
add_subdirectory(MultiLoadSet)
add_subdirectory(MultiPhys_Homogenize_2D)
add_subdirectory(RegHeaviside_2D)
IF (NOT DISABLE_LCM_EXODIFF_SENSITIVE_TESTS) 
add_subdirectory(RegHeaviside_3D)
ENDIF()
add_subdirectory(ResidualStrain)
add_subdirectory(Restart)
add_subdirectory(Thermal_Mech_2D)
add_subdirectory(Thermal_MultiMech_3D)
add_subdirectory(VolumeObj_StiffnessCon)
add_subdirectory(VolumeObj_StressCon)

IF(ENABLE_Cogent)
  MACRO (createExoDiffAlgebraTestWithOutFile testName testArgs exoFileName exoDiffOutFile)
    set (runtestScriptSrc ${ATO_TEST_DIR}/runtestAlgebra.cmake)
    set (runtestScriptBin ${CMAKE_CURRENT_BINARY_DIR}/runtestAlgebra_${testName}.cmake)

    SET(TEST_PROG           ${Albany.exe})
    SET(TEST_NAME           ${testName})
    SET(TEST_ARGS           ${testArgs})
    SET(EXO_FILE_NAME       ${exoFileName})
    SET(EXO_DIFF_FILE_NAME  ${exoDiffOutFile})
    SET(DATA_DIR            ${CMAKE_CURRENT_SOURCE_DIR})

    configure_file(${runtestScriptSrc} ${runtestScriptBin} @ONLY)

    add_test(NAME ${testName} 
             COMMAND ${CMAKE_COMMAND}
             -P ${runtestScriptBin})
  ENDMACRO(createExoDiffAlgebraTestWithOutFile)

  MACRO (createExoDiffAlgebraTest testName testArgs exoFileName)
    createExoDiffTestWithOutFile(${testName} ${testArgs} ${exoFileName} exodiff.out)
  ENDMACRO(createExoDiffAlgebraTest)

  add_subdirectory(Cogent_Tab)
  add_subdirectory(Cogent_Opt)
  add_subdirectory(Cogent_ThermoMech)
ENDIF()
