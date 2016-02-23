//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputLD.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputLD.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputPlasLD.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputPlasLD.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputPlasLDMueLuT64.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputPlasLDMueLuT64.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRAdaptMmodelMueLuT64.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRAdaptMmodelMueLuT64.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRAdaptMmodel.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRAdaptMmodel.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRStressAdaptMmodel1M.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRStressAdaptMmodel1M.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRStressAdaptParasolid.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRStressAdaptParasolid.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole_4_0.smb
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole_4_0.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole_4_1.smb
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole_4_1.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole_4_2.smb
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole_4_2.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole_4_3.smb
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole_4_3.smb COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole.xmt_txt
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole.xmt_txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole_mmodel.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole_mmodel.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data.perf
               ${CMAKE_CURRENT_BINARY_DIR}/data.perf COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
IF(ENABLE_MESHMODEL)
  add_test(${testName}_MMODEL_perf ${performanceTestScript})
ENDIF()

# Disable test if there isn't an entry for the current machine in data.perf

# Ignore empty tokens in "listification" of strings
CMAKE_POLICY(SET CMP0007 OLD)

FILE(READ data.perf contents)

STRING(REGEX REPLACE ";" "\\\\;" contents "${contents}")
STRING(REGEX REPLACE "\n" ";" contents "${contents}")

FOREACH(text_line ${contents})
#  message("${text_line}")
  STRING(REGEX REPLACE "   " ";" text_line "${text_line}")
  STRING(REGEX REPLACE "  " ";" text_line "${text_line}")
  STRING(REGEX REPLACE " " ";" text_line "${text_line}")
  LIST(GET text_line 0 output)
  IF("${machineName}" STREQUAL output)
    RETURN()
  ENDIF()
ENDFOREACH()

IF(ENABLE_MESHMODEL)
  set_tests_properties(${testName}_MMODEL_perf  PROPERTIES REQUIRED_FILES "machine ${machineName} not found in data.perf file")
ENDIF()


