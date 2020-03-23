#//*****************************************************************//
#//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#//    This Software is released under the BSD license detailed     //
#//    in the file "license.txt" in the top-level Albany directory  //
#//*****************************************************************//

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputLD.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputLD.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputPlasLD.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputPlasLD.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputPlasLDMueLuT64.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputPlasLDMueLuT64.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRAdaptMmodelMueLuT64.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRAdaptMmodelMueLuT64.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRAdaptMmodel.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRAdaptMmodel.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRStressAdaptMmodel1M.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRStressAdaptMmodel1M.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRStressAdaptParasolid.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRStressAdaptParasolid.yaml COPYONLY)
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
add_test(${testName}_perf ${performanceTestScript})

# Disable test if there isn't an entry for the current machine in data.perf

FILE(READ data.perf contents)

STRING(FIND "${contents}" "${machineName}" machinematched)

IF(${machinematched} EQUAL -1)
   set_tests_properties(${testName}_perf  PROPERTIES REQUIRED_FILES "machine ${machineName} not found in data.perf file")
ENDIF()


