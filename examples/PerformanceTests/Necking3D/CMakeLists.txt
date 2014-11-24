#/********************************************************************\
#*            Albany, Copyright (2010) Sandia Corporation             *
#*                                                                    *
#* Notice: This computer software was prepared by Sandia Corporation, *
#* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
#* the Department of Energy (DOE). All rights in the computer software*
#* are reserved by DOE on behalf of the United States Government and  *
#* the Contractor as provided in the Contract. You are authorized to  *
#* use this computer software for Governmental purposes but it is not *
#* to be released or distributed to the public. NEITHER THE GOVERNMENT*
#* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
#* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
#* including this sentence must appear on any copies of this software.*
#*    Questions to Andy Salinger, agsalin@sandia.gov                  *
#\********************************************************************/


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


