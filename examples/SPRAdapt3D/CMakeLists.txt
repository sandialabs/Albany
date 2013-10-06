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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastSPRAdaptParasolid.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastSPRAdaptParasolid.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole_fmdb_14k.sms
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole_fmdb_14k.sms COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eighth_bar_hole.xmt_txt
               ${CMAKE_CURRENT_BINARY_DIR}/eighth_bar_hole.xmt_txt COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable


IF(ENABLE_PARASOLID)

  IF(NOT ALBANY_PARALLEL_ONLY)
    add_test(NAME ${testName}_SERIAL
      COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_ARGS=inputElastSPRAdaptParasolid.xml
      "-DFILENAME=eighth_bar_hole_output.pvd" -P
      ${CMAKE_CURRENT_SOURCE_DIR}/pvdwritten.cmake)
  ENDIF()
  
  
  add_test(NAME ${testName}
    COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${Albany.exe}" -DTEST_ARGS=inputElastSPRAdaptParasolid.xml
    "-DFILENAME=eighth_bar_hole_output.pvd" -P
    ${CMAKE_CURRENT_SOURCE_DIR}/pvdwritten.cmake)

ENDIF()
