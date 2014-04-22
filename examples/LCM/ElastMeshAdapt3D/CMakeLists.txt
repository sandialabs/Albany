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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastAdaptT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastAdaptT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMmodelT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMmodelT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/necking-64rgn-fmdb.sms
               ${CMAKE_CURRENT_BINARY_DIR}/necking-64rgn-fmdb.sms COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/necking-64rgn-model.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/necking-64rgn-model.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/necking-gmi.sat
               ${CMAKE_CURRENT_BINARY_DIR}/necking-gmi.sat COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NeckingModel_EnglishUnits.xmt_txt
               ${CMAKE_CURRENT_BINARY_DIR}/NeckingModel_EnglishUnits.xmt_txt COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable

IF(ENABLE_PARASOLID)
  if (ALBANY_IFPACK2)
    add_test(NAME ${testName}_ElastAdapt_Tpetra
       COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${AlbanyT.exe}" -DTEST_ARGS=inputElastAdaptT.xml
       "-DFILENAME=necking-64rgn_outputT.pvd" -P
       ${CMAKE_CURRENT_SOURCE_DIR}/pvdwritten.cmake)
  endif()
ENDIF()

IF(ENABLE_MESHMODEL)
  if (ALBANY_IFPACK2)
    add_test(NAME ${testName}_Mmodel_Tpetra
       COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${AlbanyT.exe}" -DTEST_ARGS=inputMmodelT.xml
       "-DFILENAME=necking-64rgn_outputT.pvd" -P
       ${CMAKE_CURRENT_SOURCE_DIR}/pvdwritten.cmake)
  endif()
ENDIF()
