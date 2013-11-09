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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastAdapt.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastAdapt.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAdapt.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputAdapt.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMmodel.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMmodel.xml COPYONLY)
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

IF(NOT ALBANY_PARALLEL_ONLY AND ENABLE_PARASOLID)
  add_test(NAME ${testName}_SERIAL
     COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${SerialAlbany.exe}" -DTEST_ARGS=
     "-DFILENAME=necking-64rgn_output.pvd" -P
     ${CMAKE_CURRENT_SOURCE_DIR}/pvdwritten.cmake)
ENDIF()

IF(ENABLE_PARASOLID)
  add_test(NAME ${testName}
     COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${Albany.exe}" -DTEST_ARGS=
     "-DFILENAME=necking-64rgn_output.pvd" -P
     ${CMAKE_CURRENT_SOURCE_DIR}/pvdwritten.cmake)

  add_test(NAME ${testName}_ElastAdapt_Parasolid
     COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${Albany.exe}" -DTEST_ARGS=inputElastAdapt.xml
     "-DFILENAME=necking-64rgn_output.pvd" -P
     ${CMAKE_CURRENT_SOURCE_DIR}/pvdwritten.cmake)
ENDIF()

IF(ENABLE_MESHMODEL)
  add_test(NAME ${testName}_Mmodel
     COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${Albany.exe}" -DTEST_ARGS=inputMmodel.xml
     "-DFILENAME=necking-64rgn_output.pvd" -P
     ${CMAKE_CURRENT_SOURCE_DIR}/pvdwritten.cmake)
ENDIF()
