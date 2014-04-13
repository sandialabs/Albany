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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanics.xml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanics.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastAdapt.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastAdapt.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechNodeSizeField.xml
               ${CMAKE_CURRENT_BINARY_DIR}/mechNodeSizeField.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechanicsT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/mechanicsT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputElastAdaptT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputElastAdaptT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mechNodeSizeFieldT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/mechNodeSizeFieldT.xml COPYONLY)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/smooth_tension_90_symm_coarse_tets.exo
#               ${CMAKE_CURRENT_BINARY_DIR}/smooth_tension_90_symm_coarse_tets.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/smooth_tension_coarse_tet4.exo
               ${CMAKE_CURRENT_BINARY_DIR}/smooth_tension_coarse_tet4.exo COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable

#Epetra tests
#add_test(${testName}_ElasticML ${Albany.exe} inputElastAdapt.xml)
#add_test(${testName} ${Albany.exe} mechanics.xml)

#add_test(NAME ${testName}_NodeSizeField
#         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
#         -DFILE_NAME=smooth_tension_coarse_tet4_out -DTEST_ARGS=mechNodeSizeField.xml -DMPIMNP=${MPIMNP}
#         -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
#         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
#set_tests_properties(${testName}_NodeSizeField  PROPERTIES REQUIRED_FILES "${SEACAS_EPU};${SEACAS_EXODIFF}")

if (ALBANY_IFPACK2)
add_test(${testName}_ElasticML_Tpetra ${AlbanyT.exe} inputElastAdaptT.xml)
add_test(${testName}_Tpetra ${AlbanyT.exe} mechanicsT.xml)

# Not ready, need to resize NodalDataBlock, etc
#add_test(NAME ${testName}_NodeSizeField_Tpetra
#         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
#         -DFILE_NAME=smooth_tension_coarse_tet4_out -DTEST_ARGS=mechNodeSizeFieldT.xml -DMPIMNP=${MPIMNP}
#         -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
#         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
#set_tests_properties(${testName}_NodeSizeField_Tpetra  PROPERTIES REQUIRED_FILES "${SEACAS_EPU};${SEACAS_EXODIFF}")
endif()

