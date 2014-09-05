

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testB.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testB.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testD.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testD.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests 
#  First Test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testA.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testA.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_TestA ${Albany.exe} input_fo_ismip-hom_testA.xml)
endif()

#  Second test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testC.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testC.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_TestC ${Albany.exe} input_fo_ismip-hom_testC.xml)
endif()

if (ALBANY_IFPACK2)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testAT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testAT.xml COPYONLY)
add_test(${testName}_TestA_Tpetra ${AlbanyT.exe} input_fo_ismip-hom_testAT.xml)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testCT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testCT.xml COPYONLY)
add_test(${testName}_TestC_Tpetra ${AlbanyT.exe} input_fo_ismip-hom_testCT.xml)
endif()
