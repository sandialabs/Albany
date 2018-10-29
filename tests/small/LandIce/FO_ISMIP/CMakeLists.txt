

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testB.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testB.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testD.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testD.yaml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests 
#  First Test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testA.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testA.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_TestA ${Albany.exe} input_fo_ismip-hom_testA.yaml)
endif()

#  Second test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testC.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testC.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_TestC ${Albany.exe} input_fo_ismip-hom_testC.yaml)
endif()

if (ALBANY_IFPACK2)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testAT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testAT.yaml COPYONLY)
add_test(${testName}_TestA_Tpetra ${AlbanyT.exe} input_fo_ismip-hom_testAT.yaml)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testCT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testCT.yaml COPYONLY)
add_test(${testName}_TestC_Tpetra ${AlbanyT.exe} input_fo_ismip-hom_testCT.yaml)
endif()
