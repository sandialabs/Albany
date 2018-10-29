

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal_flip.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal_flip.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_xz.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_xz.yaml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests
#  First test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincos2D.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincos2D.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincos2DT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincos2DT.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinCosGlen ${Albany.exe} input_fo_sincos2D.yaml)
endif()
add_test(${testName}_SinCosGlen_Tpetra ${AlbanyT.exe} input_fo_sincos2DT.yaml)

#  Second test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincosz.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincosz.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincoszT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincoszT.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinCosZ3DPeriodicXY ${Albany.exe} input_fo_sincosz.yaml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_SinCosZ3DPeriodicXY PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()
endif()

add_test(${testName}_SinCosZ3DPeriodicXY_Tpetra ${AlbanyT.exe} input_fo_sincoszT.yaml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_SinCosZ3DPeriodicXY_Tpetra PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()

#  Third test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sinexp_neumann.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sinexp_neumann.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sinexp_neumannT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sinexp_neumannT.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinExp_Neumann ${Albany.exe} input_fo_sinexp_neumann.yaml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_SinExp_Neumann PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()
endif()

add_test(${testName}_SinExp_Neumann_Tpetra ${AlbanyT.exe} input_fo_sinexp_neumannT.yaml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_SinExp_Neumann_Tpetra PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()


#  Fourth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basalT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basalT.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_CosExp_Basal ${Albany.exe} input_fo_cosexp_basal.yaml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_CosExp_Basal PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()
endif()

add_test(${testName}_CosExp_Basal_Tpetra ${AlbanyT.exe} input_fo_cosexp_basalT.yaml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_CosExp_Basal_Tpetra PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()

#  Fifth test
IF(ALBANY_SEACAS)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal_all_glensLaw.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal_all_glensLaw.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal_all_glensLawT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal_all_glensLawT.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_FO_CosExp_BasalAll ${SerialAlbany.exe} input_fo_cosexp_basal_all_glensLaw.yaml)
endif()
add_test(${testName}_FO_CosExp_BasalAll_Tpetra ${SerialAlbanyT.exe}
input_fo_cosexp_basal_all_glensLawT.yaml)
ENDIF()

