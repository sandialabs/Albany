

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal_flip.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal_flip.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_xz.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_xz.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests
#  First test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincos2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincos2D.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincos2DT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincos2DT.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinCosGlen ${Albany.exe} input_fo_sincos2D.xml)
endif()
add_test(${testName}_SinCosGlen_Tpetra ${AlbanyT.exe} input_fo_sincos2DT.xml)

#  Second test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincosz.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincosz.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincoszT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincoszT.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinCosZ3DPeriodicXY ${Albany.exe} input_fo_sincosz.xml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_SinCosZ3DPeriodicXY PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()
endif()

add_test(${testName}_SinCosZ3DPeriodicXY_Tpetra ${AlbanyT.exe} input_fo_sincoszT.xml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_SinCosZ3DPeriodicXY_Tpetra PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()

#  Third test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sinexp_neumann.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sinexp_neumann.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sinexp_neumannT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sinexp_neumannT.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinExp_Neumann ${Albany.exe} input_fo_sinexp_neumann.xml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_SinExp_Neumann PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()
endif()

add_test(${testName}_SinExp_Neumann_Tpetra ${AlbanyT.exe} input_fo_sinexp_neumannT.xml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_SinExp_Neumann_Tpetra PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()


#  Fourth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basalT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basalT.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_CosExp_Basal ${Albany.exe} input_fo_cosexp_basal.xml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_CosExp_Basal PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()
endif()

add_test(${testName}_CosExp_Basal_Tpetra ${AlbanyT.exe} input_fo_cosexp_basalT.xml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_CosExp_Basal_Tpetra PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()

#  Fifth test
IF(ALBANY_SEACAS)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal_all_glensLaw.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal_all_glensLaw.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal_all_glensLawT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal_all_glensLawT.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_FO_CosExp_BasalAll ${SerialAlbany.exe} input_fo_cosexp_basal_all_glensLaw.xml)
endif()
add_test(${testName}_FO_CosExp_BasalAll_Tpetra ${SerialAlbanyT.exe}
input_fo_cosexp_basal_all_glensLawT.xml)
ENDIF()

