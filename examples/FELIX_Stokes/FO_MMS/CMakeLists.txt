

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal_flip.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal_flip.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests
#  First test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincos2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincos2D.xml COPYONLY)
add_test(${testName}_SinCosGlen ${Albany.exe} input_fo_sincos2D.xml)

#  Second test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincosz.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincosz.xml COPYONLY)
add_test(${testName}_SinCosZ3DPeriodicXY ${Albany.exe} input_fo_sincosz.xml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_SinCosZ3DPeriodicXY PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()

#  Third test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sinexp_neumann.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sinexp_neumann.xml COPYONLY)
add_test(${testName}_SinExp_Neumann ${Albany.exe} input_fo_sinexp_neumann.xml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_SinExp_Neumann PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()


#  Fourth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal.xml COPYONLY)
add_test(${testName}_CosExp_Basal ${Albany.exe} input_fo_cosexp_basal.xml)
if(ALBANY_CTEST_TIMEOUT)
  set_tests_properties (${testName}_CosExp_Basal PROPERTIES TIMEOUT ${ALBANY_CTEST_TIMEOUT}) 
endif()

#  Fifth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal_all_glensLaw.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal_all_glensLaw.xml COPYONLY)
add_test(${testName}_FO_CosExp_BasalAll ${SerialAlbany.exe} input_fo_cosexp_basal_all_glensLaw.xml)


