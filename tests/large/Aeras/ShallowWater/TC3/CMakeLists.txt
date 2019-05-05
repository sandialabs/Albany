 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTRythmos.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTRythmos.yaml COPYONLY)
if (ALBANY_TEMPUS) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTTempus.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTTempus.yaml COPYONLY)
endif () 
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tc3_interp.ncl
#               ${CMAKE_CURRENT_BINARY_DIR}/tc3_interp.ncl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tc3_native.ncl
               ${CMAKE_CURRENT_BINARY_DIR}/tc3_native.ncl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tc3_native_satellite.ncl
               ${CMAKE_CURRENT_BINARY_DIR}/tc3_native_satellite.ncl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../grids/QUAD4/uniform_16_quad4.g
               ${CMAKE_CURRENT_BINARY_DIR}/uniform_16_quad4.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_CHECK_FPE)
IF (NOT ALBANY_STRONG_FPE_CHECK) 
add_test(Aeras_${testName}_RungeKutta4_Rythmos ${Albany.exe} inputTRythmos.yaml)
set_tests_properties(Aeras_${testName}_RungeKutta4_Rythmos PROPERTIES LABELS "Aeras;Tpetra;Forward")
if (ALBANY_TEMPUS)
add_test(Aeras_${testName}_RungeKutta4_Tempus ${Albany.exe} inputTTempus.yaml)
set_tests_properties(Aeras_${testName}_RungeKutta4_Tempus PROPERTIES LABELS "Aeras;Tpetra;Forward")
endif() 
endif() 
endif() 
