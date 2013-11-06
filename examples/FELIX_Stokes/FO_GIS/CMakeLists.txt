# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_ascii.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_ascii.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_refine.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_refine.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis10km_ascii.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis10km_ascii.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis10km.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis10km.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis5km_ascii.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis5km_ascii.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis5km.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis5km.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis2km_ascii.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis2km_ascii.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis2km.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis2km.xml COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests 
#  First Test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_test.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_test.xml COPYONLY)
add_test(${testName}_Gis20km ${Albany.exe} input_fo_gis20km_test.xml)
IF(ALBANY_STK_PERCEPT)
add_test(${testName}_Gis20km_refine ${Albany8.exe} input_fo_gis20km_refine.xml)
ENDIF()

