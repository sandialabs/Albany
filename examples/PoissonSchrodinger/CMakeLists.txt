
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSchroPo.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSchroPo.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputCIPo.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputCIPo.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputPoissonMOSDot.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputPoissonMOSDot.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSchrodingerMOSDot.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSchrodingerMOSDot.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosDotQuantum.exo
               ${CMAKE_CURRENT_BINARY_DIR}/mosDotQuantum.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosDotQuantum.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/mosDotQuantum.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosDotQuantum.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/mosDotQuantum.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosDotQuantum.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/mosDotQuantum.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mosDotQuantum.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/mosDotQuantum.exo.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)#

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(${testName}_SchroPo ${Albany.exe} inputSchroPo.xml)

IF (ALBANY_CI)
add_test(${testName}_CIPo ${Albany.exe} inputSchroPo.xml)
ENDIF()
